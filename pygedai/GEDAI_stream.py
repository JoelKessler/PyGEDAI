"""
Real-time streaming GEDAI for continuous EEG cleaning.

The stream object encapsulates stateful threshold management behind next so
that multiple concurrent streams can operate independently.

License: PolyForm Noncommercial License 1.0.0
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import warnings
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency during import time
    np = None  # type: ignore[assignment]

from .GEDAI import gedai

CallbackType = Callable[[torch.Tensor, int, torch.Tensor], None]

class GEDAIStream:
    """Stateful GEDAI stream exposing next to clean incoming EEG chunks."""

    def __init__(
        self,
        sfreq: float = 250.0,
        leadfield: Union[str, torch.Tensor, None] = None,
        threshold_update_interval_sec: float = 300.0,
        initial_threshold_delay_sec: float = 60.0,
        denoising_strength: str = "auto",
        epoch_size_in_cycles: float = 12.0,
        lowcut_frequency: float = 0.5,
        wavelet_levels: Optional[int] = 9,
        matlab_levels: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        buffer_max_sec: float = 600.0,
        TolX: float = 1e-1,
        maxiter: int = 500,
        max_concurrent_chunks: int = 1,
        num_workers: Optional[int] = None,
    ) -> None:
        if leadfield is None:
            raise ValueError("leadfield is required to initialize GEDAIStream")

        # Cache runtime options so threshold updates and cleaning share the same configuration.
        self.sfreq = float(sfreq)
        self.denoising_strength = denoising_strength
        self.epoch_size_in_cycles = epoch_size_in_cycles
        self.lowcut_frequency = lowcut_frequency
        self.wavelet_levels = wavelet_levels
        self.matlab_levels = matlab_levels
        self.device = torch.device(device)
        self.dtype = dtype
        self.TolX = TolX
        self.maxiter = maxiter

        self.threshold_update_interval_sec = float(threshold_update_interval_sec)
        self.initial_threshold_delay_sec = float(initial_threshold_delay_sec)
        self.buffer_max_sec = float(buffer_max_sec)

        self.threshold_update_interval_samples = max(
            int(round(self.threshold_update_interval_sec * self.sfreq)), 1
        )
        self.initial_threshold_delay_samples = max(
            int(round(self.initial_threshold_delay_sec * self.sfreq)), 0
        )
        self.buffer_max_samples = max(int(round(self.buffer_max_sec * self.sfreq)), 1)

        max_concurrent_chunks_int = int(max_concurrent_chunks)
        if max_concurrent_chunks_int == -1:
            self.max_concurrent_chunks = -1
        elif max_concurrent_chunks_int >= 1:
            self.max_concurrent_chunks = max_concurrent_chunks_int
        else:
            raise ValueError("max_concurrent_chunks must be -1 or a positive integer")

        if num_workers is not None:
            num_workers_int = int(num_workers)
            if num_workers_int < 1:
                raise ValueError("num_workers must be at least 1 when provided")
            self._num_workers: Optional[int] = num_workers_int
        else:
            if self.max_concurrent_chunks == -1:
                # Defer to ThreadPoolExecutor's default worker heuristic when no explicit cap is supplied.
                self._num_workers = None
            else:
                self._num_workers = self.max_concurrent_chunks

        self._executor: Optional[ThreadPoolExecutor] = None
        self._semaphore: Optional[threading.Semaphore] = None
        self._order_lock = threading.Lock()
        self._pending_callbacks: Dict[
            int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[CallbackType]]
        ] = {}
        self._async_lock = threading.Lock()
        self._async_condition = threading.Condition(self._async_lock)
        self._threshold_update_in_progress = False
        self._active_async_tasks = 0
        self._next_callback_index = 0
        self._chunk_sequence = 0

        # Load the reference covariance once and reuse it across incoming chunks.
        self._leadfield = self._load_leadfield(leadfield)
        self._closed = False
        self._reset_internal_state(reset_channels=True)

    def next(
        self,
        eeg_chunk: torch.Tensor,
        callback: Optional[CallbackType] = None,
    ) -> Optional[torch.Tensor]:
        """Clean the next EEG chunk, optionally dispatching results asynchronously.

        When callback is provided the heavy GEDAI cleaning runs in a worker pool and the
        callback is invoked with (cleaned_chunk, chunk_index, raw_chunk) in submission order.
        In that mode the method returns None immediately once the chunk is queued. Without a
        callback the method blocks and returns the cleaned chunk, preserving the pre-existing
        synchronous behaviour.
        """
        self._ensure_open()

        if eeg_chunk.ndim != 2:
            raise ValueError("eeg_chunk must be 2D (n_channels, n_samples)")

        chunk = eeg_chunk.to(device=self.device, dtype=self.dtype)
        n_channels, n_samples = chunk.shape
        if n_samples == 0:
            raise ValueError("eeg_chunk must contain at least one sample")

        if self._n_channels is None:
            if self._leadfield.shape != (n_channels, n_channels):
                raise ValueError(
                    f"leadfield shape must be ({n_channels}, {n_channels}); got {self._leadfield.shape}"
                )
            self._initialize_channels(n_channels)
        elif self._n_channels != n_channels:
            raise ValueError(
                f"Chunk channel count ({n_channels}) does not match initialized stream ({self._n_channels})"
            )

        self._append_to_buffer(chunk)
        self._samples_seen += n_samples

        self._maybe_update_thresholds()
        if callback is None:
            return self._clean_chunk(chunk)

        chunk_index = self._chunk_sequence
        self._chunk_sequence += 1
        self._enqueue_async_chunk(chunk, callback, chunk_index)
        return None

    def reset(self) -> None:
        self._ensure_open()
        self._shutdown_executor(cancel_futures=True)
        self._reset_internal_state(reset_channels=True)

    def close(self) -> None:
        if self._closed:
            return

        self._shutdown_executor(cancel_futures=True)
        self._reset_internal_state(reset_channels=True)
        self._leadfield = None
        self._closed = True

    def __enter__(self) -> GEDAIStream:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot use GEDAIStream after it has been closed")

    def _shutdown_executor(self, cancel_futures: bool) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=cancel_futures)
            self._executor = None
        self._semaphore = None
        with self._async_condition:
            self._active_async_tasks = 0
            self._threshold_update_in_progress = False
            self._async_condition.notify_all()

    def _reset_internal_state(self, reset_channels: bool) -> None:
        self._buffer: Optional[torch.Tensor] = None
        self._samples_seen: int = 0
        self._thresholds_per_band: Optional[torch.Tensor] = None
        self._lowcut_frequency_used: Optional[float] = None
        if reset_channels:
            self._n_channels: Optional[int] = None
        self._last_threshold_update_sample: int = 0
        self._initial_threshold_computed: bool = False
        self._pending_callbacks.clear()
        with self._async_condition:
            self._active_async_tasks = 0
            self._threshold_update_in_progress = False
            self._async_condition.notify_all()
        self._next_callback_index = 0
        self._chunk_sequence = 0

    def _load_leadfield(self, leadfield: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(leadfield, torch.Tensor):
            tensor = leadfield.to(device=self.device, dtype=self.dtype)
        else:
            try:
                tensor = torch.load(leadfield).to(device=self.device, dtype=self.dtype)
            except Exception:
                if np is None:
                    raise ImportError(
                        "numpy is required to load leadfield tensors from disk"
                    ) from None
                loaded = np.load(leadfield)
                tensor = torch.as_tensor(loaded, device=self.device, dtype=self.dtype)

        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("leadfield (refCov = L @ L.T) must be a square matrix")

        return tensor.contiguous()

    def _initialize_channels(self, n_channels: int) -> None:
        self._n_channels = n_channels
        self._buffer = None
        self._samples_seen = 0
        self._thresholds_per_band = None
        self._lowcut_frequency_used = None
        self._last_threshold_update_sample = 0
        self._initial_threshold_computed = False

    def _append_to_buffer(self, chunk: torch.Tensor) -> None:
        # Maintain a rolling buffer capped by buffer_max_samples to drive threshold updates.
        chunk_for_buffer = chunk.detach().clone()
        if self._buffer is None:
            self._buffer = chunk_for_buffer
        else:
            self._buffer = torch.cat([self._buffer, chunk_for_buffer], dim=1).contiguous()

        if self._buffer.size(1) > self.buffer_max_samples:
            excess = self._buffer.size(1) - self.buffer_max_samples
            self._buffer = self._buffer[:, excess:]

    def _ensure_executor(self) -> None:
        if self._executor is None:
            if self._num_workers is None:
                self._executor = ThreadPoolExecutor()
            else:
                self._executor = ThreadPoolExecutor(max_workers=self._num_workers)
            if self.max_concurrent_chunks == -1:
                self._semaphore = None
            else:
                self._semaphore = threading.Semaphore(self.max_concurrent_chunks)

    def _enqueue_async_chunk(
        self,
        chunk: torch.Tensor,
        callback: Optional[CallbackType],
        chunk_index: int,
    ) -> None:
        self._ensure_executor()
        if self._executor is None:
            raise RuntimeError("Async executor is not available")

        chunk_to_process = chunk.detach().clone().contiguous()

        semaphore = self._semaphore
        if semaphore is not None:
            semaphore.acquire()

        with self._async_condition:
            while self._threshold_update_in_progress:
                self._async_condition.wait()
            self._active_async_tasks += 1
            thresholds_copy = (
                self._thresholds_per_band.detach().clone()
                if self._thresholds_per_band is not None
                else None
            )
            lowcut_used = self._lowcut_frequency_used

        if thresholds_copy is not None:
            thresholds_copy = thresholds_copy.to(device=self.device, dtype=self.dtype)

        try:
            future = self._executor.submit(
                self._process_chunk_async,
                chunk_to_process,
                thresholds_copy,
                lowcut_used,
            )
        except Exception:
            if semaphore is not None:
                semaphore.release()
            self._finish_async_task()
            raise

        future.add_done_callback(
            lambda fut, idx=chunk_index, cb=callback: self._handle_async_result(fut, idx, cb)
        )

    def _finish_async_task(self) -> None:
        with self._async_condition:
            if self._active_async_tasks > 0:
                self._active_async_tasks -= 1
            if self._threshold_update_in_progress and self._active_async_tasks == 0:
                self._async_condition.notify_all()

    def _process_chunk_async(
        self,
        chunk: torch.Tensor,
        thresholds_per_band: Optional[torch.Tensor],
        lowcut_frequency_used: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cleaned = self._clean_chunk(
            chunk,
            thresholds_per_band=thresholds_per_band,
            lowcut_frequency_used=lowcut_frequency_used,
        )
        return chunk, cleaned

    def _handle_async_result(
        self,
        future,
        chunk_index: int,
        callback: Optional[CallbackType],
    ) -> None:
        try:
            original, cleaned = future.result()
        except CancelledError:
            if self._semaphore is not None:
                self._semaphore.release()
            self._finish_async_task()
            return
        except Exception as exc:
            warnings.warn(f"Cleaning failed: {exc}. Returning unprocessed chunk.")
            original = cleaned = None

        if cleaned is None and original is not None:
            cleaned = original

        ready_callbacks: list[Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor], Optional[CallbackType]]] = []
        with self._order_lock:
            self._pending_callbacks[chunk_index] = (cleaned, original, callback)
            while self._next_callback_index in self._pending_callbacks:
                stored_cleaned, stored_original, stored_callback = self._pending_callbacks.pop(
                    self._next_callback_index
                )
                ready_callbacks.append(
                    (self._next_callback_index, stored_cleaned, stored_original, stored_callback)
                )
                self._next_callback_index += 1

        if self._semaphore is not None:
            self._semaphore.release()

        for idx, cleaned_chunk, original_chunk, cb in ready_callbacks:
            if cb is not None and cleaned_chunk is not None and original_chunk is not None:
                try:
                    cb(cleaned_chunk, idx, original_chunk)
                except Exception as cb_exc:
                    warnings.warn(f"Callback raised an exception: {cb_exc}")

        self._finish_async_task()

    def _maybe_update_thresholds(self) -> None:
        if self._buffer is None:
            return

        should_update = False
        if not self._initial_threshold_computed:
            if self._samples_seen >= self.initial_threshold_delay_samples:
                should_update = True
        else:
            samples_since_update = self._samples_seen - self._last_threshold_update_sample
            if samples_since_update >= self.threshold_update_interval_samples:
                should_update = True

        if not should_update:
            return

        # Block new async tasks and wait for in-flight cleanings to finish before recomputing thresholds.
        with self._async_condition:
            while self._threshold_update_in_progress:
                self._async_condition.wait()
            self._threshold_update_in_progress = True
            while self._active_async_tasks > 0:
                self._async_condition.wait()

        # Run GEDAI on the accumulated buffer to refresh thresholds while preserving the buffer contents.
        was_computed = self._initial_threshold_computed
        try:
            result = gedai(
                self._buffer,
                sfreq=self.sfreq,
                denoising_strength=self.denoising_strength,
                leadfield=self._leadfield,
                epoch_size_in_cycles=self.epoch_size_in_cycles,
                lowcut_frequency=self.lowcut_frequency,
                wavelet_levels=self.wavelet_levels,
                matlab_levels=self.matlab_levels,
                device=self.device,
                dtype=self.dtype,
                TolX=self.TolX,
                maxiter=self.maxiter,
                skip_checks_and_return_cleaned_only=False,
            )
        except Exception as exc:
            warnings.warn(f"Threshold computation failed: {exc}. Using previous thresholds.")
            with self._async_condition:
                self._threshold_update_in_progress = False
                self._async_condition.notify_all()
            return

        self._thresholds_per_band = result["artifact_threshold_per_band"].detach().to(
            device=self.device
        ).clone()
        self._lowcut_frequency_used = float(result["lowcut_frequency_used"])

        self._initial_threshold_computed = True
        self._last_threshold_update_sample = self._samples_seen

        message = "Initial" if not was_computed else "Periodic"
        print(f"GEDAI Stream: {message} thresholds computed at {self._samples_seen / self.sfreq:.1f}s")

        with self._async_condition:
            self._threshold_update_in_progress = False
            self._async_condition.notify_all()

    def _clean_chunk(
        self,
        chunk: torch.Tensor,
        thresholds_per_band: Optional[torch.Tensor] = None,
        lowcut_frequency_used: Optional[float] = None,
    ) -> torch.Tensor:
        thresholds = thresholds_per_band
        if thresholds is None:
            if not self._initial_threshold_computed or self._thresholds_per_band is None:
                return chunk
            thresholds = self._thresholds_per_band

        cleaning_lowcut = (
            lowcut_frequency_used
            if lowcut_frequency_used is not None
            else (
                self._lowcut_frequency_used
                if self._lowcut_frequency_used is not None
                else self.lowcut_frequency
            )
        )

        try:
            thresholds_for_run = thresholds.to(device=self.device, dtype=self.dtype)
            # Reuse existing thresholds to avoid recomputing the optimizer for every chunk.
            return gedai(
                chunk,
                sfreq=self.sfreq,
                denoising_strength=self.denoising_strength,
                leadfield=self._leadfield,
                epoch_size_in_cycles=self.epoch_size_in_cycles,
                lowcut_frequency=cleaning_lowcut,
                wavelet_levels=self.wavelet_levels,
                matlab_levels=self.matlab_levels,
                device=self.device,
                dtype=self.dtype,
                TolX=self.TolX,
                maxiter=self.maxiter,
                skip_checks_and_return_cleaned_only=True,
                artifact_thresholds_override=thresholds_for_run,
            )
        except Exception as exc:
            warnings.warn(f"Cleaning failed: {exc}. Returning unprocessed chunk.")
            return chunk

    @property
    def state(self) -> dict:
        # Expose the mutable pieces so callers can snapshot or debug the stream state.
        return {
            "buffer": self._buffer,
            "samples_seen": self._samples_seen,
            "thresholds_per_band": self._thresholds_per_band,
            "lowcut_frequency_used": self._lowcut_frequency_used,
            "refCOV": self._leadfield,
            "n_channels": self._n_channels,
            "last_threshold_update_sample": self._last_threshold_update_sample,
            "initial_threshold_computed": self._initial_threshold_computed,
            "max_concurrent_chunks": self.max_concurrent_chunks,
            "num_workers": self._num_workers,
            "pending_async_callbacks": len(self._pending_callbacks),
            "next_callback_index": self._next_callback_index,
        }

def gedai_stream(
    sfreq: float = 250.0,
    leadfield: Union[str, torch.Tensor, None] = None,
    threshold_update_interval_sec: float = 300.0,
    initial_threshold_delay_sec: float = 60.0,
    denoising_strength: str = "auto",
    epoch_size_in_cycles: float = 12.0,
    lowcut_frequency: float = 0.5,
    wavelet_levels: Optional[int] = 9,
    matlab_levels: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    buffer_max_sec: float = 600.0,
    TolX: float = 1e-1,
    maxiter: int = 500,
    max_concurrent_chunks: int = 1,
    num_workers: Optional[int] = None,
) -> GEDAIStream:
    """Factory returning a configured GEDAIStream instance."""

    return GEDAIStream(
        sfreq=sfreq,
        leadfield=leadfield,
        threshold_update_interval_sec=threshold_update_interval_sec,
        initial_threshold_delay_sec=initial_threshold_delay_sec,
        denoising_strength=denoising_strength,
        epoch_size_in_cycles=epoch_size_in_cycles,
        lowcut_frequency=lowcut_frequency,
        wavelet_levels=wavelet_levels,
        matlab_levels=matlab_levels,
        device=device,
        dtype=dtype,
        buffer_max_sec=buffer_max_sec,
        TolX=TolX,
        maxiter=maxiter,
        max_concurrent_chunks=max_concurrent_chunks,
        num_workers=num_workers,
    )
