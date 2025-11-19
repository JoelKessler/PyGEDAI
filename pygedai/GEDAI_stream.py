"""
Real-time streaming GEDAI for continuous EEG cleaning.

The stream object encapsulates stateful threshold management behind next so
that multiple concurrent streams can operate independently.

License: PolyForm Noncommercial License 1.0.0
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union
from collections import deque

import torch
import warnings
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor

try:
    import numpy as np # type: ignore[import-not-found]
except ImportError: # pragma: no cover - optional dependency during import time
    np = None # type: ignore[assignment]

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
        processing_window_sec: Optional[float] = None,
        moving_window_chunk_sec: Optional[float] = None,
        TolX: float = 1e-1,
        maxiter: int = 500,
        max_concurrent_chunks: int = 1,
        num_workers: Optional[int] = None,
        verbose_timing: bool = False,
    ) -> None:
        """Initialise the streaming GEDAI cleaner.

        Parameters mirror the gedai function while adding streaming specific parameters.
        moving_window_chunk_sec controls how many seconds of history are concatenated
        to each chunk during cleaning. This extends the context passed to GEDAI but only
        the tail matching the new chunk is returned. The appended history is always raw EEG
        and limited so the combined window covers at most the requested duration. When combined with processing_window_sec
        the moving window must not exceed the processing window size. For accurate context
        snapshots, the chunks supplied to next should not be larger than the configured
        processing window.
        """
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

        if processing_window_sec is None:
            self.processing_window_sec: Optional[float] = None
            self._processing_window_samples: Optional[int] = None
        else:
            window_sec = float(processing_window_sec)
            if window_sec <= 0:
                raise ValueError("processing_window_sec must be positive when provided")
            window_samples = max(int(round(window_sec * self.sfreq)), 1)
            self.processing_window_sec = window_sec
            self._processing_window_samples = window_samples
        self._window_ready_chunks = deque()

        self._buffer_storage: Optional[torch.Tensor] = None
        self._buffer_count: int = 0
        self._cleaning_history_storage: Optional[torch.Tensor] = None
        self._cleaning_history_count: int = 0
        self._window_residual_storage: Optional[torch.Tensor] = None
        self._window_residual_count: int = 0
        self._window_history_storage: Optional[torch.Tensor] = None
        self._window_history_len: int = 0

        if moving_window_chunk_sec is None:
            self.moving_window_chunk_sec: Optional[float] = None
            self._moving_window_chunk_samples: int = 0
        else:
            window_sec = float(moving_window_chunk_sec)
            if window_sec <= 0:
                raise ValueError("moving_window_chunk_sec must be positive when provided")
            samples = max(int(round(window_sec * self.sfreq)), 1)
            if (
                self._processing_window_samples is not None
                and samples > self._processing_window_samples
            ):
                raise ValueError(
                    "moving_window_chunk_sec must not exceed processing_window_sec"
                )
            self.moving_window_chunk_sec = window_sec
            self._moving_window_chunk_samples = samples

        self._cleaning_history: Optional[torch.Tensor] = None

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
        self.verbose_timing = bool(verbose_timing)

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

        When a callback is provided the heavy GEDAI cleaning runs in a worker pool and the
        callback is invoked with (cleaned_chunk, chunk_index, raw_chunk) in submission order.
        In that mode the method returns None immediately once the chunk is queued. Without a
        callback the method blocks and returns the cleaned chunk.

        If moving_window_chunk_sec was configured the method prepends the requested amount
        of raw history (clipped so window plus chunk stays within the configured duration) before
        running GEDAI, but only the tail corresponding to eeg_chunk (or the assembled
        processing window when processing_window_sec is set) is returned or handed to the
        callback.
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
        self._update_cleaning_history(chunk)
        history_context = self._get_cleaning_history_context(n_samples)

        self._maybe_update_thresholds()
        if self._processing_window_samples is None:
            if callback is None:
                return self._clean_chunk(chunk, history=history_context)

            chunk_index = self._chunk_sequence
            self._chunk_sequence += 1
            self._enqueue_async_chunk(chunk, callback, chunk_index, history_context)
            return None

        self._add_chunk_to_processing_window(chunk, history_context)

        if callback is None:
            ready_entry = self._pop_ready_window_chunk()
            if ready_entry is None:
                return None
            processing_chunk, processing_history = ready_entry
            return self._clean_chunk(processing_chunk, history=processing_history)

        while True:
            ready_entry = self._pop_ready_window_chunk()
            if ready_entry is None:
                break
            processing_chunk, processing_history = ready_entry
            chunk_index = self._chunk_sequence
            self._chunk_sequence += 1
            self._enqueue_async_chunk(processing_chunk, callback, chunk_index, processing_history)

        return None

    def reset(self) -> None:
        """Cancel outstanding work and clear accumulated state while staying open."""
        self._ensure_open()
        self._shutdown_executor(cancel_futures=True)
        self._reset_internal_state(reset_channels=True)

    def close(self) -> None:
        """Dispose of the worker pool and release the cached leadfield."""
        if self._closed:
            return

        self._shutdown_executor(cancel_futures=True)
        self._reset_internal_state(reset_channels=True)
        self._leadfield = None
        self._closed = True

    def __enter__(self) -> GEDAIStream:
        """Support use as a context manager for automatic shutdown."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        """Close the stream when leaving a context manager scope."""
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        """Raise if the stream has been closed."""
        if self._closed:
            raise RuntimeError("Cannot use GEDAIStream after it has been closed")

    def _shutdown_executor(self, cancel_futures: bool) -> None:
        """Tear down the worker pool and reset async bookkeeping."""
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
        """Clear buffers and thresholds so the stream behaves like new."""
        self._buffer: Optional[torch.Tensor] = None
        self._buffer_count = 0
        if reset_channels:
            self._buffer_storage = None
        self._samples_seen: int = 0
        self._thresholds_per_band: Optional[torch.Tensor] = None
        self._lowcut_frequency_used: Optional[float] = None
        if reset_channels:
            self._n_channels: Optional[int] = None
        self._last_threshold_update_sample: int = 0
        self._initial_threshold_computed: bool = False
        self._cleaning_history = None
        self._cleaning_history_count = 0
        if reset_channels:
            self._cleaning_history_storage = None
        self._pending_callbacks.clear()
        if self._processing_window_samples is not None:
            self._window_ready_chunks = deque()
        self._window_residual_count = 0
        if reset_channels:
            self._window_residual_storage = None
        self._window_history_len = 0
        if reset_channels:
            self._window_history_storage = None
        with self._async_condition:
            self._active_async_tasks = 0
            self._threshold_update_in_progress = False
            self._async_condition.notify_all()
        self._next_callback_index = 0
        self._chunk_sequence = 0

    def _load_leadfield(self, leadfield: Union[str, torch.Tensor]) -> torch.Tensor:
        """Load the reference covariance factor from disk or accept a tensor."""
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
        """Store the expected channel count and reset dependent caches."""
        self._n_channels = n_channels
        self._buffer = None
        self._buffer_count = 0
        self._buffer_storage = torch.empty(
            (n_channels, self.buffer_max_samples), device=self.device, dtype=self.dtype
        )
        self._samples_seen = 0
        self._thresholds_per_band = None
        self._lowcut_frequency_used = None
        self._last_threshold_update_sample = 0
        self._initial_threshold_computed = False
        if self._moving_window_chunk_samples > 0:
            self._cleaning_history_storage = torch.empty(
                (n_channels, self._moving_window_chunk_samples),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._cleaning_history_storage = None
        self._cleaning_history = None
        self._cleaning_history_count = 0
        if self._processing_window_samples is not None:
            self._window_residual_storage = torch.empty(
                (n_channels, self._processing_window_samples),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._window_residual_storage = None
        self._window_residual_count = 0
        if self._moving_window_chunk_samples > 0:
            self._window_history_storage = torch.empty(
                (n_channels, self._moving_window_chunk_samples),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._window_history_storage = None
        self._window_history_len = 0

    @staticmethod
    def _append_to_sliding_buffer(
        storage: torch.Tensor,
        current_len: int,
        capacity: int,
        data: torch.Tensor,
    ) -> int:
        data_len = data.size(1)
        if data_len >= capacity:
            storage[:, :capacity] = data[:, -capacity:]
            return capacity

        available = capacity - current_len
        if data_len <= available:
            storage[:, current_len : current_len + data_len] = data
            current_len += data_len
            return current_len

        overflow = data_len - available
        if overflow > 0 and current_len > 0:
            keep = current_len - overflow
            if keep > 0:
                storage[:, :keep] = storage[:, overflow : overflow + keep]
            current_len = max(keep, 0)

        storage[:, current_len : current_len + data_len] = data
        current_len += data_len
        if current_len > capacity:
            current_len = capacity
        return current_len

    def _append_to_buffer(self, chunk: torch.Tensor) -> None:
        """Maintain the rolling threshold buffer capped by buffer_max_samples."""
        if self._buffer_storage is None:
            raise RuntimeError("Buffer storage is not initialised; call next() after channels are set")

        capacity = self.buffer_max_samples
        self._buffer_count = self._append_to_sliding_buffer(
            self._buffer_storage,
            self._buffer_count,
            capacity,
            chunk,
        )

        self._buffer = (
            self._buffer_storage[:, : self._buffer_count] if self._buffer_count > 0 else None
        )

    def _get_cleaning_history_context(self, chunk_samples: int) -> Optional[torch.Tensor]:
        """Return raw history preceding the chunk, limited by the moving window."""
        if (
            self._moving_window_chunk_samples <= chunk_samples
            or self._cleaning_history_storage is None
            or self._cleaning_history_count <= chunk_samples
        ):
            return None

        max_context = self._moving_window_chunk_samples - chunk_samples
        available = self._cleaning_history_count - chunk_samples
        context_len = min(available, max_context)
        if context_len <= 0:
            return None

        start = self._cleaning_history_count - chunk_samples - context_len
        end = start + context_len
        return self._cleaning_history_storage[:, start:end].clone()

    def _prime_processing_window_history(self, base_context: Optional[torch.Tensor]) -> None:
        """Seed the per-window history buffer with pre-chunk raw context."""
        if self._processing_window_samples is None:
            return
        if self._moving_window_chunk_samples <= 0:
            return
        if self._window_residual_count != 0:
            return
        if self._window_history_storage is None:
            return

        if base_context is None or base_context.numel() == 0:
            self._window_history_len = 0
            return

        context_len = base_context.size(1)
        max_len = min(context_len, self._moving_window_chunk_samples)
        start = context_len - max_len
        self._window_history_storage[:, :max_len] = base_context[:, start : start + max_len]
        self._window_history_len = max_len

    def _append_to_window_history_storage(self, data: torch.Tensor) -> None:
        """Extend the per-window history with new raw samples from the current chunk."""
        if self._moving_window_chunk_samples <= 0:
            return
        if self._window_history_storage is None:
            return
        self._window_history_len = self._append_to_sliding_buffer(
            self._window_history_storage,
            self._window_history_len,
            self._moving_window_chunk_samples,
            data,
        )

    def _extract_window_history_for_chunk(self, chunk_samples: int) -> Optional[torch.Tensor]:
        """Fetch the raw context immediately preceding a processing window chunk."""
        if self._moving_window_chunk_samples <= chunk_samples:
            return None
        if self._window_history_storage is None or self._window_history_len <= chunk_samples:
            return None

        max_context = self._moving_window_chunk_samples - chunk_samples
        available = self._window_history_len - chunk_samples
        context_len = min(available, max_context)
        if context_len <= 0:
            return None

        start = self._window_history_len - chunk_samples - context_len
        end = start + context_len
        return self._window_history_storage[:, start:end].clone()

    def _update_cleaning_history(self, chunk: torch.Tensor) -> None:
        """Extend the cleaning history while keeping it capped to the configured size."""
        if self._moving_window_chunk_samples <= 0 or self._cleaning_history_storage is None:
            return

        self._cleaning_history_count = self._append_to_sliding_buffer(
            self._cleaning_history_storage,
            self._cleaning_history_count,
            self._moving_window_chunk_samples,
            chunk,
        )

        self._cleaning_history = (
            self._cleaning_history_storage[:, : self._cleaning_history_count]
            if self._cleaning_history_count > 0
            else None
        )

    def _add_chunk_to_processing_window(
        self,
        chunk: torch.Tensor,
        base_context: Optional[torch.Tensor],
    ) -> None:
        """Accumulate chunks until a processing window is ready, tracking its history."""
        if self._processing_window_samples is None:
            return

        if self._window_residual_storage is None:
            raise RuntimeError("Processing window storage is not initialised; call next() after channels are set")

        target = self._processing_window_samples
        chunk_samples = chunk.size(1)
        processed = 0

        if self._window_residual_count == 0 and base_context is not None:
            self._prime_processing_window_history(base_context)
            base_context = None

        while processed < chunk_samples:
            needed = target - self._window_residual_count
            take = min(needed, chunk_samples - processed)
            start = self._window_residual_count
            end = start + take
            self._window_residual_storage[:, start:end] = chunk[:, processed : processed + take]
            self._append_to_window_history_storage(chunk[:, processed : processed + take])
            self._window_residual_count += take
            processed += take

            if self._window_residual_count == target:
                ready_chunk = self._window_residual_storage[:, :target].clone()
                ready_history = self._extract_window_history_for_chunk(target)
                self._window_ready_chunks.append((ready_chunk, ready_history))
                self._window_residual_count = 0

    def _pop_ready_window_chunk(self) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Return the next assembled processing window and its history if ready."""
        if not self._window_ready_chunks:
            return None
        return self._window_ready_chunks.popleft()

    def _ensure_executor(self) -> None:
        """Lazily create the worker pool used for asynchronous cleaning."""
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
        history: Optional[torch.Tensor],
    ) -> None:
        """Queue chunk cleaning work, preserving submission order for callbacks."""
        self._ensure_executor()
        if self._executor is None:
            raise RuntimeError("Async executor is not available")

        chunk_to_process = chunk.detach().clone().contiguous()
        history_to_process = history.detach().clone() if history is not None else None

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
        if history_to_process is not None:
            history_to_process = history_to_process.to(device=self.device, dtype=self.dtype)

        try:
            future = self._executor.submit(
                self._process_chunk_async,
                chunk_to_process,
                thresholds_copy,
                lowcut_used,
                history_to_process,
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
        history: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clean a chunk in a worker thread while keeping the original chunk."""
        cleaned = self._clean_chunk(
            chunk,
            thresholds_per_band=thresholds_per_band,
            lowcut_frequency_used=lowcut_frequency_used,
            history=history,
        )
        return chunk, cleaned

    def _handle_async_result(
        self,
        future,
        chunk_index: int,
        callback: Optional[CallbackType],
    ) -> None:
        """Release resources and dispatch callbacks once a chunk finishes cleaning."""
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
        """Recompute thresholds when the delay or periodic cadence requires it."""
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
                verbose_timing=self.verbose_timing,
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
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invoke GEDAI using cached thresholds and optional history context."""
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

        window_input = chunk
        chunk_samples = chunk.size(1)
        if history is not None and history.numel() > 0:
            history = history.to(device=self.device, dtype=self.dtype)
            window_input = torch.cat([history, chunk], dim=1).contiguous()

        try:
            thresholds_for_run = thresholds.to(device=self.device, dtype=self.dtype)
            # Reuse existing thresholds to avoid recomputing the optimizer for every chunk.
            cleaned_window = gedai(
                window_input,
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
                verbose_timing=self.verbose_timing,
            )
            if history is not None and history.numel() > 0:
                return cleaned_window[:, -chunk_samples:].contiguous()
            return cleaned_window
        except Exception as exc:
            warnings.warn(f"Cleaning failed: {exc}. Returning unprocessed chunk.")
            return chunk

    @property
    def state(self) -> dict:
        """Return a dictionary snapshot of the mutable stream state."""
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
            "processing_window_sec": self.processing_window_sec,
            "processing_window_samples": self._processing_window_samples,
            "moving_window_chunk_sec": self.moving_window_chunk_sec,
            "moving_window_chunk_samples": self._moving_window_chunk_samples,
            "cleaning_history_samples": self._cleaning_history_count,
            "pending_async_callbacks": len(self._pending_callbacks),
            "next_callback_index": self._next_callback_index,
            "verbose_timing": self.verbose_timing,
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
    processing_window_sec: Optional[float] = None,
    moving_window_chunk_sec: Optional[float] = None,
    TolX: float = 1e-1,
    maxiter: int = 500,
    max_concurrent_chunks: int = 1,
    num_workers: Optional[int] = None,
    verbose_timing: bool = False,
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
        processing_window_sec=processing_window_sec,
        moving_window_chunk_sec=moving_window_chunk_sec,
        TolX=TolX,
        maxiter=maxiter,
        max_concurrent_chunks=max_concurrent_chunks,
        num_workers=num_workers,
        verbose_timing=verbose_timing,
    )
