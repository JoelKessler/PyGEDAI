"""
Real-time streaming GEDAI for continuous EEG cleaning.

The stream object encapsulates stateful threshold management behind next so
that multiple concurrent streams can operate independently.

License: PolyForm Noncommercial License 1.0.0
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import warnings

from .GEDAI import gedai


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

        # Load the reference covariance once and reuse it across incoming chunks.
        self._leadfield = self._load_leadfield(leadfield)
        self._closed = False
        self._reset_internal_state(reset_channels=True)

    def next(self, eeg_chunk: torch.Tensor) -> torch.Tensor:
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
        return self._clean_chunk(chunk)

    def reset(self) -> None:
        self._ensure_open()
        self._reset_internal_state(reset_channels=True)

    def close(self) -> None:
        if self._closed:
            return

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

    def _reset_internal_state(self, reset_channels: bool) -> None:
        self._buffer: Optional[torch.Tensor] = None
        self._samples_seen: int = 0
        self._thresholds_per_band: Optional[torch.Tensor] = None
        self._lowcut_frequency_used: Optional[float] = None
        if reset_channels:
            self._n_channels: Optional[int] = None
        self._last_threshold_update_sample: int = 0
        self._initial_threshold_computed: bool = False

    def _load_leadfield(self, leadfield: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(leadfield, torch.Tensor):
            tensor = leadfield.to(device=self.device, dtype=self.dtype)
        else:
            try:
                tensor = torch.load(leadfield).to(device=self.device, dtype=self.dtype)
            except Exception:
                import numpy as np

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
            return

        self._thresholds_per_band = result["artifact_threshold_per_band"].detach().to(
            device=self.device
        ).clone()
        self._lowcut_frequency_used = float(result["lowcut_frequency_used"])

        self._initial_threshold_computed = True
        self._last_threshold_update_sample = self._samples_seen

        message = "Initial" if not was_computed else "Periodic"
        print(f"GEDAI Stream: {message} thresholds computed at {self._samples_seen / self.sfreq:.1f}s")

    def _clean_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if not self._initial_threshold_computed or self._thresholds_per_band is None:
            return chunk

        cleaning_lowcut = (
            self._lowcut_frequency_used if self._lowcut_frequency_used is not None else self.lowcut_frequency
        )

        try:
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
                artifact_thresholds_override=self._thresholds_per_band,
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
    )
