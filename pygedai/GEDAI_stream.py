"""
Real-time streaming GEDAI for continuous EEG cleaning.

This module provides a stateful streaming interface that:
1. Accumulates incoming EEG chunks
2. Periodically recomputes artifact thresholds
3. Applies cleaning continuously with cached thresholds

License: PolyForm Noncommercial License 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import warnings

from .GEDAI import gedai

@dataclass
class GEDAIStreamState:
    """State maintained across streaming calls."""
    buffer: Optional[torch.Tensor] = None
    samples_seen: int = 0

    thresholds_per_band: Optional[torch.Tensor] = None
    lowcut_frequency_used: Optional[float] = None
    refCOV: Optional[torch.Tensor] = None
    leadfield: Optional[torch.Tensor] = None

    last_threshold_update_sample: int = 0
    initial_threshold_computed: bool = False

    sfreq: float = 250.0
    n_channels: int = 0
    threshold_update_interval_samples: int = 0
    initial_threshold_delay_samples: int = 0

    denoising_strength: str = "auto"
    epoch_size_in_cycles: float = 12.0
    lowcut_frequency: float = 0.5
    wavelet_levels: Optional[int] = 9
    matlab_levels: Optional[int] = None
    device: Union[str, torch.device] = "cpu"
    dtype: torch.dtype = torch.float32
    TolX: float = 1e-1
    maxiter: int = 500

def gedai_stream(
    eeg_chunk: torch.Tensor,
    state: Optional[GEDAIStreamState] = None,
    *,
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
) -> Tuple[torch.Tensor, GEDAIStreamState]:
    """
    Stream-based GEDAI cleaning for real-time EEG processing.

    Parameters
    - eeg_chunk: Incoming EEG chunk, shape (n_channels, n_samples).
    - state: State from previous call. If None, initializes new stream.
    - sfreq: Sampling frequency in Hz.
    - leadfield: Leadfield covariance matrix or path to load it (required for first call).
    - threshold_update_interval_sec: How often to recompute artifact thresholds (seconds). Default: 300.0 (5 min).
    - initial_threshold_delay_sec: Wait this long before computing first threshold (seconds). Default: 60.0.
    - denoising_strength: Denoising strength passed to GEDAI ("auto-", "auto", or "auto+").
    - epoch_size_in_cycles: Number of wave cycles per epoch for each band.
    - lowcut_frequency: Exclude bands with upper frequency <= this value (Hz).
    - wavelet_levels: Number of wavelet decomposition levels (ignored when matlab_levels is provided).
    - matlab_levels: MATLAB-style level selection (overrides wavelet_levels when provided).
    - device: Computation device.
    - dtype: Data type for computation.
    - buffer_max_sec: Maximum buffer duration (seconds). Older data is discarded. Default: 600.0.
    - TolX: Optimization tolerance forwarded to GEDAI.
    - maxiter: Maximum iterations forwarded to GEDAI.

    Returns
    cleaned_chunk: Cleaned EEG for the current chunk (same shape as input).
    state: Updated state to pass to next call.
    """

    if eeg_chunk.ndim != 2:
        raise ValueError("eeg_chunk must be 2D (n_channels, n_samples)")

    chunk = eeg_chunk.to(device=device, dtype=dtype)
    n_channels, n_samples_chunk = chunk.shape
    if n_samples_chunk == 0:
        raise ValueError("eeg_chunk must contain at least one sample")

    if state is None:
        if leadfield is None:
            raise ValueError("leadfield must be provided on first call")

        threshold_update_samples = int(round(threshold_update_interval_sec * sfreq))
        initial_delay_samples = int(round(initial_threshold_delay_sec * sfreq))
        buffer_max_samples = int(round(buffer_max_sec * sfreq))
        if buffer_max_samples <= 0:
            raise ValueError("buffer_max_sec must correspond to at least one sample")

        device_obj = torch.device(device)
        if isinstance(leadfield, torch.Tensor):
            leadfield_tensor = leadfield.to(device=device_obj, dtype=dtype)
        else:
            try:
                leadfield_tensor = torch.load(leadfield).to(device=device_obj, dtype=dtype)
            except Exception:
                import numpy as np # Local import to avoid mandatory dependency at module load

                loaded = np.load(leadfield)
                leadfield_tensor = torch.as_tensor(loaded, device=device_obj, dtype=dtype)

        if leadfield_tensor.shape != (n_channels, n_channels):
            raise ValueError(f"leadfield shape must be ({n_channels}, {n_channels})")

        state = GEDAIStreamState(
            buffer=chunk.detach().clone(),
            samples_seen=0,
            leadfield=leadfield_tensor,
            sfreq=sfreq,
            n_channels=n_channels,
            threshold_update_interval_samples=max(threshold_update_samples, 1),
            initial_threshold_delay_samples=max(initial_delay_samples, 0),
            denoising_strength=denoising_strength,
            epoch_size_in_cycles=epoch_size_in_cycles,
            lowcut_frequency=lowcut_frequency,
            wavelet_levels=wavelet_levels,
            matlab_levels=matlab_levels,
            device=device_obj,
            dtype=dtype,
            TolX=TolX,
            maxiter=maxiter,
        )
        state.refCOV = None
        buffer_max_samples = int(round(buffer_max_sec * sfreq))
    else:
        if state.n_channels != n_channels:
            raise ValueError(
                f"Chunk channel count ({n_channels}) does not match initialized stream ({state.n_channels})"
            )
        buffer_max_samples = int(round(buffer_max_sec * state.sfreq))
        if buffer_max_samples <= 0:
            raise ValueError("buffer_max_sec must correspond to at least one sample")

        chunk = chunk.to(device=state.device, dtype=state.dtype)
        if leadfield is not None and state.leadfield is not None:
            if isinstance(leadfield, torch.Tensor):
                candidate = leadfield.to(device=state.device, dtype=state.dtype)
            else:
                try:
                    candidate = torch.load(leadfield).to(device=state.device, dtype=state.dtype)
                except Exception:
                    import numpy as np

                    loaded = np.load(leadfield)
                    candidate = torch.as_tensor(loaded, device=state.device, dtype=state.dtype)
            if candidate.shape != state.leadfield.shape or not torch.allclose(candidate, state.leadfield):
                raise ValueError("leadfield provided does not match the one stored in the stream state")
        elif leadfield is not None and state.leadfield is None:
            if isinstance(leadfield, torch.Tensor):
                state.leadfield = leadfield.to(device=state.device, dtype=state.dtype)
            else:
                try:
                    state.leadfield = torch.load(leadfield).to(device=state.device, dtype=state.dtype)
                except Exception:
                    import numpy as np

                    loaded = np.load(leadfield)
                    state.leadfield = torch.as_tensor(loaded, device=state.device, dtype=state.dtype)
        if state.leadfield is None:
            raise ValueError("leadfield must be provided before streaming can continue")

        chunk_for_buffer = chunk.detach().clone()
        if state.buffer is None:
            state.buffer = chunk_for_buffer
        else:
            state.buffer = torch.cat([state.buffer, chunk_for_buffer], dim=1).contiguous()

    state.samples_seen += n_samples_chunk

    # Trim buffer to maximum duration while keeping most recent data
    if state.buffer is None:
        state.buffer = chunk.detach().clone()
    current_samples = state.buffer.size(1)
    if current_samples > buffer_max_samples:
        excess = current_samples - buffer_max_samples
        state.buffer = state.buffer[:, excess:]

    # Determine whether to recompute thresholds
    should_update_threshold = False
    if not state.initial_threshold_computed:
        if state.samples_seen >= state.initial_threshold_delay_samples:
            should_update_threshold = True
    else:
        samples_since_update = state.samples_seen - state.last_threshold_update_sample
        if samples_since_update >= state.threshold_update_interval_samples:
            should_update_threshold = True

    if should_update_threshold:
        was_computed = state.initial_threshold_computed
        try:
            result = gedai(
                state.buffer,
                sfreq=state.sfreq,
                denoising_strength=state.denoising_strength,
                leadfield=state.leadfield,
                epoch_size_in_cycles=state.epoch_size_in_cycles,
                lowcut_frequency=state.lowcut_frequency,
                wavelet_levels=state.wavelet_levels,
                matlab_levels=state.matlab_levels,
                device=state.device,
                dtype=state.dtype,
                TolX=state.TolX,
                maxiter=state.maxiter,
                skip_checks_and_return_cleaned_only=False,
            )

            state.thresholds_per_band = result["artifact_threshold_per_band"].detach().to(device=state.device).clone()
            state.lowcut_frequency_used = float(result["lowcut_frequency_used"])
            state.refCOV = result.get("refCOV", None)
            if state.refCOV is not None:
                state.refCOV = state.refCOV.detach().clone()

            state.initial_threshold_computed = True
            state.last_threshold_update_sample = state.samples_seen

            message = "Initial" if not was_computed else "Periodic"
            print(
                f"GEDAI Stream: {message} thresholds computed at {state.samples_seen / state.sfreq:.1f}s"
            )
        except Exception as exc:
            warnings.warn(f"Threshold computation failed: {exc}. Using previous thresholds.")

    if state.initial_threshold_computed and state.thresholds_per_band is not None:
        try:
            cleaning_lowcut = (
                state.lowcut_frequency_used if state.lowcut_frequency_used is not None else state.lowcut_frequency
            )
            cleaned_chunk = gedai(
                chunk,
                sfreq=state.sfreq,
                denoising_strength=state.denoising_strength,
                leadfield=state.leadfield,
                epoch_size_in_cycles=state.epoch_size_in_cycles,
                lowcut_frequency=cleaning_lowcut,
                wavelet_levels=state.wavelet_levels,
                matlab_levels=state.matlab_levels,
                device=state.device,
                dtype=state.dtype,
                TolX=state.TolX,
                maxiter=state.maxiter,
                skip_checks_and_return_cleaned_only=True,
                artifact_thresholds_override=state.thresholds_per_band,
            )
        except Exception as exc:
            warnings.warn(f"Cleaning failed: {exc}. Returning unprocessed chunk.")
            cleaned_chunk = chunk
    else:
        cleaned_chunk = chunk

    return cleaned_chunk, state

def reset_stream(state: GEDAIStreamState) -> GEDAIStreamState:
    """Reset streaming state while preserving configuration."""
    state.buffer = None
    state.samples_seen = 0
    state.thresholds_per_band = None
    state.lowcut_frequency_used = None
    state.refCOV = None
    state.last_threshold_update_sample = 0
    state.initial_threshold_computed = False

    return state