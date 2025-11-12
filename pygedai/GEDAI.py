# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

"""GEDAI: Generalized Eigenvalue Deartifacting Instrument (Python port).

This module implements the GEDAI pipeline using torch for numerical
operations. It provides helpers for converting between numpy and torch,
MODWT analysis and synthesis using the Haar filters, center-of-energy
alignment for zero-phase MRA, leadfield covariance loading, and the
top-level gedai function that runs the full cleaning pipeline.

The implementation follows MATLAB MODWT conventions for analysis
filters and provides an exact inverse in the frequency domain.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import torch
try:
    torch.set_num_threads(1) # intra-op
except Exception as ex:
    print(ex)
try:
    torch.set_num_interop_threads(1) # inter-op
except Exception as ex:
    print(ex)

from typing import Union, Dict, Any, Optional, List
import math
import warnings

from . import profiling
from .auxiliaries.GEDAI_per_band import gedai_per_band
from .auxiliaries.SENSAI_basic import sensai_basic
from .auxiliaries.GEDAI_nonRankDeficientAveRef import gedai_non_rank_deficient_avg_ref
from .auxiliaries.modwt import modwt_haar, modwtmra_haar

from concurrent.futures import ThreadPoolExecutor

def batch_gedai(
    eeg_batch: torch.Tensor, # 3D tensor (batch_size, n_channels, n_samples)
    sfreq: float,
    denoising_strength: str = "auto",
    leadfield: torch.Tensor = None,
    *,
    epoch_size_in_cycles: float = 12.0,
    lowcut_frequency: float = 0.5,
    wavelet_levels: Optional[int] = 9,
    matlab_levels: Optional[int] = None,
    chanlabels: Optional[List[str]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    parallel: bool = True,
    max_workers: int | None = None,
    verbose_timing: bool = False,
    TolX: float = 1e-1,
    maxiter: int = 500
):
    if verbose_timing:
        profiling.reset()
        profiling.enable(True)
        profiling.mark("start_batch")

    if eeg_batch.ndim != 3:
        raise ValueError("eeg_batch must be 3D (batch_size, n_channels, n_samples).")
    if leadfield is None or (leadfield.shape != (eeg_batch.shape[1], eeg_batch.shape[1])):
        raise ValueError("leadfield must be provided with shape (n_channels, n_channels).")

    def _one(eeg_idx: int) -> torch.Tensor:
        if verbose_timing:
            profiling.mark(f"one_start_idx_{eeg_idx}")
        try:
            if verbose_timing:
                profiling.mark(f"one_end_idx_{eeg_idx}")
            return gedai(
                eeg_batch[eeg_idx], sfreq,
                denoising_strength=denoising_strength,
                epoch_size_in_cycles=epoch_size_in_cycles,
                lowcut_frequency=lowcut_frequency,
                leadfield=leadfield,
                wavelet_levels=wavelet_levels,
                matlab_levels=matlab_levels,
                chanlabels=chanlabels,
                device=device,
                dtype=dtype,
                skip_checks_and_return_cleaned_only=True,
                batched=True,
                verbose_timing=bool(verbose_timing),
                TolX=TolX,
                maxiter=maxiter
            )
        except:
            print(f"GEDAI failed for batch index {eeg_idx}. Returning unmodified data.")
            return eeg_batch[eeg_idx]

    eeg_idx_total = eeg_batch.size(0)
    if not parallel:
        results = [_one(eeg_idx) for eeg_idx in range(eeg_idx_total)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_one, range(eeg_idx_total)))
    cleaned_batch = torch.stack(results, dim=0)
    
    if verbose_timing:
        profiling.mark("batch_done")
        profiling.report()

    return cleaned_batch # cleaned batch

def gedai(
    eeg: torch.Tensor,
    sfreq: float,
    denoising_strength: str = "auto",
    leadfield: Union[str, torch.Tensor] = None,
    *,
    epoch_size_in_cycles: float = 12.0,
    lowcut_frequency: float = 0.5,
    wavelet_levels: Optional[int] = 9,
    matlab_levels: Optional[int] = None,
    chanlabels: Optional[List[str]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    skip_checks_and_return_cleaned_only: bool = False,
    batched=False,
    verbose_timing: bool = False,
    TolX: float = 1e-1,
    maxiter: int = 500
) -> Union[Dict[str, Any], torch.Tensor]:
    """Run the GEDAI cleaning pipeline on raw or preprocessed EEG.

    Parameters
    - eeg: array-like or tensor shaped (n_channels, n_samples).
    - sfreq: sampling frequency in Hz.
    - denoising_strength: passed to per-band denoiser helpers.
    - leadfield: leadfield descriptor or matrix used to load reference covariance.
    - epoch_size_in_cycles: number of wave cycles per wavelet band epoch (default 12).
    - lowcut_frequency: exclude wavelet bands whose upper bound <= this frequency (Hz).
    - wavelet_levels / matlab_levels: level selection for MODWT analysis.
    - chanlabels: optional channel label list for leadfield mapping.
    - device / dtype: torch device and dtype for computation.
    - skip_checks_and_return_cleaned_only: if True, skips input validation
      and returns only the cleaned EEG tensor.

    The function returns a dictionary containing cleaned data,
    estimated artifacts, per-band sensai scores and thresholds, the
    epoch size actually used, and the reference covariance matrix.
    - epoch_sizes_per_band: Tensor of per-band epoch durations (seconds) used during wavelet cleaning.
    - lowcut_frequency_used: Low-cut frequency (Hz) applied to exclude low-frequency wavelet bands.

    Explainer:
    Raw EEG data is fully cleaned by breaking it into frequency bands and cleaning each band.
    Each frequency band contains different types of artifacts and neural signals.
    Low frequencies e.g. eye movements, high frequencies e.g. muscle artifacts. Mid frequencies e.g. real brain signals (alpha, beta).

    1. Load data and apply non rank deficient average reference. Pad data to full epochs.
    2. Broadband denoising pass to remove gross artifacts. (all frequencies together)
    3. MODWT decomposition into frequency bands using Haar wavelets. (Split cleaned broadband into wavelet_levels bands, 1 = muscle noise, 9 = drift).
    4. Exclude very slow frequencies which ususally are just drift, exclude bottom bands.
    5. For all bands in parallel identify artifacts and clean.
    6. Reconstruct cleaned EEG by summing all cleaned bands.
    7. Compute quality score.
    """
    if eeg is None:
        raise ValueError("eeg must be provided.")
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_samples).")
    if leadfield is None:
        raise ValueError("leadfield is required.")
    if chanlabels is not None:
        raise NotImplementedError("chanlabels handling not implemented yet.")
    
    eeg = eeg.to(device=device, dtype=dtype)
    epoch_size_in_cycles = float(epoch_size_in_cycles)
    if epoch_size_in_cycles <= 0.0:
        raise ValueError("epoch_size_in_cycles must be positive.")

    if verbose_timing:
        profiling.mark("start_gedai")

    n_ch = int(eeg.size(0))
    broadband_epoch_seconds = 1.0
    epoch_size_used = _ensure_even_epoch_size(broadband_epoch_seconds, sfreq)

    if verbose_timing:
        profiling.mark("post_checks")

    if isinstance(leadfield, torch.Tensor):
        leadfield_t = leadfield.to(device=device, dtype=dtype)
    elif isinstance(leadfield, str):
        try:
            leadfield_t = torch.load(leadfield).to(device=device, dtype=dtype)
        except:
            import numpy as np
            loaded = np.load(leadfield)
            leadfield_t = torch.as_tensor(loaded, device=device, dtype=dtype)
    else:
        raise ValueError("leadfield must be ndarray, path string, tensor.")

    if int(leadfield_t.ndim) != 2 or int(leadfield_t.size(0)) != n_ch or int(leadfield_t.size(1)) != n_ch:
        raise ValueError(
            f"leadfield covariance must be ({n_ch}, {n_ch}), got {leadfield_t.shape}."
        )
    refCOV = leadfield_t

    if verbose_timing:
        profiling.mark("leadfield_loaded")

    # apply non-rank-deficient average reference
    eeg_ref = gedai_non_rank_deficient_avg_ref(eeg)

    if verbose_timing:
        profiling.mark("avg_ref_applied")

    # pad right to next full epoch, then trim back later
    T_in = int(eeg_ref.size(1))
    epoch_samp = int(round(epoch_size_used * sfreq))  # e.g., 126 when enforcing even samples at 125 Hz
    rem = T_in % epoch_samp
    pad_right = (epoch_samp - rem) if rem != 0 else 0
    eeg_ref_proc = _pad_reflect_tail(eeg_ref, pad_right)

    if verbose_timing:
        profiling.mark("padding_done")

    # broadband denoising uses the numpy-based helper and is returned as numpy
    cleaned_broadband, _, sensai_broadband, thresh_broadband = gedai_per_band(
        eeg_ref_proc, sfreq, None, "auto-", epoch_size_used, refCOV.to(device=device), "parabolic", False,
        device=device, dtype=dtype, verbose_timing=bool(verbose_timing), TolX=TolX, maxiter=maxiter,
        skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only
    )
    if verbose_timing:
        profiling.mark("broadband_denoise")
    
    # Ensure cleaned_broadband is on the correct device
    cleaned_broadband = cleaned_broadband.to(device=device, dtype=dtype)
    
    # compute MODWT coefficients and validate perfect reconstruction
    J = (2 ** int(matlab_levels) + 1) if (matlab_levels is not None) else int(wavelet_levels)
    coeffs = modwt_haar(cleaned_broadband, J)
    if verbose_timing:
        profiling.mark("modwt_analysis")

    bands = modwtmra_haar(coeffs)
    if verbose_timing:
        profiling.mark("mra_constructed")
    
    # frequency bookkeeping for wavelet bands
    num_bands_total = int(bands.size(0))
    freq_dtype = torch.float64 if cleaned_broadband.dtype == torch.float64 else torch.float32
    levels = torch.arange(1, num_bands_total + 1, device=cleaned_broadband.device, dtype=freq_dtype)
    denom_upper = torch.pow(2.0, levels)
    denom_lower = torch.pow(2.0, levels + 1.0)
    upper_frequencies_tensor = float(sfreq) / denom_upper
    lower_frequencies_tensor = float(sfreq) / denom_lower
    center_frequencies_tensor = 0.5 * (lower_frequencies_tensor + upper_frequencies_tensor)
    center_frequencies: List[float] = center_frequencies_tensor.tolist()
    lower_frequencies: List[float] = lower_frequencies_tensor.tolist()
    upper_frequencies: List[float] = upper_frequencies_tensor.tolist()

    lowcut_frequency = float(lowcut_frequency)
    if lowcut_frequency < 0.0:
        raise ValueError("lowcut_frequency must be non-negative.")

    mask = upper_frequencies_tensor <= lowcut_frequency
    lowest_wavelet_bands_to_exclude = int(mask.sum().item())
    num_bands_to_process = num_bands_total - lowest_wavelet_bands_to_exclude

    data_sample_count = int(cleaned_broadband.size(1))
    lowcut_in_use = lowcut_frequency
    while num_bands_to_process > 0:
        lowest_idx = num_bands_to_process - 1
        required_samples = (epoch_size_in_cycles / lower_frequencies[lowest_idx]) * float(sfreq)
        if required_samples <= data_sample_count:
            break
        warnings.warn(
            (
                "EEG data length is too short for the epoch size required by the lowest "
                f"frequency band ({center_frequencies[lowest_idx]:.3g} Hz). Increasing lowcut_frequency."
            ),
            RuntimeWarning,
        )
        lowcut_in_use = upper_frequencies[lowest_idx]
        mask = upper_frequencies_tensor <= lowcut_in_use
        lowest_wavelet_bands_to_exclude = int(mask.sum().item())
        num_bands_to_process = num_bands_total - lowest_wavelet_bands_to_exclude

    lowcut_frequency = lowcut_in_use

    if num_bands_to_process <= 0:
        cleaned = cleaned_broadband
        if skip_checks_and_return_cleaned_only:
            # trim back to original length if we padded
            if pad_right:
                cleaned = cleaned[:, :T_in]
            return cleaned
        
        artifacts = eeg_ref_proc[:, :cleaned.size(1)] - cleaned
        try:
            sensai_score = float(
                sensai_basic(
                    cleaned, 
                    artifacts, 
                    float(sfreq), 
                    float(epoch_size_used), 
                    refCOV, 
                    1.0,
                    verbose_timing=verbose_timing)[0]
            )
        except Exception as ex:
            sensai_score = None
            
        # trim back to original length if we padded
        if pad_right:
            cleaned = cleaned[:, :T_in]
            artifacts = artifacts[:, :T_in]
        return dict(
            cleaned=cleaned,
            artifacts=artifacts,
            sensai_score=sensai_score,
            sensai_score_per_band=torch.tensor([float(sensai_broadband)], device=device, dtype=dtype),
            artifact_threshold_per_band=torch.tensor([float(thresh_broadband)], device=device, dtype=dtype),
            artifact_threshold_broadband=float(thresh_broadband),
            epoch_size_used=float(epoch_size_used),
            refCOV=refCOV,
            epoch_sizes_per_band=torch.empty(0, device=device, dtype=dtype),
            lowcut_frequency_used=float(lowcut_frequency),
        )

    # determine per-band epoch sizes based on cycle count
    if num_bands_to_process > 0:
        freq_subset = lower_frequencies_tensor[:num_bands_to_process].to(torch.float64)
        sfreq_tensor = torch.as_tensor(float(sfreq), dtype=torch.float64, device=freq_subset.device)
        epoch_cycles = torch.full_like(freq_subset, float(epoch_size_in_cycles), dtype=torch.float64)
        ideal_samples = (epoch_cycles / freq_subset) * sfreq_tensor
        # MATLAB half-away-from-zero rounding in vector form, keep integers exact.
        rounded_samples = torch.sign(ideal_samples) * torch.floor(torch.abs(ideal_samples) + 0.5)
        rounded_samples = torch.where(rounded_samples <= 0.0, torch.full_like(rounded_samples, 2.0), rounded_samples)
        rounded_samples_long = rounded_samples.to(torch.int64)
        odd_mask = (rounded_samples_long & 1) != 0
        if bool(odd_mask.any()):
            dist_lo = torch.abs(ideal_samples - (rounded_samples_long - 1).to(torch.float64))
            dist_hi = torch.abs(ideal_samples - (rounded_samples_long + 1).to(torch.float64))
            choose_minus = dist_lo < dist_hi
            adjusted_odds = torch.where(choose_minus, rounded_samples_long - 1, rounded_samples_long + 1)
            rounded_samples_long = torch.where(odd_mask, adjusted_odds, rounded_samples_long)
        rounded_samples_long = torch.clamp(rounded_samples_long, min=2)
        epoch_sizes_per_wavelet_band: List[float] = (rounded_samples_long.to(torch.float64) / sfreq_tensor).cpu().tolist()
    else:
        epoch_sizes_per_wavelet_band = []

    # denoise kept bands and sum them
    bands_to_process = bands[:num_bands_to_process]
    filt = torch.zeros_like(bands_to_process)

    if not skip_checks_and_return_cleaned_only:
        sensai_scores = [float(sensai_broadband)]
        thresholds = [float(thresh_broadband)]

    if verbose_timing:
        profiling.mark("prepare_band_processing")

    def _call_gedai_band(band_payload):
        band_idx, band_sig = band_payload
        current_epoch_size = epoch_sizes_per_wavelet_band[band_idx]
        if skip_checks_and_return_cleaned_only:
            cleaned_band, _, _, _ = gedai_per_band(
                band_sig, sfreq, None, denoising_strength, current_epoch_size, 
                refCOV, "parabolic", False,
                device=device, dtype=dtype, verbose_timing=bool(verbose_timing),
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
                TolX=TolX, maxiter=maxiter
            )
            return band_idx, cleaned_band, None, None
        else:
            cleaned_band, _, s_band, thr_band = gedai_per_band(
                band_sig, sfreq, None, denoising_strength, current_epoch_size, 
                refCOV, "parabolic", False,
                device=device, dtype=dtype, verbose_timing=bool(verbose_timing),
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
                TolX=TolX, maxiter=maxiter
            )
            return band_idx, cleaned_band, s_band, thr_band
        
    band_list = [(b, bands_to_process[b]) for b in range(bands_to_process.size(0))]

    if skip_checks_and_return_cleaned_only:
    # parallel map returning cleaned tensors
        if not batched:
            with ThreadPoolExecutor() as ex:
                results = list(ex.map(_call_gedai_band, band_list))
            for band_idx, cleaned_band, _, _ in results:
                filt[band_idx] = cleaned_band
            if verbose_timing:
                profiling.mark("bands_denoised_parallel")
        else:
            for payload in band_list:
                band_idx, cleaned_band, _, _ = _call_gedai_band(payload)
                filt[band_idx] = cleaned_band
            if verbose_timing:
                profiling.mark("bands_denoised_serial")
    else:
        if batched:
            raise NotImplementedError("Batched processing with sensai scores not implemented yet.")
        
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(_call_gedai_band, band) for band in band_list]
            for fut in futures:
                band_idx, cleaned_band, s_band, thr_band = fut.result()
                filt[band_idx] = cleaned_band
                sensai_scores.append(float(s_band))
                thresholds.append(float(thr_band))
                if verbose_timing:
                    profiling.mark(f"band_done_{band_idx}")
    cleaned = filt.sum(dim=0)

    if verbose_timing:
        profiling.mark("bands_summed")

    if skip_checks_and_return_cleaned_only:
        # trim back to original length if we padded
        if pad_right:
            cleaned = cleaned[:, :T_in]
        if verbose_timing:
            profiling.mark("done_return_cleaned_only")
            profiling.report()
        return cleaned
    
    artifacts = eeg_ref_proc[:, :cleaned.size(1)] - cleaned

    try:
        sensai_score = float(
            sensai_basic(
                cleaned, 
                artifacts, 
                float(sfreq), 
                float(epoch_size_used), 
                refCOV, 
                1.0,
                verbose_timing=verbose_timing)[0]
        )
    except Exception as ex:
        sensai_score = None
        
    # trim back to original length if we padded
    if pad_right:
        cleaned = cleaned[:, :T_in]
        artifacts = artifacts[:, :T_in]

    if verbose_timing:
        profiling.mark("sensai_final")
        profiling.report()

    return dict(
        cleaned=cleaned,
        artifacts=artifacts,
        sensai_score=sensai_score,
        sensai_score_per_band=torch.as_tensor(sensai_scores, device=device, dtype=dtype),
        artifact_threshold_per_band=torch.as_tensor(thresholds, device=device, dtype=dtype),
        artifact_threshold_broadband=float(thresh_broadband),
        epoch_size_used=float(epoch_size_used),
        refCOV=refCOV,
        epoch_sizes_per_band=torch.as_tensor(epoch_sizes_per_wavelet_band, device=device, dtype=dtype),
        lowcut_frequency_used=float(lowcut_frequency),
    )

def _pad_reflect_tail(data: torch.Tensor, pad_right: int) -> torch.Tensor:
    """Reflectively pad the right edge by pad_right samples."""
    if pad_right <= 0 or data.size(1) == 0:
        return data
    width = int(data.size(1))
    repeats = (int(pad_right) + width - 1) // width
    flipped = torch.flip(data, dims=[1])
    padding = flipped.repeat(1, repeats)
    padding = padding[:, :pad_right]
    return torch.cat([data, padding], dim=1)

# MATLAB rounding and epoch-size parity 
def _matlab_round_half_away_from_zero(x: float) -> int:
    """Round a float following MATLAB's half-away-from-zero rule.

    This matches MATLAB behavior where .5 values round away from zero.
    """
    xt = float(x)
    r = math.floor(abs(xt) + 0.5)
    r = r if xt >= 0 else -r
    return int(r)

def _ensure_even_epoch_size(epoch_size_sec: float, sfreq: float) -> float:
    """Return an epoch size (in seconds) corresponding to an even number of samples.

    The function computes the ideal number of samples for the requested
    epoch duration and adjusts to the nearest even integer using the
    MATLAB rounding rule above. The returned value is the adjusted
    duration in seconds.
    """
    ideal = epoch_size_sec * sfreq
    nearest = _matlab_round_half_away_from_zero(float(ideal))
    if nearest % 2 != 0:
        dist_lo = abs(float(ideal) - (nearest - 1))
        dist_hi = abs(float(ideal) - (nearest + 1))
        nearest = (nearest - 1) if dist_lo < dist_hi else (nearest + 1)
    return float(nearest) / float(sfreq)
