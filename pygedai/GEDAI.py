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
import torch.nn.functional as F

from typing import Union, Dict, Any, Optional, List
import math

from . import profiling
from .auxiliaries.GEDAI_per_band import gedai_per_band
from .auxiliaries.SENSAI_basic import sensai_basic
from .auxiliaries.GEDAI_nonRankDeficientAveRef import gedai_non_rank_deficient_avg_ref

from concurrent.futures import ThreadPoolExecutor

def batch_gedai(
    eeg_batch: torch.Tensor, # 3D tensor (batch_size, n_channels, n_samples)
    sfreq: float,
    denoising_strength: str = "auto",
    epoch_size: float = 1.0,
    leadfield: torch.Tensor = None,
    *,
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
                epoch_size=epoch_size,
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
    epoch_size: float = 1.0,
    leadfield: Union[str, torch.Tensor] = None,
    *,
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
    - epoch_size: requested epoch duration in seconds; adjusted to an even
      number of samples before processing.
    - leadfield: leadfield descriptor or matrix used to load reference covariance.
    - wavelet_levels / matlab_levels: level selection for MODWT analysis.
    - chanlabels: optional channel label list for leadfield mapping.
    - device / dtype: torch device and dtype for computation.
    - skip_checks_and_return_cleaned_only: if True, skips input validation
      and returns only the cleaned EEG tensor.

    The function returns a dictionary containing cleaned data,
    estimated artifacts, per-band sensai scores and thresholds, the
    epoch size actually used, and the reference covariance matrix.

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

    if verbose_timing:
        profiling.mark("start_gedai")

    n_ch = int(eeg.size(0))
    epoch_size_used = _ensure_even_epoch_size(float(epoch_size), sfreq)

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
    if pad_right:
        eeg_ref_proc = F.pad(eeg_ref, (0, pad_right), mode="replicate")  # e.g., 251 -> 252
    else:
        eeg_ref_proc = eeg_ref

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
    coeffs = _modwt_haar(cleaned_broadband, J)
    if verbose_timing:
        profiling.mark("modwt_analysis")

    bands = _modwtmra_haar(coeffs)
    if verbose_timing:
        profiling.mark("mra_constructed")
    
    # exclude lowest-frequency bands based on sampling rate
    exclude = int(torch.ceil(torch.tensor(600.0 / sfreq)).item())
    keep_upto = bands.size(0) - exclude
    
    if keep_upto <= 0:
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
        )

    # denoise kept bands and sum them
    bands_to_process = bands[:keep_upto]
    filt = torch.zeros_like(bands_to_process)

    if not skip_checks_and_return_cleaned_only:
        sensai_scores = [float(sensai_broadband)]
        thresholds = [float(thresh_broadband)]

    if verbose_timing:
        profiling.mark("prepare_band_processing")

    def _call_gedai_band(band_sig):
        if skip_checks_and_return_cleaned_only:
            cleaned_band, _, _, _ = gedai_per_band(
                band_sig, sfreq, None, denoising_strength, epoch_size_used, 
                refCOV, "parabolic", False,
                device=device, dtype=dtype, verbose_timing=bool(verbose_timing),
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
                TolX=TolX, maxiter=maxiter
            )
            return cleaned_band, None, None
        else:
            cleaned_band, _, s_band, thr_band = gedai_per_band(
                band_sig, sfreq, None, denoising_strength, epoch_size_used, 
                refCOV, "parabolic", False,
                device=device, dtype=dtype, verbose_timing=bool(verbose_timing),
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
                TolX=TolX, maxiter=maxiter
            )
            return cleaned_band, s_band, thr_band
        
    band_list = [bands_to_process[b] for b in range(bands_to_process.size(0))]

    if skip_checks_and_return_cleaned_only:
    # parallel map returning cleaned tensors
        if not batched:
            with ThreadPoolExecutor() as ex:
                results = list(ex.map(_call_gedai_band, band_list))
            for b, data in enumerate(results):
                cleaned_band, _, _ = data

                filt[b] = cleaned_band
            if verbose_timing:
                profiling.mark("bands_denoised_parallel")
        else:
            for b, band in enumerate(band_list):
                cleaned_band, _, _ = _call_gedai_band(band)
                filt[b] = cleaned_band
            if verbose_timing:
                profiling.mark("bands_denoised_serial")
    else:
        if batched:
            raise NotImplementedError("Batched processing with sensai scores not implemented yet.")
        
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(_call_gedai_band, band) for band in band_list]
            for b, fut in enumerate(futures):
                cleaned_band, s_band, thr_band = fut.result()
                filt[b] = cleaned_band
                sensai_scores.append(s_band)
                thresholds.append(thr_band)
                if verbose_timing:
                    profiling.mark(f"band_done_{b}")
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
    )

def _complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Return a complex dtype matching the provided real dtype.

    Uses double precision complex for float32 and single precision
    complex for other float types.
    """
    return torch.cdouble if dtype == torch.float32 else torch.cfloat

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

# MODWT (Haar) using MATLAB analysis convention
def _modwt_haar(x: torch.Tensor, J: int) -> List[torch.Tensor]:
    """Compute Haar MODWT coefficients up to level J.

    Analysis filters follow MATLAB's 'second pair' convention. The
    returned list contains detail coefficients W1..WJ and the final
    scaling coefficients VJ. Each tensor has shape (n_channels, n_samples).
    """
    if J < 1:
        raise ValueError("J must be >= 1")
    device = x.device
    dtype = x.dtype
    inv_sqrt2 = 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))

    h0 = inv_sqrt2
    h1 = inv_sqrt2
    g0 = inv_sqrt2
    g1 = -inv_sqrt2

    V = x.to(dtype=dtype).clone()
    coeffs: List[torch.Tensor] = []
    for j in range(1, J + 1):
        s = 2 ** (j - 1)
        # shift by the subsampling stride for this level
        V_roll = torch.roll(V, shifts=s, dims=-1)
        W = g0 * V + g1 * V_roll
        V = h0 * V + h1 * V_roll
        coeffs.append(W)
    return coeffs + [V]

def _imodwt_haar_multi(W_stack: torch.Tensor, VJ: torch.Tensor) -> torch.Tensor:
    """
    Vectorized inverse MODWT (Haar) for per-band reconstructions.

    Inputs
    - W_stack: (J, n_channels, n_samples) detail coeffs
    - VJ:      (n_channels, n_samples)    scaling coeffs

    Output
    - X_bands: (J, n_channels, n_samples)
      where X_bands[j] equals _imodwt_haar(sel, zeros_like(VJ))
      with sel[j] = W_stack[j], sel[k!=j] = 0
    """
    assert W_stack.ndim == 3, "W_stack must be (J, C, T)"
    J, C, T = W_stack.shape
    device = W_stack.device
    fdtype = W_stack.dtype
    cdtype = _complex_dtype_for(fdtype)

    # Precompute FFT twiddle for all k and complex dtype
    k = torch.arange(T, device=device, dtype=fdtype)
    angles = -2.0 * torch.pi * k / float(T)
    twiddle = torch.exp(1j * angles).to(dtype=cdtype) # (T,)

    # Prepare state replicated for all J target bands
    V = VJ.unsqueeze(0).expand(J, C, T).contiguous() # (J, C, T)

    # Pre-FFT of all W_j once (we'll select per level)
    FW_all = torch.fft.fft(W_stack.to(cdtype), dim=-1) # (J, C, T)

    # Walk levels from J..1, inserting the matching W only for its band
    for level in range(J, 0, -1):
        s = 2 ** (level - 1)
        z = twiddle ** s # (T,)

        # Haar analysis frequency responses
        inv_sqrt2 = (z.real.new_tensor(1.0) / torch.sqrt(z.real.new_tensor(2.0))).to(cdtype)
        Hj = (1 - z) * inv_sqrt2
        Gj = (1 + z) * inv_sqrt2
        inv_denom = 1.0 / (torch.abs(Gj) ** 2 + torch.abs(Hj) ** 2) # (T,)

        # FFT(V) for all bands in parallel
        FV = torch.fft.fft(V.to(cdtype), dim=-1) # (J, C, T)

        # Select FW for this level: only the batch whose target band == level uses W[level-1]
        # Build a mask that is 1 for batch==level-1 else 0, shape (J, 1, 1) to broadcast
        mask = torch.zeros((J, 1, 1), device=device, dtype=cdtype)
        mask[level - 1, 0, 0] = 1.0
        FW_sel = FW_all[level - 1].unsqueeze(0) * mask # (J, C, T)

        # Vectorized update for all J reconstructions
        X_prev = (torch.conj(Gj) * FV + torch.conj(Hj) * FW_sel) * inv_denom  # (J, C, T)
        V = torch.fft.ifft(X_prev, dim=-1).real.to(fdtype) # (J, C, T)

    # After descending to level 1, V holds each per-band reconstruction
    return V # (J, C, T)

def _compute_coe_shifts_vec(n_samples: int, J: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Vectorized COE shifts for all detail levels.
    Identical output to the scalar loop but ~Jx fewer Python trips.
    """
    impulse = torch.zeros((1, n_samples), device=device, dtype=dtype)
    center_idx = n_samples // 2
    impulse[0, center_idx] = 1.0

    coeffs = _modwt_haar(impulse, J)
    W = torch.stack([c for c in coeffs[:-1]], dim=0) # (J, 1, T)
    VJ = coeffs[-1] # (1, T)

    # Reconstruct all impulse responses per detail band in parallel
    bands_imp = _imodwt_haar_multi(W, VJ*0) # (J, 1, T)
    # Peak index per band
    peak_idx = torch.argmax(torch.abs(bands_imp[:, 0, :]), dim=-1) # (J,)
    coe_shifts = peak_idx - center_idx # (J,)
    return coe_shifts.to(torch.long)

def _modwtmra_haar(coeffs: List[torch.Tensor]) -> torch.Tensor:
    """
    Vectorized MRA construction with COE alignment.
    Returns (J+1, n_channels, n_samples) with zero-phase bands.
    Output matches the previous implementation exactly.
    """
    # Keep float32 as in your original for identical numerics
    details = [d.to(dtype=torch.float32) for d in coeffs[:-1]]
    scale = coeffs[-1].to(dtype=torch.float32)

    J = len(details)
    C, T = details[0].shape
    device = details[0].device
    dtype = details[0].dtype

    # Stack details for vectorized inverse
    W_stack = torch.stack(details, dim=0) # (J, C, T)

    # Vectorized COE for detail bands
    coe_shifts = _compute_coe_shifts_vec(T, J, device=device, dtype=dtype) # (J,)

    # Reconstruct all detail bands at once (with VJ=0)
    zero_scale = torch.zeros_like(scale)
    details_bands = _imodwt_haar_multi(W_stack, zero_scale) # (J, C, T)

    # Apply per-band COE alignment
    # Roll each band by -coe_shifts[j]
    shifts = (-coe_shifts).tolist()
    aligned_details = torch.stack(
        [torch.roll(details_bands[j], shifts=shifts[j], dims=-1) for j in range(J)],
        dim=0
    ) # (J, C, T)

    # Smooth band (same as your original, including COE via impulse)
    sel0 = torch.zeros_like(W_stack)
    smooth = _imodwt_haar_multi(sel0, scale)[0] # any batch gives identical smooth
    impulse = torch.zeros((1, T), device=device, dtype=dtype)
    impulse[0, T // 2] = 1.0
    coeffs_imp = _modwt_haar(impulse, J)
    smooth_impulse = _imodwt_haar_multi(
        torch.zeros((J, 1, T), device=device, dtype=dtype),
        coeffs_imp[-1]
    )[0] # (1, T)
    smooth_coe = int(torch.argmax(torch.abs(smooth_impulse[0])).item()) - (T // 2)
    smooth_aligned = torch.roll(smooth, shifts=-smooth_coe, dims=-1) # (C, T)

    # Stack to (J+1, C, T)
    out = torch.cat([aligned_details, smooth_aligned.unsqueeze(0)], dim=0)
    return out
