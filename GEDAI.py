"""GEDAI: Generalized Eigenvalue Deartifacting Instrument (Python port).

This module implements the GEDAI pipeline using torch for numerical
operations. It provides helpers for converting between numpy and torch,
MODWT analysis and synthesis using the Haar filters, center-of-energy
alignment for zero-phase MRA, leadfield covariance loading, and the
top-level gedai function that runs the full cleaning pipeline.

The implementation follows MATLAB MODWT conventions for analysis
filters and provides an exact inverse in the frequency domain.
"""
# TODO: Clarify with autor, If uneven number after halving epochs e.g. sfreq=125, epoch_size=1.0s, 250 samples -> 125 is uneven -> round up and return original shape?
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
import numpy as np
import math
import profiling

from auxiliaries.GEDAI_per_band import gedai_per_band
from auxiliaries.SENSAI_basic import sensai_basic

from concurrent.futures import ThreadPoolExecutor

def batch_gedai(
    eeg_batch: Union[torch.Tensor, list[torch.Tensor]], # list if varying lengths
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
):
    if verbose_timing:
        profiling.reset()
        profiling.enable(True)
        profiling.mark("start_batch")

    is_list = isinstance(eeg_batch, list)
    if not is_list and eeg_batch.ndim != 3:
        raise ValueError("eeg_batch must be 3D (batch_size, n_channels, n_samples).")
    if leadfield is None or (
        is_list and leadfield.shape != (eeg_batch[0].shape[0], eeg_batch[0].shape[0])) or (
        not is_list and leadfield.shape != (eeg_batch.shape[1], eeg_batch.shape[1])):
        raise ValueError("leadfield must be provided with shape (n_channels, n_channels).")

    def _one(eeg_idx: int) -> torch.Tensor:
        if verbose_timing:
            profiling.mark(f"one_start_idx_{eeg_idx}")

        eeg_batch[eeg_idx] = gedai(
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
            verbose_timing=bool(verbose_timing)
        )

        if verbose_timing:
            profiling.mark(f"one_end_idx_{eeg_idx}")
        return True

    if is_list:
        eeg_idx_total = len(eeg_batch)
    else:
        eeg_idx_total = eeg_batch.size(0)
    if not parallel:
        results = [_one(eeg_idx) for eeg_idx in range(eeg_idx_total)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_one, range(eeg_idx_total)))
    if verbose_timing:
        profiling.mark("batch_done")
        profiling.report()

    if is_list:
        return eeg_batch

    return eeg_batch #torch.stack(results, dim=0).to(device=device)

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
            loaded = np.load(leadfield)
            leadfield_t = torch.as_tensor(loaded, device=device, dtype=dtype)
        except:
            leadfield_t = torch.load(leadfield).to(device=device, dtype=dtype)
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
    eeg_ref = _non_rank_deficient_avg_ref(eeg)

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
        device=device, dtype=dtype, verbose_timing=bool(verbose_timing)
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
    if skip_checks_and_return_cleaned_only:
        xrec = _imodwt_haar(coeffs[:-1], coeffs[-1])
        assert torch.allclose(xrec, cleaned_broadband, rtol=1e-10, atol=1e-12), "MODWT inverse failed PR"
    
    bands = _modwtmra_haar(coeffs)
    if verbose_timing:
        profiling.mark("mra_constructed")
    if skip_checks_and_return_cleaned_only:
        assert torch.allclose(bands.sum(dim=0), cleaned_broadband, rtol=1e-10, atol=1e-12), "MRA additivity failed"
    
    # exclude lowest-frequency bands based on sampling rate
    exclude = int(torch.ceil(torch.tensor(600.0 / sfreq)).item())
    keep_upto = bands.size(0) - exclude
    
    if keep_upto <= 0:
        cleaned = cleaned_broadband
        # trim back to original length if we padded
        if pad_right:
            cleaned = cleaned[:, :T_in]
        if skip_checks_and_return_cleaned_only:
            return cleaned
        
        artifacts = eeg_ref[:, :cleaned.size(1)] - cleaned
        try:
            sensai_score = float(
                sensai_basic(cleaned, artifacts, float(sfreq), float(epoch_size_used), refCOV, 1.0)[0]
            )
        except Exception:
            sensai_score = None
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
            return gedai_per_band(
                band_sig, sfreq, None, denoising_strength, epoch_size_used, 
                refCOV, "parabolic", False,
                device=device, dtype=dtype, verbose_timing=bool(verbose_timing),
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only
            )
        else:
            cleaned_band, _, s_band, thr_band = gedai_per_band(
                band_sig, sfreq, None, denoising_strength, epoch_size_used, 
                refCOV, "parabolic", False,
                device=device, dtype=dtype, verbose_timing=bool(verbose_timing),
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only
            )
            return cleaned_band, s_band, thr_band
        
    band_list = [bands_to_process[b] for b in range(bands_to_process.size(0))]

    if skip_checks_and_return_cleaned_only:
    # parallel map returning cleaned tensors
        if not batched:
            with ThreadPoolExecutor() as ex:
                results = list(ex.map(_call_gedai_band, band_list))
            for b, cleaned_band in enumerate(results):
                filt[b] = cleaned_band
            if verbose_timing:
                profiling.mark("bands_denoised_parallel")
        else:
            for b, band in enumerate(band_list):
                filt[b] = _call_gedai_band(band)
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

    # trim back to original length if we padded
    if pad_right:
        cleaned = cleaned[:, :T_in]

    if skip_checks_and_return_cleaned_only:
        if verbose_timing:
            profiling.mark("done_return_cleaned_only")
            profiling.report()
        return cleaned
    
    artifacts = eeg_ref[:, :cleaned.size(1)] - cleaned

    try:
        sensai_score = float(
            sensai_basic(cleaned, artifacts, float(sfreq), float(epoch_size_used), refCOV, 1.0)[0]
        )
    except Exception:
        sensai_score = None

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

# referencing & leadfield
def _non_rank_deficient_avg_ref(eeg: torch.Tensor) -> torch.Tensor:
    """Apply a non-rank-deficient average reference to EEG data.

    The method subtracts the channel mean while preserving full rank by
    dividing by (n_ch + 1). Input `eeg` is expected shape (n_channels, n_samples).
    """
    n_ch = eeg.size(0)
    return eeg - (eeg.sum(dim=0, keepdim=True) / n_ch) # matlab uses n_ch other possible (n_ch + 1.0)

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

def _imodwt_haar(W_list: List[torch.Tensor], VJ: torch.Tensor) -> torch.Tensor:
    """Synthesize a time-domain signal from Haar MODWT bands.

    The inverse is computed in the frequency domain using a least
    squares combination of detail and scaling filters to ensure an
    exact reconstruction for periodic signals.
    """
    V = VJ.to(dtype=VJ.dtype).clone()
    device = V.device
    fdtype = V.dtype
    cdtype = _complex_dtype_for(fdtype)

    n = V.size(-1)
    k = torch.arange(n, device=device, dtype=fdtype)
    angles = -2.0 * torch.pi * k / float(n)
    twiddle = torch.exp(1j * angles).to(dtype=cdtype)

    for j in range(len(W_list), 0, -1):
        s = 2 ** (j - 1)
        # frequency-domain factors for this level
        z = twiddle ** s
        one = torch.ones_like(z)
        inv_sqrt2 = (one.real.new_tensor(1.0) / torch.sqrt(one.real.new_tensor(2.0))).to(cdtype)

        Hj = (one - z) * inv_sqrt2
        Gj = (one + z) * inv_sqrt2

        inv_denom = 1.0 / (torch.abs(Gj) ** 2 + torch.abs(Hj) ** 2)

        FV = torch.fft.fft(V.to(cdtype), dim=-1)
        FW = torch.fft.fft(W_list[j - 1].to(cdtype), dim=-1)
        X_prev = (torch.conj(Gj) * FV + torch.conj(Hj) * FW) * inv_denom
        V = torch.fft.ifft(X_prev, dim=-1).real.to(fdtype)
    return V

def _compute_coe_shifts(n_samples: int, J: int, device, dtype=torch.float32) -> torch.Tensor:
    """Compute center-of-energy shifts for each detail level.

    An impulse signal is analyzed and reconstructed for each detail
    band to locate the impulse peak. The offset from the center is
    returned for alignment of MRA bands to produce zero-phase outputs.
    """
    impulse = torch.zeros((1, n_samples), device=device, dtype=dtype)
    center_idx = n_samples // 2
    impulse[0, center_idx] = 1.0

    coeffs = _modwt_haar(impulse, J)
    details = coeffs[:-1]
    scale = coeffs[-1]

    coe_shifts = torch.zeros(J, dtype=torch.long, device=device)

    for j in range(J):
        sel = [torch.zeros_like(d) for d in details]
        sel[j] = details[j]
        band = _imodwt_haar(sel, torch.zeros_like(scale))
        peak_idx = int(torch.argmax(torch.abs(band[0])).item())
        coe_shifts[j] = peak_idx - center_idx

    return coe_shifts

def _modwtmra_haar(coeffs: List[torch.Tensor]) -> torch.Tensor:
    """Construct MRA bands with center-of-energy alignment.

    Returns a tensor of shape (J+1, n_channels, n_samples) where the
    first J entries are detail bands D1..DJ and the last entry is the
    smooth (scaling) band SJ. Bands are aligned to have zero phase.
    """
    details = [d.to(dtype=torch.float32) for d in coeffs[:-1]]
    scale = coeffs[-1].to(dtype=torch.float32)
    J = len(details)
    n_samples = details[0].size(-1)
    device = details[0].device
    dtype = details[0].dtype

    coe_shifts = _compute_coe_shifts(n_samples, J, device=device, dtype=dtype)

    bands: List[torch.Tensor] = []
    for j in range(J):
        sel = [torch.zeros_like(d) for d in details]
        sel[j] = details[j]
        band = _imodwt_haar(sel, torch.zeros_like(scale))
        band_aligned = torch.roll(band, shifts=int(-coe_shifts[j].item()), dims=-1)
        bands.append(band_aligned)

    sel0 = [torch.zeros_like(d) for d in details]
    smooth = _imodwt_haar(sel0, scale)

    # determine smooth band COE and align
    impulse = torch.zeros((1, n_samples), device=device, dtype=dtype)
    impulse[0, n_samples // 2] = 1.0
    coeffs_impulse = _modwt_haar(impulse, J)
    smooth_impulse = _imodwt_haar(
        [torch.zeros_like(d) for d in coeffs_impulse[:-1]], coeffs_impulse[-1]
    )
    smooth_coe = int(torch.argmax(torch.abs(smooth_impulse[0])).item()) - (n_samples // 2)

    smooth_aligned = torch.roll(smooth, shifts=-smooth_coe, dims=-1)
    bands.append(smooth_aligned)

    out = torch.stack(bands, dim=0)

    return out
