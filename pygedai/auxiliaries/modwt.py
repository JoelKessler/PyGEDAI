import torch
from typing import List, Dict, Tuple, Optional

# Cache COE shift indices per (n_samples, J) pair so repeated GEDAI calls
# don't re-run impulse MODWT reconstructions. Values are stored on CPU and
# moved to the requested device on demand.
_COE_SHIFT_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
# Cache the smooth-band COE shift (single integer) per (n_samples, J)
_SMOOTH_COE_CACHE: Dict[Tuple[int, int], int] = {}

# MODWT (Haar) using MATLAB analysis convention
def modwt_haar(x: torch.Tensor, J: int) -> List[torch.Tensor]:
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

def modwtmra_haar(
    coeffs: List[torch.Tensor],
    max_detail_bands: Optional[int] = None,
    return_smooth: bool = True,
) -> torch.Tensor:
    """
    Vectorized MRA construction with COE alignment.
    Returns (J+1, n_channels, n_samples) with zero-phase bands.
    Output matches the previous implementation exactly.
    """
    # Keep float32 as in your original for identical numerics
    details_all = [d.to(dtype=torch.float32) for d in coeffs[:-1]]
    scale = coeffs[-1].to(dtype=torch.float32)

    if max_detail_bands is not None:
        details = details_all[: max(0, min(max_detail_bands, len(details_all)))]
    else:
        details = details_all

    J = len(details)
    if J == 0:
        raise ValueError("At least one detail band is required for MODWT MRA.")
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
    smooth = None
    if return_smooth:
        sel0 = torch.zeros_like(W_stack)
        smooth = _imodwt_haar_multi(sel0, scale)[0] # any batch gives identical smooth
        cache_key = (int(T), int(len(details_all)))
        smooth_coe = _SMOOTH_COE_CACHE.get(cache_key)
        if smooth_coe is None:
            impulse = torch.zeros((1, T), device=device, dtype=dtype)
            impulse[0, T // 2] = 1.0
            coeffs_imp = modwt_haar(impulse, len(details_all))
            smooth_impulse = _imodwt_haar_multi(
                torch.zeros((len(details_all), 1, T), device=device, dtype=dtype),
                coeffs_imp[-1]
            )[0]
            smooth_coe = int(torch.argmax(torch.abs(smooth_impulse[0])).item()) - (T // 2)
            _SMOOTH_COE_CACHE[cache_key] = smooth_coe
        smooth_aligned = torch.roll(smooth, shifts=-smooth_coe, dims=-1) # (C, T)

    if return_smooth:
        return torch.cat([aligned_details, smooth_aligned.unsqueeze(0)], dim=0)
    return aligned_details

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

    # Prepare initial state in frequency domain for all J target bands
    V0 = VJ.unsqueeze(0).expand(J, C, T).contiguous().to(fdtype)
    FV = torch.fft.fft(V0.to(cdtype), dim=-1) # (J, C, T)

    # Pre-FFT of all W_j once (we'll select per level)
    FW_all = torch.fft.fft(W_stack.to(cdtype), dim=-1) # (J, C, T)

    # Identity used to build one-hot selectors without reallocating each loop
    eye = torch.eye(J, device=device, dtype=cdtype)

    # Walk levels from J..1, inserting the matching W only for its band
    for level in range(J, 0, -1):
        s = 2 ** (level - 1)
        z = twiddle ** s # (T,)

        # Haar analysis frequency responses
        inv_sqrt2 = (z.real.new_tensor(1.0) / torch.sqrt(z.real.new_tensor(2.0))).to(cdtype)
        Hj = (1 - z) * inv_sqrt2
        Gj = (1 + z) * inv_sqrt2
        inv_denom = 1.0 / (torch.abs(Gj) ** 2 + torch.abs(Hj) ** 2) # (T,)

        # Select FW for this level: only the batch whose target band == level uses W[level-1]
        # Build a mask that is 1 for batch==level-1 else 0, shape (J, 1, 1) to broadcast
        mask = eye[:, level - 1].view(J, 1, 1)
        FW_sel = FW_all[level - 1].unsqueeze(0) * mask # (J, C, T)

        # Vectorized update for all J reconstructions
        FV = (torch.conj(Gj) * FV + torch.conj(Hj) * FW_sel) * inv_denom  # (J, C, T)

    # After descending to level 1, V holds each per-band reconstruction
    return torch.fft.ifft(FV, dim=-1).real.to(fdtype) # (J, C, T)

def _compute_coe_shifts_vec(n_samples: int, J: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Vectorized COE shifts for all detail levels.
    Identical output to the scalar loop but ~Jx fewer Python trips.
    """
    cache_key = (int(n_samples), int(J))
    cached = _COE_SHIFT_CACHE.get(cache_key)
    if cached is None:
        cpu = torch.device("cpu")
        impulse = torch.zeros((1, n_samples), device=cpu, dtype=torch.float32)
        center_idx = n_samples // 2
        impulse[0, center_idx] = 1.0

        coeffs = modwt_haar(impulse, J)
        W = torch.stack([c for c in coeffs[:-1]], dim=0) # (J, 1, T)
        VJ = coeffs[-1] # (1, T)

        bands_imp = _imodwt_haar_multi(W, VJ * 0) # (J, 1, T)
        peak_idx = torch.argmax(torch.abs(bands_imp[:, 0, :]), dim=-1) # (J,)
        coe_shifts = peak_idx - center_idx # (J,)
        cached = coe_shifts.to(torch.long).detach()
        _COE_SHIFT_CACHE[cache_key] = cached
    return cached.to(device=device, dtype=torch.long)

def _complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Return a complex dtype matching the provided real dtype.

    Uses double precision complex for float32 and single precision
    complex for other float types.
    """
    return torch.cdouble if dtype == torch.float32 else torch.cfloat
