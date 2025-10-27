import torch
import numpy as np
from typing import Tuple, Union

from .clean_EEG import clean_eeg

def _cov_matlab_like(X: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """
    MATLAB-like covariance for X with shape (channels, samples),
    unbiased (ddof=1), Hermitian-symmetrized for stability.
    """
    X = X.to(torch.float64)
    S = X.size(1)
    if S <= ddof:
        raise ValueError(f"n_samples ({S}) must be > ddof ({ddof})")
    Xm = X - X.mean(dim=1, keepdim=True)
    cov = (Xm @ Xm.T) / float(S - ddof)
    return 0.5 * (cov + cov.T)

def _cosprod_subspace(U: torch.Tensor, V: torch.Tensor) -> float:
    """
    Product of cosines of principal angles between span(U) and span(V).
    Equal to product of singular values of U^T V.
    """
    M = U.T @ V
    s = torch.linalg.svdvals(M) # singular values in [0,1]
    s = torch.clamp(s, 0.0, 1.0)
    return float(torch.prod(s).item())

def sensai(
    EEGdata_epoched: torch.Tensor,
    srate: float,
    epoch_size: float,
    artifact_threshold: float,
    refCOV: torch.Tensor,
    Eval: torch.Tensor,
    Evec: torch.Tensor,
    noise_multiplier: float,
    top_PCs: int = 3,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[float, float, float]:
    """
    Compute SENSAI score and subspace similarities.

    Parameters:
    EEGdata_epoched: Epoched EEG data (channels x samples).
    srate: Sampling rate of the data.
    epoch_size: Duration of each epoch in seconds.
    artifact_threshold: Threshold for artifact detection.
    refCOV: Reference covariance matrix.
    Eval: Eigenvalues for each epoch.
    Evec: Eigenvectors for each epoch.
    noise_multiplier: Multiplier for noise similarity.
    top_PCs: Number of top principal components to consider.
    device: Device for computation (e.g., 'cpu', 'cuda').
    dtype: Data type for computation.

    Returns:
    Tuple containing SENSAI score, signal subspace similarity, and noise subspace similarity.
    """
    # Run GEDAI cleaning
    EEGout_data, EEG_artifacts_data, _ = clean_eeg(
        EEGdata_epoched=EEGdata_epoched.to(device=device, dtype=dtype),
        srate=float(srate),
        epoch_size=float(epoch_size),
        artifact_threshold_in=float(artifact_threshold),
        refCOV=refCOV.to(device=device, dtype=dtype),
        Eval=Eval.to(device=device, dtype=dtype),
        Evec=Evec.to(device=device, dtype=dtype),
        strict_matlab=True,
        device=device,
        dtype=dtype,
    )

    num_chans = refCOV.size(0)
    epoch_samples = int(round(float(srate) * float(epoch_size)))
    top_PCs_eff = min(int(top_PCs), num_chans)

    # Top eigenvectors of reference covariance (descending)
    wT, VT = torch.linalg.eigh(refCOV.to(device=device, dtype=dtype))  # ascending
    idxT = torch.argsort(wT, descending=True)
    VT = VT[:, idxT][:, :top_PCs_eff]

    # Reshape to epochs like NumPy order='F'
    if EEGout_data.size(0) != num_chans or EEG_artifacts_data.size(0) != num_chans:
        raise ValueError("EEGout/artifacts channel dimension mismatch with refCOV.")

    total_samples = EEGout_data.size(1)
    if total_samples % epoch_samples != 0:
        raise ValueError("Total samples are not divisible by epoch size.")
    num_epochs = total_samples // epoch_samples

    # (C, T) -> unfold( size=S, step=S ) -> (C, E, S) -> permute -> (C, S, E)
    Sig_ep = EEGout_data.unfold(1, epoch_samples, epoch_samples).permute(0, 2, 1).contiguous()
    Res_ep = EEG_artifacts_data.unfold(1, epoch_samples, epoch_samples).permute(0, 2, 1).contiguous()

    sig_sim = torch.empty(num_epochs, device=device, dtype=dtype)
    noi_sim = torch.empty(num_epochs, device=device, dtype=dtype)

    for ep in range(num_epochs):
        # Signal subspace vs template
        X = Sig_ep[:, :, ep]
        cov_sig = _cov_matlab_like(X, ddof=1)
        wS, VS = torch.linalg.eigh(cov_sig)
        VS = VS[:, torch.argsort(wS, descending=True)][:, :top_PCs_eff]
        sig_sim[ep] = _cosprod_subspace(VS, VT)

        # Noise subspace vs template
        R = Res_ep[:, :, ep]
        cov_res = _cov_matlab_like(R, ddof=1)
        wN, VN = torch.linalg.eigh(cov_res)
        VN = VN[:, torch.argsort(wN, descending=True)][:, :top_PCs_eff]
        noi_sim[ep] = _cosprod_subspace(VN, VT)

    SIGNAL_subspace_similarity = 100.0 * float(sig_sim.mean().item())
    NOISE_subspace_similarity  = 100.0 * float(noi_sim.mean().item())
    SENSAI_score = SIGNAL_subspace_similarity - float(noise_multiplier) * NOISE_subspace_similarity
    
    return float(SIGNAL_subspace_similarity), float(NOISE_subspace_similarity), float(SENSAI_score)
