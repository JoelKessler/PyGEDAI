# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch
from typing import Tuple, Union

from .clean_EEG import clean_eeg
from .subspace_angles import subspace_cosine_product
from .cov import cov_matlab_like

from .. import profiling

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
    dtype: torch.dtype = torch.float32,
    skip_checks_and_return_cleaned_only: bool = False,
    verbose_timing: bool = False
) -> Tuple[float, float, float]:
    """
    Compute SENSAI score and subspace similarities - BATCHED VERSION.
    
    Parameters:
        EEGdata_epoched: Epoched EEG data (channels x samples)
        srate: Sampling rate of the data
        epoch_size: Duration of each epoch in seconds
        artifact_threshold: Threshold for artifact detection
        refCOV: Reference covariance matrix
        Eval: Eigenvalues for each epoch
        Evec: Eigenvectors for each epoch
        noise_multiplier: Multiplier for noise similarity
        top_PCs: Number of top principal components to consider
        device: Device for computation (e.g., 'cpu', 'cuda')
        dtype: Data type for computation
    
    Returns:
        Tuple containing:
            - SIGNAL_subspace_similarity (float)
            - NOISE_subspace_similarity (float)
            - SENSAI_score (float)

    Explainer: 
    1. Clean data into cleaned signal and extracted artifacts using clean_eeg.
    2. Compare patterns to reference template (refCOV). Extract top 3 dominant PCs from refCOV.
    3. Cleaned signal and artifats are split into epochs.
    4. 
        Evaluate similarity between dominant patterns in cleaned signale and compare to reference. (Signal similarity score)
        Evaluate similarity between dominant patterns in artifacts and compare to reference. (Noise similarity score)
    5. Compute SENSAI score as difference between signal similarity and noise similarity (weighted by noise_multiplier).
        High score: Good cleaning of signal.
        Low score: Poor cleaning performance, didn't separate well.
    """
    refCOV = refCOV.to(device=device, dtype=dtype)
    Eval = Eval.to(device=device, dtype=dtype)
    Evec = Evec.to(device=device, dtype=dtype)

    if verbose_timing:
        profiling.mark("sensai_start")
    EEGout_data, EEG_artifacts_data, _ = clean_eeg(
        EEGdata_epoched=EEGdata_epoched.to(device=device, dtype=dtype),
        srate=float(srate),
        epoch_size=float(epoch_size),
        artifact_threshold_in=float(artifact_threshold),
        refCOV=refCOV,
        Eval=Eval,
        Evec=Evec,
        strict_matlab=True,
        device=device,
        dtype=dtype,
        skip_checks_and_return_cleaned_only=False,
        verbose_timing=verbose_timing
    )

    num_chans = refCOV.size(0)
    epoch_samples = int(round(float(srate) * float(epoch_size)))
    top_PCs_eff = min(int(top_PCs), num_chans)

    # Top eigenvectors of reference covariance (descending)
    wT, VT = torch.linalg.eigh(refCOV)
    idxT = torch.argsort(wT, descending=True)
    VT = VT[:, idxT][:, :top_PCs_eff]  # (channels, top_PCs)

    # Validate dimensions
    if EEGout_data.size(0) != num_chans or EEG_artifacts_data.size(0) != num_chans:
        raise ValueError("EEGout/artifacts channel dimension mismatch with refCOV.")

    total_samples = EEGout_data.size(1)
    if total_samples % epoch_samples != 0:
        raise ValueError("Total samples are not divisible by epoch size.")
    num_epochs = total_samples // epoch_samples

    # Reshape to epochs: (C, T) -> (C, S, E) using unfold
    # KEY OPTIMIZATION: Transpose to (E, C, S) for batched processing, single permute
    Sig_ep = EEGout_data.unfold(1, epoch_samples, epoch_samples).permute(1, 0, 2) # (num_epochs, channels, samples)
    Res_ep = EEG_artifacts_data.unfold(1, epoch_samples, epoch_samples).permute(1, 0, 2) # (num_epochs, channels, samples)


    # OPTIMIZATION: Batched covariance computation 
    cov_sig = cov_matlab_like(Sig_ep, ddof=1)  # (num_epochs, channels, channels)
    cov_res = cov_matlab_like(Res_ep, ddof=1)  # (num_epochs, channels, channels)
    if verbose_timing:
        profiling.mark("sensai_cov_done")

    # Regularize to prevent singularity
    eps = 1e-6
    n_ch = cov_sig.shape[1]
    eye = torch.eye(n_ch, device=cov_sig.device, dtype=cov_sig.dtype).unsqueeze(0)
    cov_sig = cov_sig + eps * eye
    cov_res = cov_res + eps * eye

    # OPTIMIZATION: Batched eigenvalue decomposition 
    # torch.linalg.eigh natively supports batched input!
    wS, VS = torch.linalg.eigh(cov_sig)  # wS: (num_epochs, channels), VS: (num_epochs, channels, channels)
    wN, VN = torch.linalg.eigh(cov_res)  # wN: (num_epochs, channels), VN: (num_epochs, channels, channels)
    if verbose_timing:
        profiling.mark("sensai_eigh_done")

    # Sort eigenvalues in descending order and select top_PCs eigenvectors
    idxS = torch.argsort(wS, dim=1, descending=True)  # (num_epochs, channels)
    idxN = torch.argsort(wN, dim=1, descending=True)  # (num_epochs, channels)
    
    # Advanced indexing to select eigenvectors
    # VS has shape (num_epochs, channels, channels), we want (num_epochs, channels, top_PCs)
    batch_idx = torch.arange(num_epochs, device=device).view(-1, 1, 1).expand(num_epochs, num_chans, top_PCs_eff)
    row_idx = torch.arange(num_chans, device=device).view(1, -1, 1).expand(num_epochs, num_chans, top_PCs_eff)
    col_idx_S = idxS[:, :top_PCs_eff].unsqueeze(1).expand(num_epochs, num_chans, top_PCs_eff)
    col_idx_N = idxN[:, :top_PCs_eff].unsqueeze(1).expand(num_epochs, num_chans, top_PCs_eff)
    
    VS_top = VS[batch_idx, row_idx, col_idx_S]  # (num_epochs, channels, top_PCs)
    VN_top = VN[batch_idx, row_idx, col_idx_N]  # (num_epochs, channels, top_PCs)

    # Expand VT for batched comparison: (channels, top_PCs) -> (num_epochs, channels, top_PCs)
    VT_expanded = VT.unsqueeze(0).expand(num_epochs, -1, -1)

    # OPTIMIZATION: Batched subspace similarity computation 
    #sig_sim = _cosprod_subspace_batched(VS_top, VT_expanded)  # (num_epochs,)
    M = torch.bmm(VS_top.transpose(1,2), VT_expanded)  # (batch, k, k)
    sig_sim = torch.abs(torch.linalg.det(M))    
    # Replaced sigsim
    
    noi_sim = subspace_cosine_product(VN_top, VT_expanded) # (num_epochs,)

    # Compute final scores
    SIGNAL_subspace_similarity = 100.0 * float(sig_sim.mean().item())
    NOISE_subspace_similarity = 100.0 * float(noi_sim.mean().item())
    SENSAI_score = SIGNAL_subspace_similarity - float(noise_multiplier) * NOISE_subspace_similarity
    if verbose_timing:
        profiling.mark("sensai_done")

    return float(SIGNAL_subspace_similarity), float(NOISE_subspace_similarity), float(SENSAI_score)
