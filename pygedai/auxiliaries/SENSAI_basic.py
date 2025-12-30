# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch
from typing import Tuple, Union
from .. import profiling
from .subspace_angles import subspace_cosine_product
from .cov import cov_matlab_like

def sensai_basic(
    signal_data: torch.Tensor,
    noise_data: torch.Tensor,
    srate: float,
    epoch_size: float,
    refCOV: torch.Tensor,
    NOISE_multiplier: float,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    verbose_timing: bool,
    top_PCs: int = 3,
    regularization_lambda: float = 0.05,
) -> Tuple[float, float, float, float, torch.Tensor, int]:
    """
    Compute SENSAI score and subspace similarities.

    Parameters:
    signal_data : torch.Tensor
        Signal data (channels x samples).
    noise_data : torch.Tensor
        Noise data (channels x samples).
    srate : float
        Sampling rate of the data.
    epoch_size : float
        Duration of each epoch in seconds.
    refCOV : torch.Tensor
        Reference covariance matrix.
    NOISE_multiplier : float
        Multiplier for noise similarity.
    top_PCs : int
        Number of top principal components to consider. Default is 3.
    regularization_lambda : float
        Regularization parameter for covariance matrix. Default is 0.05.
    device : Union[str, torch.device]
        Device for computation (e.g., 'cpu', 'cuda').
    dtype : torch.dtype
        Data type for computation.

    Returns:
    Tuple[float, float, float, float, torch.Tensor, int]
        SENSAI score, signal subspace similarity, noise subspace similarity,
        mean ENOVA, the per-epoch ENOVA tensor, and the number of samples per
        epoch actually used during scoring.

    Explainer: 
    Receive both cleaned signal and removed noise.
    Then scores how well signal matches reference pattern and how much it differes from the noise pattern.
    Bigger gap between these two indicate better cleaning performance.
    """

    # Validate input dimensions
    if signal_data.ndim != 2 or noise_data.ndim != 2:
        raise ValueError("signal_data and noise_data must be (num_chans, total_samples).")
    if signal_data.size(0) != refCOV.size(0) or noise_data.size(0) != refCOV.size(0):
        raise ValueError("signal/noise first dim must match refCOV rows (num_chans).")

    num_chans = refCOV.size(0)
    if verbose_timing:
        profiling.mark("sensai_basic_start")

    # Check epoch length
    ep_len = float(srate) * float(epoch_size)
    if abs(ep_len - round(ep_len)) > 1e-9:
        raise ValueError("srate*epoch_size must be an integer number of samples.")
    S = int(round(ep_len))
    if S <= 0:
        raise ValueError("Epoch size must span at least one sample.")

    total_samples = min(signal_data.size(1), noise_data.size(1))
    total_epochs = total_samples // S
    if total_epochs == 0:
        raise ValueError("Not enough samples to form a single epoch.")
    usable_samples = total_epochs * S
    signal_data = signal_data[:, :usable_samples].contiguous()
    noise_data = noise_data[:, :usable_samples].contiguous()
    E = total_epochs
    epoch_samples_used = S

    if top_PCs > num_chans:
        raise ValueError("top_PCs must be <= number of channels.")

    # Regularize template covariance
    evals = torch.linalg.eigvalsh(refCOV)
    mean_eval = float(evals.mean().item())
    Tref_reg = (1.0 - regularization_lambda) * refCOV + regularization_lambda * mean_eval * torch.eye(
        num_chans, device=device, dtype=dtype
    )
    Tref_reg = 0.5 * (Tref_reg + Tref_reg.T)

    # Compute top eigenvectors of template covariance
    wT, VT = torch.linalg.eigh(Tref_reg)
    if verbose_timing:
        profiling.mark("sensai_basic_template_eig")
    idxT = torch.argsort(wT, descending=True)
    VT = VT[:, idxT][:, :top_PCs]

    # Reshape into epochs
    Sig_ep = signal_data.unfold(1, S, S).permute(0, 2, 1).contiguous()
    Noi_ep = noise_data.unfold(1, S, S).permute(0, 2, 1).contiguous()

    sig_sim = torch.empty(E, device=device, dtype=dtype)
    noi_sim = torch.empty(E, device=device, dtype=dtype)
    enova_per_epoch = torch.empty(E, device=device, dtype=dtype)
    dtype_for_eps = signal_data.dtype if torch.is_floating_point(signal_data) else dtype
    finfo = torch.finfo(dtype_for_eps)

    for ep in range(E):
        # Signal subspace
        X = Sig_ep[:, :, ep]
        cov_sig = cov_matlab_like(X, ddof=1, dtype=dtype)
        wS, VS = torch.linalg.eigh(cov_sig)
        idxS = torch.argsort(wS, descending=True)
        VS = VS[:, idxS][:, :top_PCs]
        sig_sim[ep] = subspace_cosine_product(VS, VT)

        # Noise subspace
        N = Noi_ep[:, :, ep]
        cov_noi = cov_matlab_like(N, ddof=1, dtype=dtype)
        wN, VN = torch.linalg.eigh(cov_noi)
        idxN = torch.argsort(wN, descending=True)
        VN = VN[:, idxN][:, :top_PCs]
        noi_sim[ep] = subspace_cosine_product(VN, VT)

        original_epoch = X + N
        var_original = torch.var(original_epoch.reshape(-1), unbiased=True)
        var_noise = torch.var(N.reshape(-1), unbiased=True)
        denom = torch.clamp(var_original, min=finfo.tiny)
        enova_ratio = torch.clamp(var_noise / denom, min=0.0)
        enova_per_epoch[ep] = enova_ratio.to(dtype=dtype)

    if verbose_timing:
        profiling.mark("sensai_basic_epochs_done")
    SIGNAL_subspace_similarity = 100.0 * float(sig_sim.mean().item())
    NOISE_subspace_similarity = 100.0 * float(noi_sim.mean().item())
    SENSAI_score = SIGNAL_subspace_similarity - float(NOISE_multiplier) * NOISE_subspace_similarity
    mean_ENOVA = float(enova_per_epoch.mean().item())

    return (
        float(SENSAI_score),
        float(SIGNAL_subspace_similarity),
        float(NOISE_subspace_similarity),
        mean_ENOVA,
        enova_per_epoch.detach().clone(),
        int(epoch_samples_used),
    )
