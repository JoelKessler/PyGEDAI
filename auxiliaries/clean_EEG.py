import torch
import numpy as np
from typing import Tuple, Optional
from .create_cosine_weights import create_cosine_weights


def clean_eeg(
    EEGdata_epoched: torch.Tensor,
    srate: float,
    epoch_size: float,
    artifact_threshold_in: float,
    refCOV: Optional[torch.Tensor],
    Eval: torch.Tensor,
    Evec: torch.Tensor,
    strict_matlab: bool = True,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Clean EEG data using GEDAI methodology.

    Parameters:
    EEGdata_epoched: Epoched EEG data (channels x samples x epochs).
    srate: Sampling rate of the EEG data.
    epoch_size: Duration of each epoch in seconds.
    artifact_threshold_in: Initial artifact threshold.
    refCOV: Reference covariance matrix.
    Eval: Eigenvalues for each epoch.
    Evec: Eigenvectors for each epoch.
    strict_matlab: Enforce MATLAB compatibility.
    device: Device for computation (e.g., 'cpu', 'cuda').
    dtype: Data type for computation.

    Returns:
    Tuple containing cleaned data, artifact data, and the artifact threshold used.
    """
    # Determine complex and real data types based on the input dtype
    ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
    rtype = dtype

    # Convert inputs to PyTorch tensors with the specified device and ctype
    EEG = EEGdata_epoched.to(device=device, dtype=ctype)
    Ev = Eval.to(device=device, dtype=ctype)
    U = Evec.to(device=device, dtype=ctype)

    # Validate input shapes and dimensions
    if EEG.ndim != 3:
        raise ValueError("EEGdata_epoched must be 3D: (num_chans, epoch_samples, num_epochs)")
    if Ev.ndim != 3 or U.ndim != 3:
        raise ValueError("Eval and Evec must be 3D arrays of shape (num_chans, num_chans, num_epochs)")

    num_chans = Ev.shape[0]
    if Ev.shape[1] != num_chans or U.shape[:2] != (num_chans, num_chans):
        raise ValueError("Eval and Evec must be (num_chans, num_chans, num_epochs)")

    if EEG.shape[0] != num_chans:
        raise ValueError("Channel count mismatch between EEGdata_epoched and Eval/Evec")

    num_epochs = Ev.shape[2]
    if U.shape[2] != num_epochs or EEG.shape[2] != num_epochs:
        raise ValueError("num_epochs must match across EEGdata_epoched, Eval, and Evec")

    if num_epochs == 0:
        # Return empty tensors if there are no epochs
        empty = torch.zeros((num_chans, 0), dtype=rtype, device=device)
        return empty, empty, float(artifact_threshold_in)

    # Extract diagonals of the covariance matrices across epochs
    Ev_b = Ev.movedim(2, 0)  # Move epoch axis to the first dimension
    diag_all = Ev_b.diagonal(dim1=1, dim2=2)  # Extract batch diagonals
    diag_all = diag_all.reshape(-1)  # Flatten the diagonals

    # Compute log-magnitudes of eigenvalues
    magnitudes = diag_all.abs()
    positive_mask = magnitudes > 0
    if not bool(torch.any(positive_mask)):
        raise ValueError("All eigenvalue magnitudes are zero; cannot compute log.")

    log_Eig_val_all = torch.log(magnitudes[positive_mask].real) + 100.0

    # Compute the empirical cumulative distribution function (ECDF)
    original_data = torch.unique(log_Eig_val_all)  # Unique sorted values
    n = original_data.numel()
    f = torch.arange(1, n + 1, device=device, dtype=rtype) / float(n)
    transformed_data = f  # ECDF values for unique data

    # Determine artifact threshold from the upper tail of the ECDF
    correction_factor = 1.00
    T1 = correction_factor * (105.0 - float(artifact_threshold_in)) / 100.0

    upper_PIT_threshold = 0.95
    outliers_mask = transformed_data > upper_PIT_threshold
    if bool(torch.any(outliers_mask)):
        Treshold1 = T1 * float(original_data[outliers_mask].min().item())
        threshold_cutoff = float(torch.exp(torch.tensor(Treshold1 - 100.0, device=device, dtype=rtype)).item())
    else:
        if strict_matlab:
            raise ValueError("No values above the 95th percentile; MATLAB would error here.")
        Treshold1 = float("-inf")
        threshold_cutoff = 0.0

    # Prepare buffers for cleaned data and artifacts
    epoch_samples = EEG.shape[1]
    if strict_matlab and (epoch_samples % 2 != 0):
        raise ValueError("epoch_samples must be even to match MATLAB indexing.")
    half_epoch = epoch_samples // 2

    # Generate cosine weights for windowing
    cw = create_cosine_weights(num_chans, srate, epoch_size, True)
    # Ensure cosine weights are on the correct device and dtype
    cw = cw.to(device=device, dtype=rtype)
    if cw.shape != (num_chans, epoch_samples):
        raise ValueError(f"cosine_weights shape {tuple(cw.shape)} != ({num_chans}, {epoch_samples})")

    artifacts = torch.zeros_like(EEG, dtype=ctype, device=device)
    cleaned_epoched = torch.zeros_like(EEG, dtype=ctype, device=device)

    # Process each epoch to clean EEG data
    for i in range(num_epochs):
        Ui = U[:, :, i].clone()  # Eigenvector matrix for the current epoch

        # Zero eigenvectors corresponding to eigenvalues below the threshold
        dvals = Ev[:, :, i].diagonal().abs().real
        mask_keep = dvals >= threshold_cutoff
        Ui[:, ~mask_keep] = 0.0

        epoch_data = EEG[:, :, i]  # EEG data for the current epoch

        # Compute artifact timecourses
        artifacts_timecourses = Ui.conj().transpose(-2, -1) @ epoch_data

        # Solve for artifact contributions using least squares
        sol = torch.linalg.lstsq(U[:, :, i].conj().transpose(-2, -1), artifacts_timecourses).solution

        artifacts[:, :, i] = sol
        cleaned_epoch = epoch_data - sol

        # Apply cosine windowing to the cleaned epoch
        if i == 0:
            cleaned_epoch[:, half_epoch:] = cleaned_epoch[:, half_epoch:] * cw[:, half_epoch:]
        elif i == num_epochs - 1:
            cleaned_epoch[:, :half_epoch] = cleaned_epoch[:, :half_epoch] * cw[:, :half_epoch]
        else:
            cleaned_epoch = cleaned_epoch * cw

        cleaned_epoched[:, :, i] = cleaned_epoch

    # Reshape epoched data into contiguous format
    cleaned_data = cleaned_epoched.permute(0, 2, 1).contiguous().reshape(num_chans, -1).real.to(rtype)
    artifacts_data = artifacts.permute(0, 2, 1).contiguous().reshape(num_chans, -1).real.to(rtype)

    return cleaned_data, artifacts_data, float(artifact_threshold_in)
