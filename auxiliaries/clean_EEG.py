import torch
import numpy as np
from typing import Tuple, Optional
from .create_cosine_weights import create_cosine_weights
import profiling

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
    dtype: torch.dtype = torch.float32,
    skip_checks_and_return_cleaned_only: bool = False
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
    rtype = dtype
    
    if profiling and hasattr(profiling, 'mark'):
        profiling.mark("clean_eeg_start")

    EEG = EEGdata_epoched
    Ev = Eval
    U = Evec
    
    # Validate inputs
    if EEG.ndim != 3:
        raise ValueError("EEGdata_epoched must be 3D: (num_chans, epoch_samples, num_epochs)")
    if Ev.ndim != 3 or U.ndim != 3:
        raise ValueError("Eval and Evec must be 3D arrays")
    
    num_chans = Ev.size(0)
    num_epochs = Ev.size(2)
    
    if num_epochs == 0:
        empty = torch.zeros((num_chans, 0), dtype=rtype, device=device)
        return empty, empty, float(artifact_threshold_in)
    
    # Extract diagonals (already batched in original)
    Ev_b = Ev.movedim(2, 0)
    diag_all = Ev_b.diagonal(dim1=1, dim2=2).reshape(-1)
    
    # Treat non-finite eigenvalues as zero magnitude
    magnitudes = diag_all.abs()
    if not torch.isfinite(magnitudes).all():
        magnitudes = torch.nan_to_num(magnitudes, nan=0.0, posinf=0.0, neginf=0.0)
    if magnitudes.max() == 0:
        # Graceful no-op: return input as "clean", zero artifacts
        # EEG: (C, S, E) -> (C, S*E)
        print("Graceful no-op: all eigenvalues are zero or non-finite.")
        X = EEG.permute(0, 2, 1).reshape(EEG.size(0), -1).contiguous()
        empty = torch.zeros_like(X)
        return X, empty, float(artifact_threshold_in)

    positive_mask = magnitudes > 0
    log_Eig_val_all = torch.log(magnitudes[positive_mask].real) + 100.0
    
    # ECDF computation
    original_data = torch.unique(log_Eig_val_all)
    n_unique = original_data.numel()
    f = torch.arange(1, n_unique + 1, device=device, dtype=rtype) / float(n_unique)
    
    # Threshold computation
    correction_factor = 1.00
    T1 = correction_factor * (105.0 - float(artifact_threshold_in)) / 100.0
    upper_PIT_threshold = 0.95
    outliers_mask = f > upper_PIT_threshold
    
    if bool(torch.any(outliers_mask)):
        Treshold1 = T1 * float(original_data[outliers_mask].min().item())
        threshold_cutoff = float(torch.exp(torch.tensor(Treshold1 - 100.0, device=device, dtype=rtype)).item())
    else:
        if strict_matlab:
            raise ValueError("No values above 95th percentile")
        threshold_cutoff = 0.0
    
    epoch_samples = EEG.size(1)
    if strict_matlab and (epoch_samples % 2 != 0):
        raise ValueError("epoch_samples must be even")
    
    half_epoch = epoch_samples // 2
    
    # Cosine weights
    cw = create_cosine_weights(num_chans, srate, epoch_size, True)
    cw = cw.to(device=device, dtype=rtype)
    
    # OPTIMIZED: BATCHED EPOCH PROCESSING
    # Reshape for batching: (num_epochs, num_chans, ...)
    U_batched = U.permute(2, 0, 1)  # (num_epochs, num_chans, num_chans)
    EEG_batched = EEG.permute(2, 0, 1)  # (num_epochs, num_chans, samples)
    Ev_batched = Ev.permute(2, 0, 1)  # (num_epochs, num_chans, num_chans)
    
    # Compute masks for all epochs at once
    dvals_batched = Ev_batched.diagonal(dim1=1, dim2=2).abs().real
    mask_keep_batched = dvals_batched >= threshold_cutoff
    profiling.mark("clean_eeg_masks_computed")

    # Detect epochs where all components are masked (degenerate)
    components_kept_per_epoch = mask_keep_batched.sum(dim=1) # (num_epochs,)
    bad_epochs = components_kept_per_epoch == 0 # (num_epochs,)

    if bad_epochs.any():
        good_epochs = ~bad_epochs

        # Start with a copy of input; we'll fill cleaned values for good epochs
        cleaned_batched = EEG_batched.clone()

        if good_epochs.any():
            # Process only good epochs through the normal path
            U_good = U_batched[good_epochs] # (G, C, C)
            EEG_good = EEG_batched[good_epochs] # (G, C, S)
            mask_good = mask_keep_batched[good_epochs] # (G, C)

            U_modified_good = U_good.clone()
            U_modified_good = U_modified_good * mask_good.unsqueeze(1) # zero masked components

            U_modified_H_good = U_modified_good.conj().transpose(-2, -1) # (G, C, C)
            artifacts_timecourses_good = torch.bmm(U_modified_H_good, EEG_good) # (G, C, S)  [bmm shape constraints: (b,n,m) x (b,m,p) → (b,n,p)]  # docs: torch.bmm
            # Solve least squares per epoch batch
            U_H_good = U_good.conj().transpose(-2, -1) # (G, C, C)
            sol_good = torch.linalg.lstsq(U_H_good, artifacts_timecourses_good).solution  # (G, C, S)  # docs: torch.linalg.lstsq

            profiling.mark("clean_eeg_lstsq_done")

            cleaned_good = EEG_good - sol_good # (G, C, S)
            cleaned_batched[good_epochs] = cleaned_good

            if not skip_checks_and_return_cleaned_only:
                sol_batched = torch.zeros_like(EEG_batched) # (E, C, S)
                sol_batched[good_epochs] = sol_good
        else:
            # All epochs are bad: true per-epoch graceful no-op, keep originals
            if not skip_checks_and_return_cleaned_only:
                sol_batched = torch.zeros_like(EEG_batched) # artifacts are zeros
    else:
        # All epochs have at least one kept component: original batched path
        U_modified = U_batched.clone()
        U_modified = U_modified * mask_keep_batched.unsqueeze(1)

        U_modified_H = U_modified.conj().transpose(-2, -1)
        artifacts_timecourses = torch.bmm(U_modified_H, EEG_batched)

        U_H = U_batched.conj().transpose(-2, -1)
        sol_batched = torch.linalg.lstsq(U_H, artifacts_timecourses).solution
        profiling.mark("clean_eeg_lstsq_done")

        cleaned_batched = EEG_batched - sol_batched
        
    if num_epochs == 1:
        pass  # No windowing for single epoch
    elif num_epochs == 2:
        cleaned_batched[0, :, half_epoch:] *= cw[:, half_epoch:]
        cleaned_batched[1, :, :half_epoch] *= cw[:, :half_epoch]
    else:
        # First epoch
        cleaned_batched[0, :, half_epoch:] *= cw[:, half_epoch:]
        # Middle epochs (vectorized!)
        cleaned_batched[1:-1] *= cw.unsqueeze(0)
        # Last epoch
        cleaned_batched[-1, :, :half_epoch] *= cw[:, :half_epoch]
    
    # RESHAPE AND RETURN
    cleaned_epoched = cleaned_batched.permute(1, 0, 2)
    cleaned_data = cleaned_epoched.contiguous().reshape(num_chans, -1).real.to(rtype)
    profiling.mark("clean_eeg_reshaped")
    
    if not skip_checks_and_return_cleaned_only:
        artifacts_batched = sol_batched.permute(1, 0, 2)
        artifacts_data = artifacts_batched.contiguous().reshape(num_chans, -1).real.to(rtype)
        return cleaned_data, artifacts_data, float(artifact_threshold_in)
    
    return cleaned_data, None, None
