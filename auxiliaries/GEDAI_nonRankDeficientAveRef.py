import torch

def gedai_non_rank_deficient_avg_ref(eeg: torch.Tensor) -> torch.Tensor:
    """
    Apply a non-rank-deficient average reference to EEG data.

    The method subtracts the channel mean while preserving full rank by
    dividing by (n_ch + 1). Input eeg is expected shape (n_channels, n_samples).
    """
    n_ch = eeg.size(0)
    return eeg - (eeg.sum(dim=0, keepdim=True) / (n_ch + 1.0)) # matlab uses n_ch other possible (n_ch + 1.0)
