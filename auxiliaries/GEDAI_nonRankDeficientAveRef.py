# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch

def gedai_non_rank_deficient_avg_ref(eeg: torch.Tensor) -> torch.Tensor:
    """
    Apply a non-rank-deficient average reference to EEG data.

    The method subtracts the channel mean while preserving full rank by
    dividing by (n_ch + 1). Input eeg is expected shape (n_channels, n_samples).

    Explainer:
    Removes common background noise in EEG data from all channels. (Shared noise accross channels)
    By dividing by (n_ch + 1) accounts for reference electrode and prevents creating rank deficiency problems.
    """
    n_ch = eeg.size(0)
    return eeg - (eeg.sum(dim=0, keepdim=True) / (n_ch + 1.0)) # matlab uses n_ch other possible (n_ch + 1.0)
