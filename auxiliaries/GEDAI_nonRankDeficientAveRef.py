from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class ChanLoc:
    """
    Represents channel location metadata.

    Attributes:
    labels: Label of the channel.
    type: Type of the channel.
    ref: Reference type for the channel.
    """
    labels: Optional[str] = None
    type: Optional[str] = None
    ref: Optional[str] = None

@dataclass
class EEGLike:
    """
    Represents EEG data structure.

    Attributes:
    data: EEG data tensor (channels x ...).
    nbchan: Number of channels.
    ref: Reference type for the EEG data.
    chanlocs: List of channel locations.
    """
    data: torch.Tensor
    nbchan: int
    ref: Optional[str] = None
    chanlocs: List[ChanLoc] = field(default_factory=list)

def gedai_non_rank_deficient_avg_ref(eeg: EEGLike) -> EEGLike:
    """
    Apply non-rank-deficient average referencing to EEG data.

    Parameters:
    eeg: EEGLike object containing EEG data and metadata.

    Returns:
    EEGLike object with updated data and reference information.
    """
    if eeg.data.ndim < 2:
        raise ValueError("EEG.data must be at least 2D (channels x ...).")
    if eeg.data.size(0) != eeg.nbchan:
        raise ValueError(f"eeg.nbchan ({eeg.nbchan}) must equal eeg.data.shape[0] ({eeg.data.size(0)}).")

    data64 = eeg.data.to(dtype=torch.float64)

    offset = data64.sum(dim=0, keepdim=True) / (eeg.nbchan + 1.0)
    data64.sub_(offset)

    eeg.data = data64
    eeg.ref = "average"

    if len(eeg.chanlocs) < eeg.nbchan:
        eeg.chanlocs += [ChanLoc() for _ in range(eeg.nbchan - len(eeg.chanlocs))]
    for i in range(eeg.nbchan):
        eeg.chanlocs[i].ref = "average"

    return eeg
