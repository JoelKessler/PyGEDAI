# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch
from typing import Union, Tuple, Dict


_COSINE_TEMPLATE_CACHE: Dict[Tuple[int, bool, torch.dtype], torch.Tensor] = {}

def create_cosine_weights(
    channels: Union[int, float],
    srate: Union[int, float],
    epoch_size: Union[int, float],
    fullshift: bool,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate cosine weights for EEG data processing.
    Used to prevent click artifacts at epoch edges. 
    Start of epoch is weighted 0, end of epoch is weighted 1. (full signal = 1)
    End of epoch ramp down to 0 again.
    Ensures smooth transitions between epochs when concatenated. (Processing in epochs)

    Parameters:
    channels : Union[int, float]
        Number of channels (non-negative integer).
    srate : Union[int, float]
        Sampling rate of the EEG data (positive value).
    epoch_size : Union[int, float]
        Duration of each epoch in seconds (positive value).
    fullshift : bool
        Whether to apply full cosine weighting.
    device : Union[str, torch.device], optional
        Device for computation (e.g., 'cpu', 'cuda'). Default is 'cpu'.
    dtype : torch.dtype, optional
        Data type for computation. Default is torch.float32.

    Returns:
    torch.Tensor
        A tensor of shape (channels, N) with cosine weights.
    """
    N_float = torch.tensor(float(srate) * float(epoch_size), dtype=dtype) 
    if not torch.isfinite(N_float):
        raise ValueError("srate*epoch_size must be finite.")
    N_round_t = torch.round(N_float)
    if torch.abs(N_float - N_round_t) > torch.tensor(1e-12, device=device, dtype=dtype):
        raise ValueError("srate*epoch_size must be integer-valued.")
    N = int(N_round_t.item())
    if N < 0:
        raise ValueError("srate*epoch_size must be non-negative.")

    ch_t = torch.tensor(float(channels), device=device, dtype=dtype)
    if not torch.isfinite(ch_t):
        raise ValueError("channels must be finite.")
    ch_round_t = torch.round(ch_t)
    if torch.abs(ch_t - ch_round_t) > torch.tensor(1e-12, device=device, dtype=dtype) or int(ch_round_t.item()) < 0:
        raise ValueError("channels must be a non-negative integer.")
    C = int(ch_round_t.item())

    if N == 0 or C == 0:
        return torch.zeros((C, N), device=device, dtype=dtype)

    template = _get_cosine_template(N, fullshift, dtype)
    template = template.to(device=device, dtype=dtype)
    return template.unsqueeze(0).expand(C, -1).clone()


def _get_cosine_template(N: int, fullshift: bool, dtype: torch.dtype) -> torch.Tensor:
    key = (N, fullshift, dtype)
    cached = _COSINE_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    device = torch.device("cpu")
    if fullshift:
        u = torch.arange(1, N + 1, device=device, dtype=dtype)
        w = 0.5 - 0.5 * torch.cos(2 * torch.pi * u / float(N))
    else:
        w = torch.zeros(N, device=device, dtype=dtype)
        if N > 1:
            u = torch.arange(1, N, device=device, dtype=dtype)
            w[:-1] = 0.5 - 0.5 * torch.cos(2 * torch.pi * u / float(N - 1))
    _COSINE_TEMPLATE_CACHE[key] = w
    return w
