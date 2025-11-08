# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch

def cov_matlab_like(X: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """
    MATLAB-like covariance with rowvar=False for 2D or batched 3D input.

    Input
    X: (C, S) or (B, C, S) tensor # channels x samples
    ddof: int = 1 # unbiased (divide by S - ddof)

    Returns
    (C, C) if input is (C, S)
    (B, C, C) if input is (B, C, S)

    Explainer:
    Computes a score of how two groups of signals/patterns align/overlap in structure.
    Identify if information is same or completely different.
    """
    if X.dim() not in (2, 3):
        raise ValueError("X must be 2D (C,S) or 3D (B,C,S).")

    X = X.to(torch.float32)

    if X.dim() == 2:
        C, S = X.shape
        if S <= ddof:
            raise ValueError(f"n_samples ({S}) must be > ddof ({ddof})")
        Xc = X - X.mean(dim=1, keepdim=True) # demean over samples
        cov = (Xc @ Xc.transpose(0, 1)) / float(S - ddof)
        return 0.5 * (cov + cov.transpose(0, 1)) # Hermitian symmetrize

    # 3D batched case
    B, C, S = X.shape
    if S <= ddof:
        raise ValueError(f"n_samples ({S}) must be > ddof ({ddof})")
    Xm = X - X.mean(dim=2, keepdim=True) # (B,C,S)
    cov = torch.bmm(Xm, Xm.transpose(1, 2)) / float(S - ddof) # (B,C,C)
    return 0.5 * (cov + cov.transpose(1, 2))
