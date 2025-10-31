import torch
from typing import Union

def subspace_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute principal angles (in radians) between column subspaces of A and B.

    This function calculates the principal angles between the column spaces of two matrices A and B.
    The angles are returned in ascending order, and the function supports both real and complex inputs.

    Parameters:
    - A: A torch.Tensor representing the first matrix. Columns should be orthonormal.
    - B: A torch.Tensor representing the second matrix. Columns should be orthonormal.

    Returns:
    - A torch.Tensor containing the principal angles in radians as a column vector (k, 1).

    Notes:
    - The function assumes that the columns of A and B are orthonormal.
    - The computation is performed in float32 or complex128 precision for numerical stability.
    """
    if A.size(0) != B.size(0):
        raise ValueError("A and B must have the same number of rows (ambient dimension).")

    # Promote dtype to ensure MATLAB-like behavior for complex inputs
    if A.is_complex() or B.is_complex():
        dtype = torch.complex128
    else:
        dtype = torch.float32

    device = A.device
    A = A.to(device=device, dtype=dtype)
    B = B.to(device=device, dtype=dtype)

    # Compute the Gram matrix G = A' * B (conjugate transpose for complex inputs)
    G = A.conj().transpose(-2, -1) @ B if A.is_complex() else A.transpose(-2, -1) @ B

    # Compute singular values of G to derive principal angles
    s = torch.linalg.svdvals(G)  # Singular values are sorted in descending order
    s = torch.clamp(s.real, -1.0, 1.0)  # Clamp values to [-1, 1] to avoid numerical drift
    angles = torch.acos(s)  # Compute angles in radians
    angles, _ = torch.sort(angles)  # Sort angles in ascending order
    return angles.unsqueeze(1)  # Return as a column vector (k, 1)
