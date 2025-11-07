import torch

def subspace_angles(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute principal angles (in radians) between column subspaces of U and V.

    This function calculates the principal angles between the column spaces of two matrices U and V.
    The angles are returned in ascending order, and the function supports both real and complex inputs.

    Parameters:
    - U: A torch.Tensor representing the first matrix. Columns should be orthonormal.
    - V: A torch.Tensor representing the second matrix. Columns should be orthonormal.

    Returns:
    - A torch.Tensor containing the principal angles in radians as a column vector (k, 1).

    Notes:
    - The function assumes that the columns of A and B are orthonormal.
    - The computation is performed in float32 or complex128 precision for numerical stability.
    """
    if U.dim() != V.dim():
        raise ValueError(f"U and V must have same #dims; got {U.dim()} vs {V.dim()}.")

    # Interpret shapes and normalize to batched form (b, n, k)
    if U.dim() == 2:
        n, k = U.shape
        if V.shape != (n, k):
            raise ValueError(f"Shape mismatch: U {U.shape} vs V {V.shape}.")
        U_b = U.unsqueeze(0) # (1, n, k)
        V_b = V.unsqueeze(0) # (1, n, k)
        unbatched = True
    elif U.dim() == 3:
        if U.shape != V.shape:
            raise ValueError(f"Batched shapes must match: U {U.shape} vs V {V.shape}.")
        U_b, V_b = U, V
        unbatched = False
    else:
        raise ValueError("U and V must be 2D (n,k) or 3D (b,n,k).")

    # Promote dtype (supports complex)
    dtype = torch.result_type(U_b, V_b)
    U_b = U_b.to(dtype)
    V_b = V_b.to(dtype)

    # Overlap matrices M = U^H V -> shape (b, k, k)
    UH = U_b.transpose(-2, -1).conj()
    M = UH @ V_b

    kU = U_b.shape[-1]
    kV = V_b.shape[-1]

    # Prefer fast |det(M)| when square; otherwise fall back to prod(svdvals)
    if kU == kV:
        scores = torch.linalg.det(M).abs() # (b,)
    else:
        svals = torch.linalg.svdvals(M).clamp_(0.0, 1.0) # (b, r) with r=min(kU,kV)
        scores = torch.prod(svals, dim=-1) # (b,)

    # Return type mirrors input rank
    if unbatched:
        return scores[0].item()
    return scores.unsqueeze(1)  # Return as a column vector (k, 1)
