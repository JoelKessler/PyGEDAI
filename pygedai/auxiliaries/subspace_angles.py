# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch

def subspace_cosine_product(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine-product subspace similarity between column spaces of U and V.
    
    Measures how similar two subspaces are by computing the product of cosines of 
    their principal angles. Returns a value between 0 (orthogonal subspaces) and 1 
    (identical subspaces).
    
    Parameters:
    - U: Tensor with orthonormal columns representing first subspace.
         Shape: (n, k) or (batch, n, k)
    - V: Tensor with orthonormal columns representing second subspace.
         Shape: (n, k) or (batch, n, k)
    
    Returns:
    - Similarity score(s):
      * Python float if U and V are 2D
      * Tensor of shape (batch,) if U and V are 3D
    
    Notes:
    - Assumes columns of U and V are orthonormal (e.g., from QR decomposition)
    - Supports both real and complex tensors
    - Uses fast determinant method for square cases, SVD for non-square
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
