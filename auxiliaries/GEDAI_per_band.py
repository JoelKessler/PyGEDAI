import torch
from typing import List, Optional, Tuple, Union


from .clean_EEG import clean_eeg
from .SENSAI_fminbnd import sensai_fminbnd
from .SENSAI import sensai
from .create_cosine_weights import create_cosine_weights

def gedai_per_band(
    eeg_data: torch.Tensor,
    srate: float,
    chanlocs,
    artifact_threshold_type,
    epoch_size: float,
    refCOV: torch.Tensor,
    optimization_type: str,
    parallel: bool,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """
    PyTorch port of MATLAB GEDAI_per_band with numerical parity.
    """
    # Input validation
    if eeg_data is None:
        raise ValueError("Cannot process empty data.")
    X = eeg_data.to(device=device, dtype=dtype)
    if X.ndim != 2:
        raise ValueError("Input EEG data must be a 2D matrix (channels x samples).")
    refCOV_t = refCOV.to(device=device, dtype=dtype)
    n_ch, pnts = X.shape

    # Epoching: require integer/even epoch_samples
    epoch_samples_float = float(srate) * float(epoch_size)
    if abs(epoch_samples_float - round(epoch_samples_float)) > 1e-12:
        raise ValueError("srate*epoch_size must yield an integer number of samples.")
    epoch_samples = int(round(epoch_samples_float))
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be positive.")
    if epoch_samples % 2 != 0:
        raise ValueError("epoch_samples must be even so shifting=epoch_samples/2 is integer.")

    num_epochs = int(pnts // epoch_samples)
    X = X[:, : epoch_samples * num_epochs]
    shifting = epoch_samples // 2

    # Reshape to (channels, samples_per_epoch, num_epochs)
    # use unfold along the time dimension to create contiguous epoch windows.
    if num_epochs > 0:
        # (C, T) -> (C, num_epochs, epoch_samples) -> (C, epoch_samples, num_epochs)
        EEGdata_epoched = X.unfold(dimension=1, size=epoch_samples, step=epoch_samples)  # (C, E, S)
        EEGdata_epoched = EEGdata_epoched.permute(0, 2, 1).contiguous()                  # (C, S, E)
    else:
        EEGdata_epoched = torch.zeros((n_ch, epoch_samples, 0), device=device, dtype=dtype)

    # Stream 2: shifted epochs
    if shifting == 0 or X.shape[1] <= 2 * shifting:
        EEGdata_epoched_2 = torch.zeros((n_ch, epoch_samples, 0), device=device, dtype=dtype)
    else:
        X2 = X[:, shifting:-shifting]
        nE2 = int(X2.shape[1] // epoch_samples)
        X2 = X2[:, : nE2 * epoch_samples]
        if nE2 > 0:
            EEGdata_epoched_2 = X2.unfold(1, epoch_samples, epoch_samples).permute(0, 2, 1).contiguous()
        else:
            EEGdata_epoched_2 = torch.zeros((n_ch, epoch_samples, 0), device=device, dtype=dtype)

    _, _, N_epochs = EEGdata_epoched.shape

    # Covariances (per epoch)
    COV = torch.zeros((n_ch, n_ch, N_epochs), device=device, dtype=dtype)
    COV_2 = torch.zeros((n_ch, n_ch, max(N_epochs - 1, 0)), device=device, dtype=dtype)

    for epo in range(N_epochs - 1):
        X1 = EEGdata_epoched[:, :, epo]
        COV[:, :, epo] = _matlab_cov(X1, ddof=1)

        if EEGdata_epoched_2.shape[2] > epo:
            X2e = EEGdata_epoched_2[:, :, epo]
            COV_2[:, :, epo] = _matlab_cov(X2e, ddof=1)

    if N_epochs > 0:
        XN = EEGdata_epoched[:, :, N_epochs - 1]
        COV[:, :, N_epochs - 1] = _matlab_cov(XN, ddof=1)

    # Reference covariance regularization
    regularization_lambda = 0.05
    eps_stability = 1e-12
    evals = torch.linalg.eigvalsh(refCOV_t) # symmetric/hermitian eigenvalues
    mean_eval = float(evals.mean().item())
    mean_eval = max(mean_eval, eps_stability)

    refCOV_reg = (
        (1.0 - regularization_lambda) * refCOV_t
        + regularization_lambda * mean_eval * torch.eye(n_ch, device=device, dtype=dtype)
    )
    refCOV_reg = 0.5 * (refCOV_reg + refCOV_reg.T)

    # GEVD per epoch (ascending eigenvalues)
    Evec = torch.zeros((n_ch, n_ch, N_epochs), device=device, dtype=dtype)
    Eval = torch.zeros((n_ch, n_ch, N_epochs), device=device, dtype=dtype)
    Evec_2 = torch.zeros((n_ch, n_ch, max(N_epochs - 1, 0)), device=device, dtype=dtype)
    Eval_2 = torch.zeros((n_ch, n_ch, max(N_epochs - 1, 0)), device=device, dtype=dtype)

    for i in range(max(N_epochs - 1, 0)):
        Vi, Di = _gevd_chol(COV[:, :, i], refCOV_reg)
        w = torch.diag(Di)
        idx = torch.argsort(w) # ascending
        Evec[:, :, i] = Vi[:, idx]
        Eval[:, :, i] = torch.diag(w[idx])

        if EEGdata_epoched_2.shape[2] > i:
            Vi2, Di2 = _gevd_chol(COV_2[:, :, i], refCOV_reg)
            w2 = torch.diag(Di2)
            idx2 = torch.argsort(w2)
            Evec_2[:, :, i] = Vi2[:, idx2]
            Eval_2[:, :, i] = torch.diag(w2[idx2])

    if N_epochs > 0:
        Vf, Df = _gevd_chol(COV[:, :, N_epochs - 1], refCOV_reg)
        wf = torch.diag(Df)
        idxf = torch.argsort(wf)
        Evec[:, :, N_epochs - 1] = Vf[:, idxf]
        Eval[:, :, N_epochs - 1] = torch.diag(wf[idxf])

    # Artifact threshold determination
    if isinstance(artifact_threshold_type, str) and artifact_threshold_type.startswith("auto"):
        if artifact_threshold_type == "auto+":
            noise_multiplier = 1.0
        elif artifact_threshold_type == "auto":
            noise_multiplier = 3.0
        else:  # 'auto-'
            noise_multiplier = 6.0

        minThreshold, maxThreshold = 0.0, 12.0

        if optimization_type == "parabolic":
            # Bridge inputs to NumPy
            optimal_artifact_threshold, _ = sensai_fminbnd(
                minThreshold, maxThreshold,
                EEGdata_epoched, srate, epoch_size,
                refCOV_t, Eval, Evec,
                noise_multiplier
            )
        elif optimization_type == "grid":
            step = 0.1
            AutomaticThresholdSweep = torch.arange(
                minThreshold, maxThreshold + 1e-12, step, device=device, dtype=dtype
            )

            SIGNAL_subspace_similarity = torch.zeros_like(AutomaticThresholdSweep)
            NOISE_subspace_similarity = torch.zeros_like(AutomaticThresholdSweep)
            SENSAI_score_sweep = torch.zeros_like(AutomaticThresholdSweep)

            for idx, thr in enumerate(AutomaticThresholdSweep):
                S_sig, S_noise, S_score = sensai(
                    EEGdata_epoched, srate, epoch_size, float(thr.item()),
                    refCOV_t, Eval, Evec,
                    noise_multiplier
                )
                SIGNAL_subspace_similarity[idx] = float(S_sig)
                NOISE_subspace_similarity[idx] = float(S_noise)
                SENSAI_score_sweep[idx] = float(S_score)

            smooth_noise = _movmean(NOISE_subspace_similarity, 6)
            diffs = smooth_noise[1:] - smooth_noise[:-1]
            if diffs.numel() > 0:
                cps = _findchangepts_mean(diffs, max_num_changes=2)
                noise_idx = len(AutomaticThresholdSweep) - 1 if len(cps) == 0 else int(cps[0])
            else:
                noise_idx = len(AutomaticThresholdSweep) - 1

            sensai_idx = int(torch.argmax(SENSAI_score_sweep).item())
            optimal_artifact_threshold = (
                AutomaticThresholdSweep[noise_idx].item()
                if sensai_idx > noise_idx
                else AutomaticThresholdSweep[sensai_idx].item()
            )
        else:
            raise ValueError("optimization_type must be 'parabolic' or 'grid'.")

        artifact_threshold = float(optimal_artifact_threshold)
    else:
        # Numeric threshold
        try:
            artifact_threshold = float(artifact_threshold_type)
        except Exception as e:
            raise ValueError("artifact_threshold_type must be 'auto*' or numeric.") from e

    # Clean EEG data
    cleaned_data_1, artifacts_data_1, artifact_threshold_out = clean_eeg(
        EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV_t, Eval, Evec,
        strict_matlab=True, device=device, dtype=dtype
    )
    cleaned_data_2, artifacts_data_2, _ = clean_eeg(
        EEGdata_epoched_2, srate, epoch_size, artifact_threshold, refCOV_t, Eval_2, Evec_2,
        strict_matlab=True, device=device, dtype=dtype
    )

    # Combine streams with cosine weights
    cosine_weights = create_cosine_weights(n_ch, srate, epoch_size, True, device=device, dtype=dtype)
    size_reconstructed_2 = cleaned_data_2.shape[1]
    sample_end = size_reconstructed_2 - shifting

    if size_reconstructed_2 > 0 and shifting > 0:
        cleaned_data_2[:, :shifting] *= cosine_weights[:, :shifting]
        cleaned_data_2[:, sample_end:] *= cosine_weights[:, shifting:]
        artifacts_data_2[:, :shifting] *= cosine_weights[:, :shifting]
        artifacts_data_2[:, sample_end:] *= cosine_weights[:, shifting:]

    cleaned_data = cleaned_data_1.clone()
    artifacts_data = artifacts_data_1.clone()

    if size_reconstructed_2 > 0:
        sl = slice(shifting, shifting + size_reconstructed_2)
        cleaned_data[:, sl] += cleaned_data_2
        artifacts_data[:, sl] += artifacts_data_2

    # Compute final SENSAI score
    _, _, SENSAI_score = sensai(
        EEGdata_epoched, srate, epoch_size, artifact_threshold_out,
        refCOV_t, Eval, Evec, 1.0
    )

    return cleaned_data, artifacts_data, float(SENSAI_score), float(artifact_threshold_out)

# Linear algebra helper functions
def _gevd_chol(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized eigendecomposition using Cholesky of B: A v = lambda B v, with B = L L^H.
    Solve: (L^{-1} A L^{-H}) y = lambda y, v = L^{-H} y.
    Returns (V, D) with D diagonal (2D).
    """
    A = A.to(dtype=torch.float64)
    B = B.to(dtype=torch.float64)

    # Force exact symmetry before Cholesky
    B = 0.5 * (B + B.T)

    # Cholesky (lower)
    L = torch.linalg.cholesky(B)

    # Compute S = L^{-1} A L^{-T} without forming inverses
    # Y = L^{-1} A  via triangular solve; then S = Y L^{-T}
    Y = torch.linalg.solve_triangular(L, A, upper=False)
    S = torch.linalg.solve_triangular(L, Y.T, upper=False).T

    # Force symmetry for stability
    S = 0.5 * (S + S.T)

    # Hermitian EVD (ascending eigenvalues)
    w, Yev = torch.linalg.eigh(S)

    # Back-substitute V = L^{-T} Y
    V = torch.linalg.solve_triangular(L.T, Yev, upper=True)
    D = torch.diag(w)
    return V, D

def _matlab_cov(X: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """
    MATLAB-like covariance for X (n_features, n_samples).
    """
    X = X.to(dtype=torch.float64)
    n_features, n_samples = X.shape
    if n_samples <= ddof:
        raise ValueError(f"n_samples ({n_samples}) must be > ddof ({ddof})")

    X_mean = X.mean(dim=1, keepdim=True)
    X_centered = X - X_mean
    cov = (X_centered @ X_centered.T) / float(n_samples - ddof)
    cov = 0.5 * (cov + cov.T)
    return cov

def _movmean(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Centered moving mean with 'shrink' endpoints (MATLAB smoothdata(...,'movmean',k)).
    x: 1D tensor
    """
    k = int(k)
    x = x.to(dtype=torch.float64)
    n = x.numel()
    if k <= 1 or n == 0:
        return x.clone()
    L = (k - 1) // 2
    R = k - L - 1
    csum = torch.cumsum(torch.cat([x.new_zeros(1), x]), dim=0) # prefix sum with 0
    out = torch.empty(n, dtype=torch.float64, device=x.device)
    for i in range(n):
        a = max(0, i - L)
        b = min(n - 1, i + R)
        s = csum[b + 1] - csum[a]
        out[i] = s / float(b - a + 1)
    return out

def _seg_sse(csum: torch.Tensor, csum2: torch.Tensor, i: int, j: int) -> float:
    """SSE of y[i:j] around its mean using prefix sums (i and j inclusive)."""
    if i > j:
        return 0.0
    n = j - i + 1
    s = (csum[j + 1] - csum[i]).item()
    s2 = (csum2[j + 1] - csum2[i]).item()
    return s2 - (s * s) / float(n)

def _findchangepts_mean(y: torch.Tensor, max_num_changes: int = 2) -> List[int]:
    """
    Mean-shift change-point detection (like MATLAB findchangepts with 'Statistic','mean').
    Returns a sorted list of 0-based split indices t such that segments are:
    [0..t], [t+1..u], [u+1..n-1] (for two changes)
    """
    if max_num_changes != 2:
        raise NotImplementedError("Only max_num_changes=2 is implemented.")
    
    y = y.to(dtype=torch.float64).flatten()
    n = y.numel()
    if n <= 1:
        return []
    csum = torch.cumsum(torch.cat([y.new_zeros(1), y]), dim=0)
    csum2 = torch.cumsum(torch.cat([y.new_zeros(1), y * y]), dim=0)

    # 0 changes
    best0 = _seg_sse(csum, csum2, 0, n - 1)

    # 1 change
    best1 = float("inf")
    t1 = None
    for t in range(0, n - 1):
        cost = _seg_sse(csum, csum2, 0, t) + _seg_sse(csum, csum2, t + 1, n - 1)
        if cost < best1:
            best1, t1 = cost, t

    # 2 changes via DP
    pref1 = [float("inf")] * n
    pref_arg = [-1] * n
    for q in range(1, n - 1):
        best = float("inf")
        arg = -1
        for s in range(0, q):
            cost = _seg_sse(csum, csum2, 0, s) + _seg_sse(csum, csum2, s + 1, q)
            if cost < best:
                best, arg = cost, s
        pref1[q] = best
        pref_arg[q] = arg

    best2 = float("inf")
    t2a = t2b = None
    for u in range(1, n - 1):
        cost = pref1[u] + _seg_sse(csum, csum2, u + 1, n - 1)
        if cost < best2:
            best2 = cost
            t2b = u
            t2a = pref_arg[u]

    # choose k in {0,1,2} with minimal SSE (ties prefer fewer changes)
    cand = [(best0, 0, ()),
            (best1, 1, (t1,)),
            (best2, 2, (t2a, t2b))]
    cand = [c for c in cand if c[0] != float("inf")]
    cand.sort(key=lambda z: (z[0], z[1]))
    _, _, splits = cand[0]
    return sorted(int(s) for s in splits if s is not None)
