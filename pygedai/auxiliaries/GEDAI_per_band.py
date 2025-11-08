# License: PolyForm Noncommercial License 1.0.0 — see LICENSE for full terms.

import torch
from typing import List, Tuple, Union
from .. import profiling


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
    TolX: float = 1e-1,
    maxiter: int = 500,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    skip_checks_and_return_cleaned_only: bool = False,
    verbose_timing: bool = False,
):
    """
    PyTorch port of MATLAB GEDAI_per_band with numerical parity (batched, optimized).
    
    Explainer: (Does a single cleaning pass)
    1. Validate format of data.
    2. Split data into epochs.
    3. Shift epochs by half and recompute covariances. (Sliding window)
    4. For each epoch compute how channels correlate with each other (covariance matrices).
    5. Add smoothing to reference covariance to ensure numerical stability.
    6. Fix epochs with no signal (dead epochs) by adding small ridge to covariance.
    7. Extract dominant patterns in epochs with eignevalues/vectors (GEVD).
    8. Determine threshold e.g. automatically using SENSAI score optimization.
    9. Clean EEG data by removing artifactual components.
    10. Reconstruct with crossfading (cosine weights) blending the epochs together smoothly.
    11. Compute final SENSAI score (measure how well the data was cleaned).
    """
    if eeg_data is None:
        raise ValueError("Cannot process empty data.")
    if verbose_timing:
        profiling.mark("gedai_per_band_start")
    if skip_checks_and_return_cleaned_only:
        X = eeg_data
    else:
        X = eeg_data.to(device=device, dtype=dtype)
    if X.ndim != 2:
        raise ValueError("Input EEG data must be a 2D matrix (channels x samples).")
    n_ch = X.size(0)
    pnts = X.size(1)
    epoch_samples_float = srate * epoch_size
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
    if num_epochs > 0:
        EEGdata_epoched = X.unfold(dimension=1, size=epoch_samples, step=epoch_samples).permute(0, 2, 1)
    else:
        EEGdata_epoched = torch.zeros((n_ch, epoch_samples, 0), device=device, dtype=dtype)
    if verbose_timing:
        profiling.mark("epoching_done")
    if shifting == 0 or X.size(1) <= 2 * shifting:
        EEGdata_epoched_2 = torch.zeros((n_ch, epoch_samples, 0), device=device, dtype=dtype)
    else:
        X2 = X[:, shifting:-shifting]
        nE2 = int(X2.size(1) // epoch_samples)
        X2 = X2[:, : nE2 * epoch_samples]
        if nE2 > 0:
            EEGdata_epoched_2 = X2.unfold(1, epoch_samples, epoch_samples).permute(0, 2, 1)
        else:
            EEGdata_epoched_2 = torch.zeros((n_ch, epoch_samples, 0), device=device, dtype=dtype)

    N_epochs = EEGdata_epoched.size(2)

    # batched covariances, shapes (C,C,E)
    if N_epochs > 0:
        COV = _batch_cov_optimized(EEGdata_epoched, ddof=1)
    else:
        COV = torch.zeros((n_ch, n_ch, 0), device=device, dtype=dtype)
    if verbose_timing:
        profiling.mark("cov_computed")
    if EEGdata_epoched_2.size(2) > 0:
        COV_2 = _batch_cov_optimized(EEGdata_epoched_2, ddof=1)
    else:
        COV_2 = torch.zeros((n_ch, n_ch, 0), device=device, dtype=dtype)
    if verbose_timing:
        profiling.mark("cov2_computed")
    # Reference covariance regularization
    regularization_lambda = 0.05
    eps_stability = 1e-12
    evals = torch.linalg.eigvalsh(refCOV)
    mean_eval = float(evals.mean().item())
    mean_eval = max(mean_eval, eps_stability)
    refCOV_reg = (
        (1.0 - regularization_lambda) * refCOV
        + regularization_lambda * mean_eval * torch.eye(n_ch, device=device, dtype=dtype)
    )
    refCOV_reg = 0.5 * (refCOV_reg + refCOV_reg.T)
    
    # HARDENING: dead-epoch ridge
    if N_epochs > 0:
        trace = COV.diagonal(dim1=0, dim2=1).sum(dim=0)  # (E,)
        dead = ~torch.isfinite(trace) | (trace <= 0)
        if bool(dead.any()):
            print(f"[GEDAI] GEVD special case: dead epochs ridge added ({int(dead.sum().item())}/{int(dead.numel())})")
            I = torch.eye(n_ch, device=device, dtype=dtype).unsqueeze(-1) # (n_ch, n_ch, 1)
            mask = dead.to(COV.dtype).view(1, 1, -1)                        # (1, 1, E)
            COV = COV + (eps_stability * mean_eval) * (I * mask)

    if EEGdata_epoched_2.size(2) > 0:
        trace2 = COV_2.diagonal(dim1=0, dim2=1).sum(dim=0)  # (E2,)
        dead2 = ~torch.isfinite(trace2) | (trace2 <= 0)
        if bool(dead2.any()):
            print(f"[GEDAI] GEVD special case: dead epochs ridge added (half-shift) ({int(dead2.sum().item())}/{int(dead2.numel())})")
            I2 = torch.eye(n_ch, device=device, dtype=dtype).unsqueeze(-1) # (n_ch, n_ch, 1)
            mask2 = dead2.to(COV_2.dtype).view(1, 1, -1) # (1, 1, E2)
            COV_2 = COV_2 + (eps_stability * mean_eval) * (I2 * mask2)
    # HARDENING END

    # GEVD (batched via Cholesky of refCOV), with SPD fallback
    if N_epochs > 0:
        Evec, Eval = _gevd_chol_batched(COV, refCOV_reg)
    else:
        Evec = torch.zeros((n_ch, n_ch, 0), device=device, dtype=dtype)
        Eval = torch.zeros((n_ch, n_ch, 0), device=device, dtype=dtype)
    if verbose_timing:
        profiling.mark("gevd_done")

    if COV_2.size(2) > 0:
        Evec_2, Eval_2 = _gevd_chol_batched(COV_2, refCOV_reg)
    else:
        Evec_2 = torch.zeros((n_ch, n_ch, 0), device=device, dtype=dtype)
        Eval_2 = torch.zeros((n_ch, n_ch, 0), device=device, dtype=dtype)
    if verbose_timing:
        profiling.mark("gevd2_done")
    # Artifact threshold determination
    if isinstance(artifact_threshold_type, str) and artifact_threshold_type.startswith("auto"):
        noise_multiplier = dict(auto=3.0, auto_plus=1.0, auto_minus=6.0).get(artifact_threshold_type.replace("+","_plus").replace("-","_minus"), 3.0)
        minThreshold, maxThreshold = 0.0, 12.0
        if optimization_type == "parabolic":
            optimal_artifact_threshold, _ = sensai_fminbnd(
                minThreshold, maxThreshold,
                EEGdata_epoched, srate, epoch_size,
                refCOV, Eval, Evec,
                noise_multiplier, TolX=TolX,
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
                maxiter=maxiter,
                verbose_timing=verbose_timing
            )
            if verbose_timing:
                profiling.mark("threshold_optimized")
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
                    refCOV, Eval, Evec,
                    noise_multiplier, skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
                    verbose_timing=verbose_timing
                )
                SIGNAL_subspace_similarity[idx] = float(S_sig)
                NOISE_subspace_similarity[idx] = float(S_noise)
                SENSAI_score_sweep[idx] = float(S_score)
            # Use movmean_optimized
            smooth_noise = _movmean_optimized(NOISE_subspace_similarity, 6)
            diffs = smooth_noise[1:] - smooth_noise[:-1]
            if diffs.numel() > 0:
                cps = _findchangepts_mean_optimized(diffs, max_num_changes=2)
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
        try:
            artifact_threshold = float(artifact_threshold_type)
        except Exception as e:
            raise ValueError("artifact_threshold_type must be 'auto*' or numeric.") from e
    # Clean EEG data
    cleaned_data_1, artifacts_data_1, artifact_threshold_out = clean_eeg(
        EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Eval, Evec,
        strict_matlab=True, device=device, dtype=dtype, 
        skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
        verbose_timing=verbose_timing
    )
    if verbose_timing:
        profiling.mark("clean_eeg_1_done")
    cleaned_data_2, artifacts_data_2, _ = clean_eeg(
        EEGdata_epoched_2, srate, epoch_size, artifact_threshold, refCOV, Eval_2, Evec_2,
        strict_matlab=True, device=device, dtype=dtype, 
        skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
        verbose_timing=verbose_timing
    )
    if verbose_timing:
        profiling.mark("clean_eeg_2_done")
    cosine_weights = create_cosine_weights(n_ch, srate, epoch_size, True, device=device, dtype=dtype)
    size_reconstructed_2 = cleaned_data_2.size(1)
    sample_end = size_reconstructed_2 - shifting
    if size_reconstructed_2 > 0 and shifting > 0:
        cleaned_data_2[:, :shifting] *= cosine_weights[:, :shifting]
        cleaned_data_2[:, sample_end:] *= cosine_weights[:, shifting:]
        if not skip_checks_and_return_cleaned_only:
            artifacts_data_2[:, :shifting] *= cosine_weights[:, :shifting]
            artifacts_data_2[:, sample_end:] *= cosine_weights[:, shifting:]
    cleaned_data = cleaned_data_1.clone()
    if not skip_checks_and_return_cleaned_only:
        artifacts_data = artifacts_data_1.clone()
    if size_reconstructed_2 > 0:
        sl = slice(shifting, shifting + size_reconstructed_2)
        cleaned_data[:, sl] += cleaned_data_2
        if not skip_checks_and_return_cleaned_only:
            artifacts_data[:, sl] += artifacts_data_2
    if verbose_timing:
        profiling.mark("combine_done")
    if skip_checks_and_return_cleaned_only:
        return cleaned_data, None, None, None
    _, _, SENSAI_score = sensai(
        EEGdata_epoched, srate, epoch_size, artifact_threshold_out,
        refCOV, Eval, Evec, 1.0, skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only,
        verbose_timing=verbose_timing
    )
    if verbose_timing:
        profiling.mark("sensai_done")
    return cleaned_data, artifacts_data, float(SENSAI_score), float(artifact_threshold_out)

def _batch_cov_optimized(X: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """
    Batched MATLAB-like covariance for X (channels, samples, epochs).
    Returns (channels, channels, epochs).

    Explainer:
    Computes covariance matrices for multiple epochs in a batched manner.
    For each epoch subtract mean and compute covariance to understand channel relationships.
    Symmetrizes the covariance matrices to ensure numerical stability. (Rounding errors)
    Return all covariances.
    """
    X = X.to(dtype=torch.float32)
    n_ch, n_samples, n_epochs = X.shape
    if n_samples <= ddof:
        raise ValueError(f"n_samples ({n_samples}) must be > ddof ({ddof})")
    X_batch = X.permute(2, 0, 1)  # (epochs, channels, samples)
    X_mean = X_batch.mean(dim=2, keepdim=True)
    X_centered = X_batch - X_mean
    cov = torch.bmm(X_centered, X_centered.transpose(1, 2)) / float(n_samples - ddof)
    cov = 0.5 * (cov + cov.transpose(1, 2))
    return cov.permute(1, 2, 0)

def _gevd_chol_batched(A_batch: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched generalized eigendecomposition using Cholesky of B.
    A_batch: (n_ch, n_ch, n_epochs)
    B: (n_ch, n_ch) - shared reference covariance
    Returns (V, D) with V: (n_ch, n_ch, n_epochs), D: (n_ch, n_ch, n_epochs)

    Explainer:
    For each epoch finds eigenvalues/vectors that show how different epoch is to reference.
    Big eigenvalue = strong pattern in epoch not in reference (signal).
    Adds small jitter if the math breaks and retries.
    """
    n_ch, _, n_epochs = A_batch.shape
    A_batch = A_batch.to(dtype=torch.float32)
    B = B.to(dtype=torch.float32)
    B = 0.5 * (B + B.T)

    # Robust Cholesky, handle non-SPD cases without an exception and add jitter only when needed
    L, info = torch.linalg.cholesky_ex(B)
    if int(info) != 0:
        print("[GEDAI] GEVD special case: cholesky jitter fallback (refCOV not SPD)")
        I = torch.eye(n_ch, dtype=B.dtype, device=B.device)
        # scale jitter by average diagonal magnitude; try a few times
        eps = float(torch.diag(B).mean().clamp_min(1e-12)) * 1e-6
        for _ in range(3):
            B = B + eps * I
            L, info = torch.linalg.cholesky_ex(B)
            if int(info) == 0:
                break
            eps *= 10.0
        if int(info) != 0:
            raise RuntimeError("refCOV is not SPD even after jitter.")

    A = A_batch.permute(2, 0, 1) # (E, n, n)
    L_expanded = L.unsqueeze(0).expand(n_epochs, -1, -1)
    Y = torch.linalg.solve_triangular(L_expanded, A, upper=False)
    S = torch.linalg.solve_triangular(L_expanded, Y.transpose(1, 2), upper=False).transpose(1, 2)
    S = 0.5 * (S + S.transpose(1, 2))
    w, Yev = torch.linalg.eigh(S)
    V = torch.linalg.solve_triangular(L_expanded.transpose(1, 2), Yev, upper=True)
    D = torch.diag_embed(w)
    return V.permute(1, 2, 0), D.permute(1, 2, 0)

def _movmean_optimized(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Optimized centered moving mean using conv1d.

    Smooths noisy signal by computing moving average.
    Reduces noisyness from SENSAI score calculations when using different thresholds. 
    Makes it more likely to find the true underlying pattern.
    """
    k = int(k)
    x = x.to(dtype=torch.float32)
    n = x.numel()
    if k <= 1 or n == 0:
        return x.clone()
    kernel = torch.ones(1, 1, k, dtype=torch.float32, device=x.device) / k
    L = (k - 1) // 2
    R = k - L - 1
    x_padded = x.view(1, 1, -1)
    x_padded = torch.nn.functional.pad(x_padded, (L, R), mode='replicate')
    out = torch.nn.functional.conv1d(x_padded, kernel, padding=0)
    out = out.view(-1)
    for i in range(min(L, n)):
        a = max(0, i - L)
        b = min(n - 1, i + R)
        out[i] = x[a:b+1].mean()
    for i in range(max(0, n - R), n):
        a = max(0, i - L)
        b = min(n - 1, i + R)
        out[i] = x[a:b+1].mean()
    return out

def _findchangepts_mean_optimized(y: torch.Tensor, max_num_changes: int = 2) -> List[int]:
    """
    Optimized mean-shift change-point detection.

    Explainer:
    Places where signal suddenly changes. Splits signal into more and more segments.
    Until it finds the best split points that minimize overall error.
    Returns indices where abrupt changes occur. (Jumps in signal values. Find spike locations.)
    """
    if max_num_changes != 2:
        raise NotImplementedError("Only max_num_changes=2 is implemented.")
    y = y.to(dtype=torch.float32).flatten()
    n = y.numel()
    if n <= 1:
        return []
    csum = torch.cumsum(torch.cat([y.new_zeros(1), y]), dim=0)
    csum2 = torch.cumsum(torch.cat([y.new_zeros(1), y * y]), dim=0)

    def seg_sse_batch(starts, ends):
        n_segs = ends - starts + 1
        sums = csum[ends + 1] - csum[starts]
        sums2 = csum2[ends + 1] - csum2[starts]
        return sums2 - (sums * sums) / n_segs.float()
    
    best0 = ((csum2[n] - csum2[0]) - (csum[n] - csum[0])**2 / n).item()
    t_range = torch.arange(0, n - 1, device=y.device)
    left_sse = seg_sse_batch(torch.zeros_like(t_range), t_range)
    right_sse = seg_sse_batch(t_range + 1, torch.full_like(t_range, n - 1))
    costs_1 = left_sse + right_sse
    best1_idx = torch.argmin(costs_1)
    best1 = costs_1[best1_idx].item()
    t1 = best1_idx.item()
    pref1 = torch.full((n,), float("inf"), dtype=torch.float32)
    pref_arg = torch.full((n,), -1, dtype=torch.long)
    for q in range(1, n - 1):
        s_range = torch.arange(0, q)
        left_sse_2 = torch.tensor([seg_sse_batch(torch.tensor([0]), torch.tensor([s])).item() for s in s_range], dtype=torch.float32)
        mid_sse_2 = torch.tensor([seg_sse_batch(torch.tensor([s + 1]), torch.tensor([q])).item() for s in s_range], dtype=torch.float32)
        costs_2 = left_sse_2 + mid_sse_2
        best_idx = torch.argmin(costs_2)
        pref1[q] = costs_2[best_idx]
        pref_arg[q] = best_idx
    best2 = float("inf")
    t2a = t2b = None
    for u in range(1, n - 1):
        cost = pref1[u] + seg_sse_batch(torch.tensor([u + 1]), torch.tensor([n - 1])).item()
        if cost < best2:
            best2 = cost
            t2b = u
            t2a = pref_arg[u].item()
    cand = [(best0, 0, ()), (best1, 1, (t1,)), (best2, 2, (t2a, t2b))]
    cand = [c for c in cand if c[0] != float("inf")]
    cand.sort(key=lambda z: (z[0], z[1]))
    _, _, splits = cand[0]
    return sorted(int(s) for s in splits if s is not None)

