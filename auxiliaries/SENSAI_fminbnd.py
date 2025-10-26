import torch
from typing import Tuple, Callable
from .SENSAI import sensai


def fminbnd_brent(
    fun: Callable[[float], float],
    x1: float,
    x2: float,
    xtol: float = 1e-1,
    maxiter: int = 500,
) -> Tuple[float, float, int, dict]:
    """
    Brent's method for scalar function minimization.

    Parameters:
    fun: Objective function to minimize.
    x1: Lower bound of the interval.
    x2: Upper bound of the interval.
    xtol: Tolerance for convergence.
    maxiter: Maximum number of iterations.

    Returns:
    Tuple containing the optimal x, function value at x, exit flag, and additional info.
    """
    # Ensure ascending bounds
    a, b = (float(x1), float(x2)) if x1 < x2 else (float(x2), float(x1))

    # Constants
    CGOLD = 0.3819660112501051  # (3 - sqrt(5)) / 2
    sqrt_eps = float(torch.sqrt(torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64)).item())

    # Start strictly inside (never evaluate endpoints)
    x = w = v = a + CGOLD * (b - a)
    fx = fw = fv = float(fun(x))
    nfev = 1

    e = 0.0  # distance moved on step before last
    d = 0.0

    converged = False
    for _ in range(maxiter):
        xm = 0.5 * (a + b)

        # MATLAB-like tolerance: absolute TolX + 3*|x|*sqrt(eps)
        tol1 = float(xtol) + 3.0 * abs(x) * sqrt_eps
        tol2 = 2.0 * tol1

        # Termination test
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            converged = True
            break

        # Attempt parabolic step if movement from previous step is significant
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d

            parabolic_ok = (
                (abs(p) < abs(0.5 * q * etemp)) and
                (p > q * (a - x)) and (p < q * (b - x))
            )

            if parabolic_ok:
                d = p / q
                u = x + d
                # If parabolic step would land too close to a bound, nudge by tol1
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if (xm - x) >= 0.0 else -tol1
            else:
                # Golden-section step
                e = (a - x) if (x >= xm) else (b - x)
                d = CGOLD * e
        else:
            # Golden-section step
            e = (a - x) if (x >= xm) else (b - x)
            d = CGOLD * e

        # Ensure we move at least tol1
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0.0 else -tol1))
        fu = float(fun(u)); nfev += 1

        # Update bracket and points
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v, w = w, u
                fv, fw = fw, fu
            elif (fu <= fv) or (v == x) or (v == w):
                v, fv = u, fu

    exitflag = 0 if converged else 1
    return x, fx, exitflag, {"nfev": nfev, "method": "brent"}


def sensai_fminbnd(
    minThreshold: float,
    maxThreshold: float,
    EEGdata_epoched: torch.Tensor,
    srate: float,
    epoch_size: float,
    refCOV: torch.Tensor,
    Eval: torch.Tensor,
    Evec: torch.Tensor,
    noise_multiplier: float,
    TolX: float = 1e-1,
    Display: int = 0,
) -> Tuple[float, float]:
    """MATLAB-style wrapper: returns (optimalThreshold, maxSENSAIScore)."""

    def objective(artifact_threshold: float) -> float:
        # minimize negative SENSAI
        _, _, score = sensai(
            EEGdata_epoched=EEGdata_epoched,
            srate=srate,
            epoch_size=epoch_size,
            artifact_threshold=float(artifact_threshold),
            refCOV=refCOV,
            Eval=Eval,
            Evec=Evec,
            noise_multiplier=noise_multiplier,
        )
        return -float(score)

    xopt, fval, ierr, info = fminbnd_brent(
        objective, float(minThreshold), float(maxThreshold),
        xtol=float(TolX), maxiter=500
    )

    if Display:  # optional, quiet by default
        print(f"[sensai_fminbnd] iters≈{info['nfev']} fevals, "
              f"exitflag={ierr}, xopt={xopt}, fmin={fval}")

    return float(xopt), float(-fval)
