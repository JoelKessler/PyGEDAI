import torch
from typing import Tuple, Callable
from .SENSAI import sensai
import math

def _sign(x):
    #  https://numpy.org/devdocs/reference/generated/numpy.sign.html
    return -1 if x < 0 else 0 if x == 0 else 1

def _minimize_scalar_bounded(func, x1, x2, args=(),
                             xtol=1e-5, maxiter=500):
    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py#L2195-L2286
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.

        ``0`` : no message printing.

        ``1`` : non-convergence notification messages only.

        ``2`` : print a message on convergence too.

        ``3`` : print iteration results.

    xtol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = float("inf")

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xtol / 3.0
    tol2 = 2.0 * tol1

    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = _sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = _sign(rat) + (rat == 0)
        x = xf + si * max(abs(rat), tol1)
        fu = func(x, *args)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xtol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            break

    fval = fx
    return xf, fval


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
    return x, fx #, exitflag, {"nfev": nfev, "method": "brent"}
 

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

    xopt, fval = _minimize_scalar_bounded(
        objective, minThreshold, maxThreshold,
        xtol=float(TolX), maxiter=500
    )

    return float(xopt), float(-fval)
