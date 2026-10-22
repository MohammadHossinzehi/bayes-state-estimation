"""Consistency metrics: RMSE, NEES, NIS and chi-square acceptance bounds."""
import numpy as np


def rmse(est, truth):
    return float(np.sqrt(np.mean((np.asarray(est) - np.asarray(truth)) ** 2)))


def nees(x_est, x_true, P):
    """Normalized estimation error squared per step; mean should be ~ state dim."""
    out = []
    for xe, xt, Pk in zip(x_est, x_true, P):
        e = xe - xt
        out.append(float(e @ np.linalg.solve(Pk, e)))
    return np.array(out)


def nis(innovations, S):
    return np.array([float(y @ np.linalg.solve(s, y)) for y, s in zip(innovations, S)])


def _normal_ppf(p):
    # Acklam-free: bisection on erf, adequate for acceptance intervals
    from math import erf, sqrt
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_bounds(dof, runs=1, level=0.95):
    """Two-sided acceptance interval for the AVERAGE of runs chi2(dof) variables,
    via the Wilson-Hilferty approximation."""
    k = dof * runs
    q = lambda p: k * (1 - 2 / (9 * k) + _normal_ppf(p) * np.sqrt(2 / (9 * k))) ** 3
    a = (1 - level) / 2
    return q(a) / runs, q(1 - a) / runs
