"""Extended and Unscented Kalman filters for x_k = f(x_{k-1}) + w, z_k = h(x_k) + v."""
import numpy as np


def numeric_jacobian(fn, x, eps=1e-6):
    """Central-difference Jacobian of fn at x."""
    x = np.asarray(x, float)
    f0 = np.atleast_1d(fn(x))
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps
        J[:, i] = (np.atleast_1d(fn(x + d)) - np.atleast_1d(fn(x - d))) / (2 * eps)
    return J


class ExtendedKalmanFilter:
    def __init__(self, f, h, Q, R, x0, P0, F_jac=None, H_jac=None, angle_idx=()):
        self.f, self.h = f, h
        self.Fj = F_jac or (lambda x: numeric_jacobian(f, x))
        self.Hj = H_jac or (lambda x: numeric_jacobian(h, x))
        self.Q, self.R = np.atleast_2d(Q), np.atleast_2d(R)
        self.x, self.P = np.asarray(x0, float).copy(), np.atleast_2d(P0).copy()
        self.angle_idx = tuple(angle_idx)  # measurement components that wrap at +-pi

    def predict(self):
        F = self.Fj(self.x)
        self.x = np.atleast_1d(self.f(self.x))
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = self.Hj(self.x)
        y = np.atleast_1d(z) - np.atleast_1d(self.h(self.x))
        for i in self.angle_idx:
            y[i] = (y[i] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return float(y @ np.linalg.solve(S, y))


class UnscentedKalmanFilter:
    """Scaled unscented transform (Wan & van der Merwe) with alpha, beta, kappa."""

    def __init__(self, f, h, Q, R, x0, P0, alpha=1e-1, beta=2.0, kappa=0.0, angle_idx=()):
        self.f, self.h = f, h
        self.Q, self.R = np.atleast_2d(Q), np.atleast_2d(R)
        self.x, self.P = np.asarray(x0, float).copy(), np.atleast_2d(P0).copy()
        n = len(self.x)
        self.n = n
        self.lam = alpha ** 2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 1 / (2 * (n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha ** 2 + beta)
        self.angle_idx = tuple(angle_idx)

    def sigma_points(self, x, P):
        L = np.linalg.cholesky((self.n + self.lam) * P)
        pts = [x] + [x + L[:, i] for i in range(self.n)] + [x - L[:, i] for i in range(self.n)]
        return np.array(pts)

    def predict(self):
        X = np.array([np.atleast_1d(self.f(p)) for p in self.sigma_points(self.x, self.P)])
        self.x = self.Wm @ X
        d = X - self.x
        self.P = (self.Wc[:, None] * d).T @ d + self.Q

    def update(self, z):
        X = self.sigma_points(self.x, self.P)
        Z = np.array([np.atleast_1d(self.h(p)) for p in X])
        zm = self.Wm @ Z
        dz, dx = Z - zm, X - self.x
        S = (self.Wc[:, None] * dz).T @ dz + self.R
        Pxz = (self.Wc[:, None] * dx).T @ dz
        K = np.linalg.solve(S, Pxz.T).T
        y = np.atleast_1d(z) - zm
        for i in self.angle_idx:
            y[i] = (y[i] + np.pi) % (2 * np.pi) - np.pi
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = (self.P + self.P.T) / 2
        return float(y @ np.linalg.solve(S, y))
