"""Linear Gaussian Kalman filter (Joseph form update) and Rauch-Tung-Striebel smoother."""
import numpy as np


class KalmanFilter:
    """x_k = F x_{k-1} + w, w~N(0,Q);  z_k = H x_k + v, v~N(0,R)."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H, self.Q, self.R = map(np.atleast_2d, (np.asarray(F, float), np.asarray(H, float),
                                                           np.asarray(Q, float), np.asarray(R, float)))
        self.x = np.asarray(x0, float).copy()
        self.P = np.atleast_2d(np.asarray(P0, float)).copy()
        self.last_innovation = None
        self.last_S = None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        z = np.atleast_1d(np.asarray(z, float))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T  # P H^T S^-1 (S symmetric)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        # Joseph form keeps P symmetric positive semidefinite under rounding
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        self.last_innovation, self.last_S = y, S
        return self.x, self.P

    def run(self, zs):
        """Filter a sequence; None entries mean a missed measurement (predict only).
        Returns dict of filtered/predicted means and covariances plus F used."""
        xp, Pp, xf, Pf, nis_vals = [], [], [], [], []
        for z in zs:
            x, P = self.predict()
            xp.append(x.copy()); Pp.append(P.copy())
            if z is not None:
                self.update(z)
                y, S = self.last_innovation, self.last_S
                nis_vals.append(float(y @ np.linalg.solve(S, y)))
            xf.append(self.x.copy()); Pf.append(self.P.copy())
        return {"x_pred": np.array(xp), "P_pred": np.array(Pp),
                "x": np.array(xf), "P": np.array(Pf), "nis": np.array(nis_vals), "F": self.F}


def rts_smooth(res):
    """Rauch-Tung-Striebel fixed-interval smoother over the output of KalmanFilter.run."""
    x, P, xp, Pp, F = res["x"], res["P"], res["x_pred"], res["P_pred"], res["F"]
    n = len(x)
    xs, Ps = x.copy(), P.copy()
    for k in range(n - 2, -1, -1):
        C = np.linalg.solve(Pp[k + 1], F @ P[k]).T  # P_k F^T Pp_{k+1}^-1
        xs[k] = x[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = P[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps
