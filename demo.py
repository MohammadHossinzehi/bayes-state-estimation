"""Bearing-and-range tracking of a turning target: EKF vs UKF vs particle filter."""
import numpy as np
from bayesfilters import (ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter,
                          simulate_nonlinear, rmse)

dt, w = 1.0, 0.05
def f(x):  # coordinated turn with known rate
    s, c = np.sin(w * dt), np.cos(w * dt)
    return np.array([x[0] + s / w * x[1] - (1 - c) / w * x[3], c * x[1] - s * x[3],
                     x[2] + (1 - c) / w * x[1] + s / w * x[3], s * x[1] + c * x[3]])
def h(x):
    return np.array([np.hypot(x[0], x[2]), np.arctan2(x[2], x[0])])

Q = np.diag([0.01, 0.05, 0.01, 0.05]); R = np.diag([4.0, (np.pi / 180 * 2) ** 2])
rng = np.random.default_rng(7)
x0 = np.array([100.0, 1.0, 50.0, -1.0])
xs, zs = simulate_nonlinear(f, h, Q, R, x0, 150, rng)
P0 = np.diag([25, 1, 25, 1.0]); start = x0 + rng.multivariate_normal(np.zeros(4), P0)

ekf = ExtendedKalmanFilter(f, h, Q, R, start, P0, angle_idx=(1,))
ukf = UnscentedKalmanFilter(f, h, Q, R, start, P0, angle_idx=(1,))
Lq = np.linalg.cholesky(Q)
def ll(z, p):
    r = z - np.stack([np.hypot(p[:, 0], p[:, 2]), np.arctan2(p[:, 2], p[:, 0])], 1)
    r[:, 1] = (r[:, 1] + np.pi) % (2 * np.pi) - np.pi
    return -0.5 * np.sum(r * r / np.diag(R), 1)
pf = ParticleFilter(2000, lambda g, n: start + g.multivariate_normal(np.zeros(4), P0, n),
                    lambda g, p: np.array([f(x) for x in p]) + g.standard_normal(p.shape) @ Lq.T,
                    ll, rng=np.random.default_rng(1))
est = {"EKF": [], "UKF": [], "PF": []}
for z in zs:
    ekf.predict(); ekf.update(z); ukf.predict(); ukf.update(z); pf.predict(); pf.update(z)
    est["EKF"].append(ekf.x.copy()); est["UKF"].append(ukf.x.copy()); est["PF"].append(pf.mean())
for k, v in est.items():
    v = np.array(v)
    print(f"{k}: position RMSE {rmse(v[:, [0, 2]], xs[:, [0, 2]]):.3f}")
