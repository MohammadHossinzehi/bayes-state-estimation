"""Ready-made motion/measurement models and simulators."""
import numpy as np


def constant_velocity(dt=1.0, q=0.1, r=1.0, dim=1):
    """Position+velocity per axis. Discrete white-noise acceleration model."""
    F1 = np.array([[1, dt], [0, 1]])
    Q1 = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    H1 = np.array([[1.0, 0.0]])
    I = np.eye(dim)
    return np.kron(I, F1), np.kron(I, H1), np.kron(I, Q1), np.eye(dim) * r ** 2


def simulate_linear(F, H, Q, R, x0, steps, rng, dropout=0.0):
    x = np.asarray(x0, float)
    xs, zs = [], []
    Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(len(Q)))
    Lr = np.linalg.cholesky(R)
    for _ in range(steps):
        x = F @ x + Lq @ rng.standard_normal(len(x))
        xs.append(x.copy())
        zs.append(None if rng.random() < dropout else H @ x + Lr @ rng.standard_normal(len(R)))
    return np.array(xs), zs


def simulate_nonlinear(f, h, Q, R, x0, steps, rng):
    x = np.asarray(x0, float)
    Lq, Lr = np.linalg.cholesky(Q), np.linalg.cholesky(R)
    xs, zs = [], []
    for _ in range(steps):
        x = np.atleast_1d(f(x)) + Lq @ rng.standard_normal(len(x))
        xs.append(x.copy())
        zs.append(np.atleast_1d(h(x)) + Lr @ rng.standard_normal(len(R)))
    return np.array(xs), zs
