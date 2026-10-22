import numpy as np
import pytest
from bayesfilters import *


def cv_setup(seed=0, steps=200, **kw):
    F, H, Q, R = constant_velocity(q=0.05, r=2.0)
    rng = np.random.default_rng(seed)
    xs, zs = simulate_linear(F, H, Q, R, [0, 1], steps, rng, **kw)
    return F, H, Q, R, xs, zs


def test_scalar_kalman_matches_closed_form():
    kf = KalmanFilter([[1]], [[1]], [[0]], [[1]], [0.0], [[1.0]])
    kf.predict(); kf.update([2.0])
    assert kf.x[0] == pytest.approx(1.0) and kf.P[0, 0] == pytest.approx(0.5)


def test_covariance_stays_symmetric_psd():
    F, H, Q, R, xs, zs = cv_setup()
    res = KalmanFilter(F, H, Q, R, [0, 0], np.eye(2) * 100).run(zs)
    for P in res["P"]:
        assert np.allclose(P, P.T)
        assert np.linalg.eigvalsh(P).min() > 0


def test_filter_beats_raw_measurements_and_smoother_beats_filter():
    F, H, Q, R, xs, zs = cv_setup(seed=3)
    res = KalmanFilter(F, H, Q, R, [0, 0], np.eye(2) * 100).run(zs)
    raw = np.array([z[0] for z in zs])
    f_err = rmse(res["x"][20:, 0], xs[20:, 0])
    assert f_err < rmse(raw[20:], xs[20:, 0])
    xsm, _ = rts_smooth(res)
    assert rmse(xsm[20:-5, 0], xs[20:-5, 0]) < f_err


def test_dropouts_grow_uncertainty():
    F, H, Q, R, xs, zs = cv_setup()
    zs = list(zs); zs[100:110] = [None] * 10
    res = KalmanFilter(F, H, Q, R, [0, 0], np.eye(2) * 100).run(zs)
    assert res["P"][109, 0, 0] > 3 * res["P"][99, 0, 0]


def test_consistency_nees_nis_in_bounds():
    runs, steps = 30, 100
    n_all, i_all = [], []
    for s in range(runs):
        F, H, Q, R, xs, zs = cv_setup(seed=100 + s, steps=steps)
        x0 = np.array([0.0, 1.0])
        kf = KalmanFilter(F, H, Q, R, x0, np.eye(2))
        res = kf.run(zs)
        n_all.append(nees(res["x"], xs, res["P"])[10:].mean())
        i_all.append(res["nis"][10:].mean())
    lo, hi = chi2_bounds(2, runs * (steps - 10), 0.99)
    assert lo < np.mean(n_all) < hi * 1.15 + 0.1
    lo, hi = chi2_bounds(1, runs * (steps - 10), 0.99)
    assert lo < np.mean(i_all) < hi


def test_numeric_jacobian_matches_analytic():
    fn = lambda x: np.array([x[0] ** 2 * x[1], np.sin(x[1])])
    x = np.array([1.5, 0.7])
    J = numeric_jacobian(fn, x)
    assert np.allclose(J, [[2 * 1.5 * 0.7, 1.5 ** 2], [0, np.cos(0.7)]], atol=1e-6)


def test_ekf_and_ukf_equal_kf_on_linear_model():
    F, H, Q, R, xs, zs = cv_setup(steps=40)
    P0 = np.eye(2) * 10
    kf = KalmanFilter(F, H, Q, R, [0, 0], P0)
    ekf = ExtendedKalmanFilter(lambda x: F @ x, lambda x: H @ x, Q, R, [0, 0], P0)
    ukf = UnscentedKalmanFilter(lambda x: F @ x, lambda x: H @ x, Q, R, [0, 0], P0)
    for z in zs:
        kf.predict(); kf.update(z); ekf.predict(); ekf.update(z); ukf.predict(); ukf.update(z)
    assert np.allclose(kf.x, ekf.x, atol=1e-5) and np.allclose(kf.x, ukf.x, atol=1e-5)
    assert np.allclose(kf.P, ekf.P, atol=1e-5) and np.allclose(kf.P, ukf.P, atol=1e-5)


def test_ukf_weights_sum_to_one():
    u = UnscentedKalmanFilter(lambda x: x, lambda x: x, np.eye(3), np.eye(3), np.zeros(3), np.eye(3))
    assert u.Wm.sum() == pytest.approx(1.0)


def test_particle_filter_matches_kalman_on_linear_gaussian():
    F, H, Q, R, xs, zs = cv_setup(seed=5, steps=60)
    kf = KalmanFilter(F, H, Q, R, [0, 1], np.eye(2)); res = kf.run(zs)
    Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(2))
    ll = lambda z, p: -0.5 * ((z[0] - p[:, 0]) ** 2) / R[0, 0]
    pf = ParticleFilter(20000, lambda g, n: np.array([0, 1.0]) + g.standard_normal((n, 2)),
                        lambda g, p: p @ F.T + g.standard_normal(p.shape) @ Lq.T, ll,
                        rng=np.random.default_rng(2))
    for z in zs:
        pf.predict(); pf.update(z)
    assert np.allclose(pf.mean(), res["x"][-1], atol=0.5)
    assert pf.resample_count > 0


def test_particle_weights_normalised_and_ess_bounds():
    pf = ParticleFilter(100, lambda g, n: g.standard_normal((n, 1)), lambda g, p: p,
                        lambda z, p: -0.5 * (z - p[:, 0]) ** 2)
    pf.update(3.0)
    assert pf.weights.sum() == pytest.approx(1.0)
    assert 1 <= pf.ess() <= 100


def test_nonlinear_trackers_converge():
    import runpy, io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path("demo.py")
    for line in buf.getvalue().strip().splitlines():
        assert float(line.split()[-1]) < 8.0, line
