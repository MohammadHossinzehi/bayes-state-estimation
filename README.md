# bayes-state-estimation

Recursive Bayesian state estimation from scratch in NumPy: a Joseph-form Kalman filter, Rauch-Tung-Striebel smoother, Extended and Unscented Kalman filters, and a bootstrap particle filter with systematic resampling, plus the NEES/NIS consistency tests used to check that a filter is honest about its own uncertainty. Only dependency is NumPy.

## Why this exists

Tracking a moving object from noisy sensors is the classic state estimation problem, and most implementations quietly get small things wrong: covariances that lose symmetry, angle residuals that wrap badly, particle weights that underflow. This project implements each estimator in about 50 lines, and then *tests the statistics*, not just the output: a filter whose reported covariance does not match its actual error is broken even if the track looks fine.

## What is inside

| Module | Contents |
| --- | --- |
| `bayesfilters/kalman.py` | Linear Kalman filter (Joseph form, missed-measurement support), RTS fixed-interval smoother |
| `bayesfilters/nonlinear.py` | EKF (analytic or central-difference Jacobians), scaled UKF (alpha, beta, kappa), angle-wrapped innovations |
| `bayesfilters/particle.py` | Bootstrap particle filter, log-domain weights, systematic resampling when ESS drops below a threshold |
| `bayesfilters/metrics.py` | RMSE, NEES, NIS, chi-square acceptance bounds (Wilson-Hilferty) |
| `bayesfilters/models.py` | Constant-velocity model builder, linear and nonlinear simulators |
| `demo.py` | Range-bearing tracking of a turning target with EKF vs UKF vs particle filter |

## Run it

```bash
pip install -r requirements.txt
python demo.py        # prints position RMSE for each filter
python -m pytest -q   # 11 tests
```

Minimal usage:

```python
import numpy as np
from bayesfilters import KalmanFilter, rts_smooth, constant_velocity, simulate_linear

F, H, Q, R = constant_velocity(dt=1.0, q=0.05, r=2.0)
xs, zs = simulate_linear(F, H, Q, R, [0, 1], 200, np.random.default_rng(0), dropout=0.1)
res = KalmanFilter(F, H, Q, R, x0=[0, 0], P0=np.eye(2) * 100).run(zs)  # None = missed measurement
smoothed, cov = rts_smooth(res)
```

## Design decisions

- **Joseph form update.** `P = (I-KH) P (I-KH)^T + K R K^T` stays symmetric positive semidefinite under rounding, unlike the textbook `(I-KH)P`. A test checks eigenvalues over 200 steps.
- **No explicit inverses.** Gains come from `np.linalg.solve`, never `inv`.
- **Angle wrapping.** Bearing innovations are wrapped to [-pi, pi) via `angle_idx`; without this an EKF diverges the moment a target crosses the branch cut.
- **Log-domain particle weights**, renormalised every update, so long runs with sharp likelihoods do not underflow to all zeros.
- **Systematic resampling** only when effective sample size falls under half the particle count, which cuts resampling noise versus resampling every step.

## How it is tested

- Closed-form check on a scalar filter (posterior mean 1.0, variance 0.5).
- On a linear model the EKF and UKF must reproduce the Kalman filter to 1e-5, and a 20k-particle filter must land near the Kalman mean.
- The filter must beat raw measurements and the smoother must beat the filter.
- Dropped measurements must inflate covariance.
- Statistical consistency: averaged NEES and NIS over 30 Monte Carlo runs must fall inside chi-square acceptance bounds.
- The nonlinear demo must keep all three trackers under a position RMSE threshold.
