"""Bootstrap particle filter with systematic resampling and effective sample size trigger."""
import numpy as np


class ParticleFilter:
    def __init__(self, n, sample_prior, transition, log_likelihood, rng=None, ess_frac=0.5):
        """sample_prior(rng, n)->(n,d); transition(rng, particles)->(n,d); log_likelihood(z, particles)->(n,)."""
        self.rng = rng or np.random.default_rng(0)
        self.n = n
        self.particles = sample_prior(self.rng, n)
        self.logw = np.full(n, -np.log(n))
        self.transition, self.loglik = transition, log_likelihood
        self.ess_frac = ess_frac
        self.resample_count = 0

    @property
    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def ess(self):
        return 1.0 / np.sum(self.weights ** 2)

    def mean(self):
        return self.weights @ self.particles

    def cov(self):
        d = self.particles - self.mean()
        return (self.weights[:, None] * d).T @ d

    def predict(self):
        self.particles = self.transition(self.rng, self.particles)

    def update(self, z):
        self.logw = self.logw + self.loglik(z, self.particles)
        self.logw -= self.logw.max()
        self.logw -= np.log(np.exp(self.logw).sum())
        if self.ess() < self.ess_frac * self.n:
            self.resample()

    def resample(self):
        w = self.weights
        pos = (self.rng.random() + np.arange(self.n)) / self.n  # systematic
        idx = np.minimum(np.searchsorted(np.cumsum(w), pos), self.n - 1)
        self.particles = self.particles[idx]
        self.logw = np.full(self.n, -np.log(self.n))
        self.resample_count += 1
