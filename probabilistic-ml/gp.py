"""A small Gaussian process regression library written with NumPy and SciPy.

Kernels take inputs of shape (n, d). The GP class fits its hyperparameters by
maximizing the log marginal likelihood and returns predictive means and variances.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm


def _scaled_sqdist(A, B, lengthscales):
    A, B = A / lengthscales, B / lengthscales
    return np.maximum(np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T, 0.0)


def rbf(A, B, variance, lengthscales):
    """Squared exponential kernel: infinitely smooth functions."""
    return variance * np.exp(-0.5 * _scaled_sqdist(A, B, lengthscales))


def matern52(A, B, variance, lengthscales):
    """Matern 5/2 kernel: twice differentiable, rougher than the RBF."""
    r = np.sqrt(5 * _scaled_sqdist(A, B, lengthscales))
    return variance * (1 + r + r**2 / 3) * np.exp(-r)


def periodic(A, B, variance, lengthscales, period=1.0):
    """Periodic kernel (one-dimensional inputs)."""
    d = np.abs(A[:, :1] - B[:, :1].T)
    return variance * np.exp(-2 * np.sin(np.pi * d / period) ** 2 / lengthscales[0] ** 2)


def linear(A, B, variance, lengthscales):
    """Linear kernel: functions that are linear in the inputs (plus a constant mean)."""
    return variance * (A / lengthscales) @ (B / lengthscales).T


KERNELS = {"rbf": rbf, "matern52": matern52, "linear": linear}


class GP:
    """Gaussian process regression with a constant mean and Gaussian noise.

    Hyperparameters (signal variance, one lengthscale per input dimension,
    noise variance and the constant mean) are fitted by maximizing the log
    marginal likelihood with several random restarts.
    """

    def __init__(self, kernel="rbf", ard=True):
        self.kernel_name = kernel
        self.kernel = KERNELS[kernel]
        self.ard = ard

    # parameter vector: [log variance, log lengthscales..., log noise, mean]
    def _unpack(self, theta):
        n_ls = self.d if self.ard else 1
        variance = np.exp(theta[0])
        lengthscales = np.exp(theta[1:1 + n_ls]) * np.ones(self.d)
        noise = np.exp(theta[1 + n_ls]) + 1e-8
        mean = theta[2 + n_ls]
        return variance, lengthscales, noise, mean

    def _neg_log_marginal_likelihood(self, theta):
        variance, lengthscales, noise, mean = self._unpack(theta)
        K = self.kernel(self.X, self.X, variance, lengthscales) + noise * np.eye(self.n)
        try:
            L = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        r = self.y - mean
        alpha = cho_solve(L, r)
        return 0.5 * r @ alpha + np.sum(np.log(np.diag(L[0]))) + 0.5 * self.n * np.log(2 * np.pi)

    def fit(self, X, y, restarts=10, seed=0, fixed=None, noise=None):
        """Fit the hyperparameters.

        `fixed` passes a full parameter vector and skips optimization; `noise` fixes only the
        noise variance (e.g. when it is known from repeated measurements).
        """
        self.X, self.y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float).ravel()
        self.n, self.d = self.X.shape
        n_ls = self.d if self.ard else 1
        if fixed is not None:
            self.theta = np.asarray(fixed, float)
        else:
            rng = np.random.default_rng(seed)
            best = None
            for _ in range(restarts):
                theta0 = np.concatenate([
                    [np.log(self.y.var() + 1e-6) + rng.normal(0, 1)],
                    np.log(np.ptp(self.X, 0)[:n_ls] + 1e-6) + rng.normal(-1, 1, n_ls),
                    [np.log(self.y.var() * 0.1 + 1e-6) + rng.normal(0, 1)],
                    [self.y.mean()],
                ])
                noise_bound = (np.log(noise), np.log(noise)) if noise is not None else (-12, 3)
                if noise is not None:
                    theta0[1 + n_ls] = np.log(noise)
                bounds = [(-10, 5)] + [(-6, 6)] * n_ls + [noise_bound] + [(None, None)]
                res = minimize(self._neg_log_marginal_likelihood, theta0,
                               method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
            self.theta = best.x
        self.variance, self.lengthscales, self.noise, self.mean = self._unpack(self.theta)
        K = self.kernel(self.X, self.X, self.variance, self.lengthscales) + self.noise * np.eye(self.n)
        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, self.y - self.mean)
        self.log_marginal_likelihood = -self._neg_log_marginal_likelihood(self.theta)
        return self

    def predict(self, Xs, include_noise=False):
        """Predictive mean and variance of the latent function (or of a new observation)."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self.kernel(Xs, self.X, self.variance, self.lengthscales)
        mean = self.mean + Ks @ self._alpha
        v = cho_solve(self._L, Ks.T)
        var = np.diag(self.kernel(Xs, Xs, self.variance, self.lengthscales)) - np.sum(Ks * v.T, 1)
        var = np.maximum(var, 1e-12) + (self.noise if include_noise else 0.0)
        return mean, var

    def sample(self, Xs, n_samples=1, seed=0):
        """Joint samples of the latent function at the points Xs."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self.kernel(Xs, self.X, self.variance, self.lengthscales)
        mean = self.mean + Ks @ self._alpha
        cov = self.kernel(Xs, Xs, self.variance, self.lengthscales) - Ks @ cho_solve(self._L, Ks.T)
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(len(Xs)))
        return mean[:, None] + L @ np.random.default_rng(seed).standard_normal((len(Xs), n_samples))

    def params(self):
        return {"signal variance": self.variance, "lengthscales": self.lengthscales,
                "noise variance": self.noise, "mean": self.mean}


def expected_improvement(gp, Xs, best, xi=0.0):
    """Expected amount by which f(x) exceeds the best value observed so far (maximization)."""
    mu, var = gp.predict(Xs)
    sd = np.sqrt(var)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def upper_confidence_bound(gp, Xs, beta=2.0):
    """Optimistic estimate of f(x): mean plus beta standard deviations."""
    mu, var = gp.predict(Xs)
    return mu + beta * np.sqrt(var)
