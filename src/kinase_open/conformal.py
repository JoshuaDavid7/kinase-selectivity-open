"""Split conformal prediction, with a floor on the ensemble spread.

A deep ensemble's standard deviation is a *relative* confidence signal, not a
calibrated one: it tells you which predictions are shakier, but its absolute
width has no coverage guarantee and in practice is far too narrow. Split
conformal fixes the absolute width using a held-out calibration set, and makes
one distribution-free promise: on exchangeable data the interval covers the
truth at least `1 - alpha` of the time.

The floor is the part that is easy to get wrong. With a purely normalized score
`|y - mu| / sigma`, any calibration point whose ensemble happens to agree
(sigma near zero) produces an enormous score, that score lands in the upper
quantile, and every interval inflates. Adding a floor bounds that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_quantile(scores: np.ndarray, alpha: float) -> float:
    """The conformal quantile, with the finite-sample correction.

    The level is ceil((n+1)(1-alpha))/n rather than 1-alpha. That correction is
    what makes the coverage guarantee hold for finite n instead of only
    asymptotically; with few calibration points it is the difference between a
    real 90% and a hopeful one.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration scores")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ConformalCalibrator:
    """Fitted on calibration residuals; applies to new predictions.

    `sigma_floor` is reported because it is a modelling choice, not a constant
    handed down: it is set from the calibration spread itself (a low percentile
    of sigma), so it adapts to how confident this particular ensemble is.
    """

    alpha: float
    q: float
    sigma_floor: float
    n_calibration: int

    def interval(self, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.q * (sigma + self.sigma_floor)
        return mu - half, mu + half

    def half_width(self, sigma: np.ndarray) -> np.ndarray:
        return self.q * (sigma + self.sigma_floor)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "q": self.q,
            "sigma_floor": self.sigma_floor,
            "n_calibration": self.n_calibration,
        }


def fit(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: float = 0.1,
    floor_percentile: float = 10.0,
) -> ConformalCalibrator:
    """Calibrate on held-out predictions.

    Only finite entries are used, so a target a compound was never measured
    against simply does not contribute.
    """
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    y, mu, sigma = y[ok], mu[ok], sigma[ok]
    if len(y) == 0:
        raise ValueError("no finite calibration points")

    floor = float(np.percentile(sigma, floor_percentile))
    if not np.isfinite(floor) or floor <= 0:
        floor = float(np.std(y - mu)) * 1e-3 or 1e-6

    scores = np.abs(y - mu) / (sigma + floor)
    return ConformalCalibrator(
        alpha=alpha,
        q=_finite_quantile(scores, alpha),
        sigma_floor=floor,
        n_calibration=len(y),
    )


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of truths inside the interval, over finite entries."""
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if ok.sum() == 0:
        return float("nan")
    return float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean())


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    ok = np.isfinite(lo) & np.isfinite(hi)
    return float((hi[ok] - lo[ok]).mean()) if ok.sum() else float("nan")


def raw_gaussian_interval(
    mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """The uncalibrated ensemble interval, for contrast.

    This is what you get by treating the ensemble spread as a Gaussian standard
    deviation and reading off z. Reporting its coverage next to the conformal
    one is the point: the gap is the argument for calibrating at all.
    """
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    return mu - z * sigma, mu + z * sigma
