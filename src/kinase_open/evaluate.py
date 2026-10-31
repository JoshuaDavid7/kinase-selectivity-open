"""Metrics, selectivity margins, classical baselines, and honest error bars."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .featurize import MORGAN_BITS

TARGETS = ["JAK1", "JAK2", "JAK3", "TYK2"]


@dataclass
class Metrics:
    n: int
    mae: float
    rmse: float
    r2: float
    spearman: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> Metrics:
    """Metrics over finite pairs only.

    R^2 is 1 - SSE/SST against the test mean. A constant equal to that mean has
    R^2 zero; a training-mean constant can score below zero under shift.
    """
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if len(y) < 2:
        return Metrics(n=int(len(y)), mae=float("nan"), rmse=float("nan"), r2=float("nan"), spearman=float("nan"))

    err = y - pred
    sst = float(((y - y.mean()) ** 2).sum())
    rho = spearmanr(y, pred).statistic if len(y) > 2 and np.ptp(y) > 0 and np.ptp(pred) > 0 else float("nan")
    return Metrics(
        n=int(len(y)),
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt((err**2).mean())),
        r2=float(1 - (err**2).sum() / sst) if sst > 0 else float("nan"),
        spearman=float(rho),
    )


def selectivity_margin(y: np.ndarray, target_index: int) -> np.ndarray:
    """pKi at one target minus the best of the others.

    Only defined where the chosen target *and* at least one other are measured;
    everything else is NaN. Reporting a margin computed against missing data is
    the quickest way to invent selectivity that is not there.
    """
    others = [i for i in range(y.shape[1]) if i != target_index]
    with warnings.catch_warnings():
        # A compound measured against none of the other targets is an expected
        # all-NaN slice, not a problem: it becomes a NaN margin just below.
        warnings.simplefilter("ignore", RuntimeWarning)
        best_other = np.nanmax(y[:, others], axis=1)
    with np.errstate(invalid="ignore"):
        margin = y[:, target_index] - best_other
    margin[~np.isfinite(y[:, target_index]) | ~np.isfinite(best_other)] = np.nan
    return margin


def matched_selectivity_margins(
    y: np.ndarray, pred: np.ndarray, target_index: int, complete_panel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compare predictions and observations against exactly the same panel.

    An unmeasured competitor must not enter only the predicted maximum.
    Partial-panel margins describe the measured subset, not full-family selectivity.
    """
    if y.shape != pred.shape:
        raise ValueError("labels and predictions must have the same shape")
    measured = np.isfinite(y)
    masked_pred = np.where(measured, pred, np.nan)
    truth = selectivity_margin(y, target_index)
    estimate = selectivity_margin(masked_pred, target_index)
    # Do not silently shrink a panel if a measured target has no prediction.
    valid = (~measured | np.isfinite(pred)).all(axis=1)
    if complete_panel:
        valid &= measured.all(axis=1)
    return np.where(valid, truth, np.nan), np.where(valid, estimate, np.nan)


def margin_preference_accuracy(y_margin: np.ndarray, pred_margin: np.ndarray) -> tuple[float, float]:
    """How often the sign of the predicted margin is right, and the majority rate.

    The majority rate is the comparison that matters: if 85% of compounds
    favour the same target, a model that always says so scores 85%.
    """
    ok = np.isfinite(y_margin) & np.isfinite(pred_margin)
    if ok.sum() == 0:
        return float("nan"), float("nan")
    truth = y_margin[ok] > 0
    agree = float(((pred_margin[ok] > 0) == truth).mean())
    majority = float(max(truth.mean(), 1 - truth.mean()))
    return agree, majority


def variance_decomposition(y: np.ndarray) -> dict:
    """How much of the spread is selectivity rather than shared potency.

    For compounds measured against every target: `within` is the mean variance
    across a compound's own four values (its selectivity profile), `between` is
    the variance of compound means (how potent it is overall). If `within` is
    small, the panel is mostly measuring one thing.
    """
    complete = y[np.isfinite(y).all(axis=1)]
    if len(complete) < 2:
        return {"n_complete": int(len(complete))}

    within = float(complete.var(axis=1, ddof=0).mean())
    between = float(complete.mean(axis=1).var(ddof=0))
    total = within + between
    return {
        "n_complete": int(len(complete)),
        "within_compound_variance": within,
        "between_compound_variance": between,
        "selectivity_fraction": float(within / total) if total > 0 else float("nan"),
    }


def classical_baselines(
    fp_train: np.ndarray,
    y_train: np.ndarray,
    fp_test: np.ndarray,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Ten fixed baselines: Morgan-only and descriptor-augmented variants.

    Each is fitted per target on that target's observed rows, because the
    sklearn models have no notion of a masked loss.
    """
    # Jaccard is a boolean-set distance. Passing continuous descriptors here
    # silently converts nonzero descriptor values to bits.
    full_train, full_test = fp_train, fp_test
    fp_train, fp_test = fp_train[:, :MORGAN_BITS], fp_test[:, :MORGAN_BITS]
    models = {
        "ridge": lambda: Ridge(alpha=1.0, random_state=None),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed),
        "extra_trees": lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=1, random_state=seed),
        "knn_1": lambda: KNeighborsRegressor(n_neighbors=1, metric="jaccard"),
        "knn_5": lambda: KNeighborsRegressor(n_neighbors=5, metric="jaccard"),
        "knn_5_distance": lambda: KNeighborsRegressor(n_neighbors=5, metric="jaccard", weights="distance"),
        "ridge_descriptors": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest_descriptors": lambda: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed),
        "extra_trees_descriptors": lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=1, random_state=seed),
    }

    out: dict[str, np.ndarray] = {}
    n_targets = y_train.shape[1]

    mean_pred = np.full((len(fp_test), n_targets), np.nan, dtype=float)
    for t in range(n_targets):
        obs = np.isfinite(y_train[:, t])
        if obs.sum():
            mean_pred[:, t] = y_train[obs, t].mean()
    out["train_mean"] = mean_pred

    for name, make in models.items():
        pred = np.full((len(fp_test), n_targets), np.nan, dtype=float)
        for t in range(n_targets):
            obs = np.isfinite(y_train[:, t])
            if obs.sum() < 5:
                continue
            est = make()
            if isinstance(est, Ridge):
                est = Ridge(alpha=1.0)
            x_fit, x_pred = fp_train[obs], fp_test
            if name.endswith("_descriptors"):
                x_fit, x_pred = full_train[obs], full_test
            if isinstance(est, KNeighborsRegressor):
                x_fit, x_pred = x_fit.astype(bool), x_pred.astype(bool)
            est.fit(x_fit, y_train[obs, t])
            pred[:, t] = est.predict(x_pred)
        out[name] = pred
    return out


def select_baseline(y_val: np.ndarray, predictions: dict[str, np.ndarray]) -> str:
    """Select one baseline by validation macro-MAE; never inspect test labels."""
    scores = {}
    for name, pred in predictions.items():
        values = [regression_metrics(y_val[:, t], pred[:, t]).mae for t in range(y_val.shape[1])]
        scores[name] = float(np.nanmean(values))
    finite = {name: score for name, score in scores.items() if np.isfinite(score)}
    if not finite:
        raise ValueError("no baseline has a finite validation score")
    return min(finite, key=lambda name: (finite[name], name))


def paired_mae_difference_ci(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, groups: np.ndarray,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Scaffold bootstrap for MAE(A)-MAE(B) on paired observations.

    Negative favors A. This is an exploratory interval, without correction for
    multiple targets/comparisons; overlap with zero does not establish equivalence.
    """
    ok = np.isfinite(y) & np.isfinite(pred_a) & np.isfinite(pred_b)
    differences = np.abs(y[ok] - pred_a[ok]) - np.abs(y[ok] - pred_b[ok])
    return scaffold_bootstrap_ci(
        differences, np.zeros_like(differences), groups[ok],
        statistic=lambda a, b: float(np.mean(a - b)), n_boot=n_boot, alpha=alpha, seed=seed,
    )


def scaffold_bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    statistic=lambda a, b: float(np.abs(a - b).mean()),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample whole scaffold groups, not individual compounds.

    Compounds in one series are not independent draws. Bootstrapping rows
    treats fifty analogues as fifty observations and produces an interval far
    too narrow to be honest; resampling groups keeps the correlation intact.
    """
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred, groups = y[ok], pred[ok], groups[ok]
    if len(y) < 2:
        return float("nan"), float("nan"), float("nan")

    unique = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)

    point = statistic(y, pred)
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in chosen])
        draws.append(statistic(y[idx], pred[idx]))

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
