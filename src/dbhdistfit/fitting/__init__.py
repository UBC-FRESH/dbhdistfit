"""Shared fitting strategies for dbhdistfit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
from lmfit import Model
from scipy.optimize import curve_fit

from ..distributions import Distribution, get_distribution
from ..typing import FitResult, InventorySpec


Objective = Callable[[np.ndarray, np.ndarray, Mapping[str, float]], float]


@dataclass(slots=True)
class FitConfig:
    distribution: str
    initial: Dict[str, float]
    bounds: Dict[str, tuple[float | None, float | None]] | None = None
    weights: np.ndarray | None = None


def _curve_fit_distribution(
    x: np.ndarray,
    y: np.ndarray,
    distribution: Distribution,
    config: FitConfig,
) -> FitResult:
    """Fit a distribution using SciPy curve_fit with optional weights."""

    def wrapped(x_vals: np.ndarray, *params: float) -> np.ndarray:
        values = dict(zip(distribution.parameters, params, strict=False))
        return distribution.pdf(x_vals, values)

    p0 = [config.initial.get(name, 1.0) for name in distribution.parameters]
    sigma = config.weights if config.weights is not None else None
    params, cov = curve_fit(
        wrapped,
        x,
        y,
        p0=p0,
        sigma=sigma,
        maxfev=int(2e5),
    )
    param_dict = dict(zip(distribution.parameters, params, strict=False))
    fitted = distribution.pdf(x, param_dict)
    residuals = y - fitted
    rss = float(np.sum(np.square(residuals)))
    return FitResult(
        distribution=distribution.name,
        parameters=param_dict,
        covariance=cov,
        gof={"rss": rss},
        diagnostics={"fitted": fitted, "residuals": residuals},
    )


def fit_with_lmfit(
    x: np.ndarray,
    y: np.ndarray,
    distribution: Distribution,
    config: FitConfig,
) -> FitResult:
    """Fit using lmfit Model for more advanced scenarios (e.g., truncation)."""

    def func(x_vals: np.ndarray, **params: float) -> np.ndarray:
        return distribution.pdf(x_vals, params)

    model = Model(func)
    params = model.make_params()
    for name in distribution.parameters:
        start = config.initial.get(name, 1.0)
        params[name].set(value=start)
        if config.bounds and name in config.bounds:
            lower, upper = config.bounds[name]
            params[name].set(min=lower, max=upper)
    weights = config.weights
    result = model.fit(y, params, x=x, weights=weights)
    param_dict = {name: result.params[name].value for name in distribution.parameters}
    cov = result.covar
    rss = float(np.sum(np.square(result.residual)))
    return FitResult(
        distribution=distribution.name,
        parameters=param_dict,
        covariance=cov,
        gof={"rss": rss, "aic": float(result.aic), "bic": float(result.bic)},
        diagnostics={"result": result},
    )


def fit_inventory(
    inventory: InventorySpec,
    distributions: Iterable[str],
    configs: Mapping[str, FitConfig],
    *,
    fitter: Callable[[np.ndarray, np.ndarray, Distribution, FitConfig], FitResult] = _curve_fit_distribution,
) -> list[FitResult]:
    """Fit a collection of candidate distributions to an inventory."""
    x = np.asarray(inventory.bins, dtype=float)
    y = np.asarray(inventory.tallies, dtype=float)
    results: list[FitResult] = []
    for name in distributions:
        dist = get_distribution(name)
        config = configs.get(name, FitConfig(distribution=name, initial={}))
        results.append(fitter(x, y, dist, config))
    return results
