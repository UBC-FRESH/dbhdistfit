"""Workflow for censored or truncated tallies using two-stage scaling."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from ..fitting import FitConfig, fit_inventory, fit_with_lmfit
from ..typing import FitResult, InventorySpec


def _default_initial(scale: float) -> dict[str, float]:
    return {"s": scale}


def fit_censored_inventory(
    dbh_cm: np.ndarray,
    density: np.ndarray,
    *,
    support: tuple[float, float],
    distributions: Iterable[str] = ("weibull", "gamma"),
    configs: Mapping[str, FitConfig] | None = None,
) -> list[FitResult]:
    """Fit complete-form PDFs to censored tallies with a two-stage scaler."""
    dbh = np.asarray(dbh_cm, dtype=float)
    values = np.asarray(density, dtype=float)
    scale_guess = float(values.max() if values.size else 1.0)
    inventory = InventorySpec(
        name="censored-inventory",
        sampling="fixed-area",
        bins=dbh,
        tallies=values,
        metadata={"support": support},
    )
    configs = dict(configs or {})
    for name in distributions:
        config = configs.get(name)
        if config is None:
            config = FitConfig(distribution=name, initial=_default_initial(scale_guess))
            configs[name] = config
        config.bounds = config.bounds or {"s": (1e-6, None)}
    return fit_inventory(inventory, distributions, configs, fitter=fit_with_lmfit)
