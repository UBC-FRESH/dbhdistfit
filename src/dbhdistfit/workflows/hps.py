"""Horizontal point sampling fitting workflow."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from ..fitting import FitConfig, fit_inventory
from ..typing import FitResult, InventorySpec
from ..weighting import hps_compression_factor, hps_expansion_factor


def fit_hps_inventory(
    dbh_cm: np.ndarray,
    tally: np.ndarray,
    *,
    baf: float,
    distributions: Iterable[str] = ("weibull", "gamma"),
    configs: Mapping[str, FitConfig] | None = None,
) -> list[FitResult]:
    """Fit HPS tallies using weighted stand-table expansion."""
    dbh = np.asarray(dbh_cm, dtype=float)
    tallies = np.asarray(tally, dtype=float)
    stand_table = tallies * hps_expansion_factor(dbh, baf=baf)
    weights = hps_compression_factor(dbh, baf=baf)
    inventory = InventorySpec(
        name="hps-inventory",
        sampling="hps",
        bins=dbh,
        tallies=stand_table,
        metadata={"baf": baf, "original_tally": tallies},
    )
    configs = dict(configs or {})
    for name in distributions:
        config = configs.get(name)
        if config is None:
            config = FitConfig(distribution=name, initial={"s": stand_table.max()})
            configs[name] = config
        config.weights = weights
    return fit_inventory(inventory, distributions, configs)
