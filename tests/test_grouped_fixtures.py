from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dbhdistfit.fitting import fit_inventory
from dbhdistfit.typing import InventorySpec
from dbhdistfit.weighting import hps_expansion_factor
from dbhdistfit.workflows.hps import fit_hps_inventory

FIXTURE_DIR = Path("tests/fixtures")


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / name)


def test_hps_psp_fixture_has_expected_stand_table() -> None:
    fixture = _load_csv("hps_psp_stand_table.csv")
    stand_table = (
        hps_expansion_factor(fixture["dbh_cm"].to_numpy(), baf=12.0) * fixture["tally"].to_numpy()
    )
    np.testing.assert_allclose(fixture["stand_table"].to_numpy(), stand_table)


def test_hps_psp_fixture_weibull_fit_matches_parity() -> None:
    fixture = _load_csv("hps_psp_stand_table.csv")
    results = fit_hps_inventory(
        fixture["dbh_cm"].to_numpy(),
        fixture["tally"].to_numpy(),
        baf=12.0,
        distributions=("weibull",),
    )
    fit = results[0]
    assert fit.distribution == "weibull"
    assert fit.diagnostics.get("method") == "grouped-ls"
    expected = {"a": 2.762844978640213, "beta": 13.778112123083137, "s": 69732.71124303175}
    for key, value in expected.items():
        assert fit.parameters[key] == pytest.approx(value, rel=1e-6)


def test_spruce_fir_grouped_fixture_runs_weibull() -> None:
    fixture = _load_csv("forestfit_spruce_fir_grouped.csv")
    inventory = InventorySpec(
        name="forestfit-spruce-fir",
        sampling="fixed-area",
        bins=fixture["bin_midpoint"].to_numpy(),
        tallies=fixture["count"].to_numpy(),
        metadata={"grouped": True},
    )
    results = fit_inventory(inventory, distributions=["weibull"], configs={})
    fit = results[0]
    assert fit.distribution == "weibull"
    assert fit.diagnostics.get("method") in {"grouped-mle", "grouped-ls"}
    expected = {"a": 2.248935447119741, "beta": 28.610478898705715}
    for key, value in expected.items():
        assert fit.parameters[key] == pytest.approx(value, rel=1e-4)
