import re

import pandas as pd
import pytest
from typer.testing import CliRunner

from dbhdistfit import __version__
from dbhdistfit.cli import app
from dbhdistfit.workflows import fit_hps_inventory

runner = CliRunner()


def test_registry_command_lists_distributions() -> None:
    result = runner.invoke(app, ["registry"])
    assert result.exit_code == 0
    assert "weibull" in result.stdout.lower()
    assert "Complete-form Weibull" in result.stdout


def test_version_option() -> None:
    result = runner.invoke(app, ["--verbose"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_fit_hps_command_outputs_weibull_first() -> None:
    result = runner.invoke(
        app,
        [
            "fit-hps",
            "examples/hps_baf12/4000002_PSP1_v1_p1.csv",
            "--baf",
            "12",
        ],
    )
    assert result.exit_code == 0
    assert "weibull" in result.stdout
    assert "gamma" in result.stdout
    assert result.stdout.index("weibull") < result.stdout.index("gamma")

    def extract(metric: str) -> float:
        pattern = rf"│\s*{metric}\s*│\s*([0-9]+\.[0-9]+)"
        match = re.search(pattern, result.stdout)
        assert match, f"{metric} row missing from CLI output"
        return float(match.group(1))

    observed = {dist: extract(dist) for dist in ("weibull", "gamma")}

    data = pd.read_csv("examples/hps_baf12/4000002_PSP1_v1_p1.csv")
    expected_results = fit_hps_inventory(
        data["dbh_cm"].to_numpy(),
        data["tally"].to_numpy(),
        baf=12.0,
        distributions=("weibull", "gamma"),
    )
    expected = {res.distribution: res.gof["rss"] for res in expected_results}

    for dist in expected:
        assert observed[dist] == pytest.approx(expected[dist], rel=1e-6)


def test_fetch_reference_data_dry_run_message() -> None:
    result = runner.invoke(app, ["fetch-reference-data"])  # default dry-run
    assert result.exit_code == 0
    assert "reference dataset" in result.stdout.lower()
    assert "dbhdistfit-data" in result.stdout
    assert "dry-run" in result.stdout.lower()
