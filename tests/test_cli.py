from typer.testing import CliRunner

from dbhdistfit import __version__
from dbhdistfit.cli import app

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
    assert "41847702.9157" in result.stdout
