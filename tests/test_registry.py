import numpy as np

from dbhdistfit.distributions import (
    GENERALIZED_BETA_DISTRIBUTIONS,
    get_distribution,
    list_distributions,
)


def test_default_registry_contains_core_distributions() -> None:
    names = list(list_distributions())
    assert "weibull" in names
    assert "gamma" in names
    dist = get_distribution("weibull")
    assert dist.parameters[0] == "a"


def test_generalized_beta_distributions_registered() -> None:
    x = np.array([20.0])
    sample_params = {
        "gb1": {"a": 1.2, "b": 100.0, "p": 2.0, "q": 3.0, "s": 1.0},
        "gb2": {"a": 1.1, "b": 90.0, "p": 2.0, "q": 2.5, "s": 1.0},
        "gg": {"a": 1.3, "beta": 30.0, "p": 2.0, "s": 1.0},
        "ib1": {"b": 40.0, "p": 2.0, "q": 3.0, "s": 1.0},
        "ug": {"b": 120.0, "d": 1.5, "q": 2.5, "s": 1.0},
        "b1": {"b": 120.0, "p": 2.0, "q": 3.0, "s": 1.0},
        "b2": {"b": 60.0, "p": 1.5, "q": 2.5, "s": 1.0},
        "sm": {"a": 1.1, "b": 75.0, "q": 2.0, "s": 1.0},
        "dagum": {"a": 1.2, "b": 80.0, "p": 2.0, "s": 1.0},
        "pareto": {"b": 15.0, "p": 2.5, "s": 1.0},
        "p": {"b": 60.0, "p": 2.0, "s": 1.0},
        "ln": {"mu": 3.0, "sigma2": 0.4, "s": 1.0},
        "ga": {"beta": 10.0, "p": 2.0, "s": 1.0},
        "w": {"a": 2.5, "beta": 25.0, "s": 1.0},
        "f": {"u": 5.0, "v": 10.0, "s": 1.0},
        "l": {"b": 80.0, "q": 2.0, "s": 1.0},
        "il": {"b": 35.0, "p": 2.0, "s": 1.0},
        "fisk": {"a": 1.3, "b": 50.0, "s": 1.0},
        "u": {"b": 80.0, "s": 1.0},
        "halfn": {"sigma2": 15.0, "s": 1.0},
        "chisq": {"p": 4.0, "s": 1.0},
        "exp": {"beta": 12.0, "s": 1.0},
        "r": {"beta": 18.0, "s": 1.0},
        "halft": {"df": 6.0, "s": 1.0},
        "ll": {"b": 45.0, "s": 1.0},
    }

    for dist in GENERALIZED_BETA_DISTRIBUTIONS:
        reg = get_distribution(dist.name)
        assert reg.parameters == dist.parameters
        params = sample_params.get(dist.name)
        assert params is not None, f"Missing sample parameters for {dist.name}"
        values = reg.pdf(x, params)
        assert np.all(np.isfinite(values))
