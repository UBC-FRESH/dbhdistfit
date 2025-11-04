"""Core package exports for dbhdistfit."""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("dbhdistfit")
except metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
    __version__ = "0.0.0"

from .typing import FitResult, FitSummary, InventorySpec  # noqa: F401
from .workflows.hps import fit_hps_inventory  # noqa: F401
from .workflows.censoring import fit_censored_inventory  # noqa: F401

__all__ = [
    "__version__",
    "FitResult",
    "FitSummary",
    "InventorySpec",
    "fit_hps_inventory",
    "fit_censored_inventory",
]
