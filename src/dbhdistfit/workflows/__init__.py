"""Workflow shortcuts for inventory fitting."""

from __future__ import annotations

from .hps import fit_hps_inventory
from .censoring import fit_censored_inventory

__all__ = ["fit_hps_inventory", "fit_censored_inventory"]
