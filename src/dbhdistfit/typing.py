"""Shared type declarations and data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ArrayLike = np.ndarray | Sequence[float]
TableLike = pd.DataFrame | Mapping[str, Sequence[float]]


@dataclass(slots=True)
class InventorySpec:
    """Describe a single inventory tally source."""

    name: str
    sampling: str  # e.g. "hps", "fixed-area"
    bins: ArrayLike
    tallies: ArrayLike
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitResult:
    """Container for a single distribution fit."""

    distribution: str
    parameters: Dict[str, float]
    covariance: Optional[np.ndarray] = None
    gof: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitSummary:
    """Aggregate fit outputs over candidate distributions."""

    inventory: InventorySpec
    results: List[FitResult]
    best: Optional[FitResult] = None
    notes: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy data frame summarising candidate results."""
        records: List[Dict[str, Any]] = []
        for result in self.results:
            record = {"distribution": result.distribution}
            record.update(result.parameters)
            record.update({f"gof_{k}": v for k, v in result.gof.items()})
            records.append(record)
        return pd.DataFrame.from_records(records)
