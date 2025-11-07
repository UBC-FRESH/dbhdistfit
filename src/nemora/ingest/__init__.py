"""Ingestion/ETL scaffolding for Nemora.

This package will host DatasetSource/TransformPipeline abstractions and
connectors for public forest inventory datasets (BC FAIB, FIA, etc.).
"""

from __future__ import annotations

__all__ = ["DatasetSource", "TransformPipeline"]


class DatasetSource:  # pragma: no cover - placeholder until implemented
    """Describe a raw inventory dataset provider (to be implemented)."""

    pass


class TransformPipeline:  # pragma: no cover - placeholder until implemented
    """Placeholder for ETL pipelines converting raw sources into Nemora tables."""

    pass
