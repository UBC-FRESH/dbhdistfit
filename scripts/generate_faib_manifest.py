#!/usr/bin/env python3
"""Generate trimmed FAIB stand table manifests for testing/documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nemora.ingest.faib import build_stand_table_from_csvs, download_faib_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Output directory for samples and manifest.")
    parser.add_argument(
        "--dataset",
        default="psp",
        choices=["psp", "non_psp"],
        help="FAIB dataset to process.",
    )
    parser.add_argument(
        "--bafs",
        type=float,
        nargs="+",
        default=[4.0, 8.0, 12.0],
        help="BAF values to generate stand tables for.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional existing FAIB download directory (skip fetch when provided).",
    )
    args = parser.parse_args()

    output_dir = args.destination
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = args.source or (output_dir / "raw")
    if args.source is None:
        download_faib_csvs(source_dir, dataset=args.dataset)

    records: list[dict[str, str | float]] = []
    for baf in args.bafs:
        table = build_stand_table_from_csvs(source_dir, baf=baf)
        table_path = output_dir / f"stand_table_baf{int(baf)}.csv"
        table.to_csv(table_path, index=False)
        records.append(
            {
                "dataset": args.dataset,
                "baf": baf,
                "rows": len(table),
                "path": str(table_path.relative_to(output_dir)),
            }
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "faib_manifest.csv", index=False)


if __name__ == "__main__":
    main()
