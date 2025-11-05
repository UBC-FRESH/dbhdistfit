# Fit HPS Inventories

This guide walks through fitting the horizontal point sampling (HPS) workflow using the
`dbhdistfit` CLI and Python API. It extends the weighted estimator described in the UBC FRESH Lab
manuscript and relies on the probability distributions catalogued in
{doc}`reference/distributions`.

## Prerequisites

- An HPS tally file with columns `dbh_cm` (bin midpoints) and `tally` (per-plot counts).
- The basal area factor (`BAF`) used during cruise design.
- Python environment with `dbhdistfit` installed. One approach:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI Workflow

1. Inspect available distributions:

```bash
dbhdistfit registry
```

2. Fit the HPS inventory with the default candidate set (`weibull`, `gamma`):

```bash
dbhdistfit fit-hps data/hps_tally.csv --baf 2.0
```

Pass additional `--distribution` options (planned) to try alternative PDFs as the CLI evolves.

## Python API Example

```python
import pandas as pd
from dbhdistfit.workflows import fit_hps_inventory

data = pd.read_csv("data/hps_tally.csv")
results = fit_hps_inventory(
    dbh_cm=data["dbh_cm"].to_numpy(),
    tally=data["tally"].to_numpy(),
    baf=2.0,
    distributions=["weibull", "gamma", "gb2"],
)

best = min(results, key=lambda r: r.gof["rss"])
print(best.distribution, best.parameters)
```

`fit_hps_inventory` expands tallies to stand tables, applies the HPS compression factors as
weights, and auto-generates starting values. Override the defaults through `FitConfig` for
specialised scenarios.

## Diagnostics

- Inspect `result.diagnostics["residuals"]` for shape or bias.
- Compare `result.gof` metrics (RSS, AICc when available) across candidates.
- Plot the empirical stand table alongside fitted curves to confirm agreement with the manuscript
  workflow.

## Next Steps

- Expose distribution filters and parameter previews in the CLI.
- Add worked examples for censored inventories and DataLad-backed datasets.
- Integrate notebook tutorials mirroring the published reproducibility bundles.
- Prepare a FAIR HPS dataset bundle (see {doc}`howto/hps_dataset`) sourced from the BC
  PSP compilations to support end-to-end parity tests.
