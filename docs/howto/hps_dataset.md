# BC PSP HPS Data

This note outlines how to assemble publicly available horizontal point sampling
datasets from the BC Forest Analysis and Inventory Branch (FAIB) compilations.
The goal is to obtain a clean, reproducible subset that mirrors the BAF 12 HPS
workflow used in the Vegetation Resource Inventory (VRI).

## Source

- FTP: `ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/`
  - `psp/` – Provincial Vegetation Resource Inventory permanent sample plots.
  - `non_psp/` – Related compilations for non‑PSP programmes.
- Metadata: `PSP_data_dictionary_20250514.xlsx`, `non_PSP_data_dictionary_20250514.xlsx`
  (download and store checksums alongside scripts).

## Relevant Tables

| File | Purpose | Key Fields |
| ---- | ------- | ---------- |
| `faib_plot_header.csv` | Plot descriptors (one per plot/visit) | `CLSTR_ID`, `PLOT`, `V_BAF`, `MEAS_DT`, `SAMPLE_TYPE` |
| `faib_sample_byvisit.csv` | Plot visit metadata | `CLSTR_ID`, `VISIT`, `FIRST_MSMT`, `MEAS_YR`, `PLOT_CLASS` |
| `faib_tree_detail.csv` | Per-tree measurements (large; chunked download) | `CLSTR_ID`, `PLOT`, `TREE_NUM`, `DBH`, `SP0`, `STATUS_CD`, `TREE_CLASS`, `MEAS_DT` |

Additional summary tables (`faib_compiled_*`) provide aggregated basal area and
heights but are optional for the initial HPS tally pipeline.

## Extraction Recipe

1. **Mirror metadata**: save the data dictionaries and record SHA256 hashes in
   `data/external/psp/CHECKSUMS`.
2. **Filter plots**: load `faib_plot_header.csv` and retain rows with
   `V_BAF == 12` and `SAMPLE_TYPE == "PSP"` (or whichever flag defines the HPS
   plots). Persist a CSV with `CLSTR_ID`, `PLOT`, `MEAS_DT`, and location fields.
3. **Join visit context**: merge `faib_sample_byvisit` on `(CLSTR_ID, PLOT)` to
   identify active measurement cycles (e.g., `FIRST_MSMT == "Y"` for baseline).
4. **Build tallies**: stream `faib_tree_detail.csv` with `pandas.read_csv(..., chunksize=...)`
   selecting the columns above; filter to plots discovered in step 2, keep live
   trees (`STATUS_CD == "L"`), and bin DBH to centimetre midpoints. Output per plot:
   - `dbh_cm` bin centre,
   - `tally` counts,
   - `baf` (12),
   - optional species/stratum attributes for future use.
   Store under `data/examples/hps_baf12/plot_<CLSTR_ID>_<VISIT>.csv`.
5. **Document lineage**: create `data/examples/hps_baf12/README.md` summarising
   the selection criteria, transformation script, and citation requirements.

## Automation TODOs

- [ ] Script in `scripts/prepare_hps_dataset.py` to execute steps 1–4 with CLI options
      for destination directories and chunk sizes.
- [ ] Pytest fixture that downloads a small fixture plot (or caches a pre-processed sample)
      to validate `fit_hps_inventory` against manuscript targets.
- [ ] Update `docs/howto/hps_workflow.md` to reference the new FAIR bundle once the
      subset lands in Git history (or DataLad store).
