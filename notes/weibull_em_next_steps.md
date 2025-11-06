# Grouped Weibull EM / Fixtures TODO

Temporary development TODO list (see ROADMAP detailed notes for the tracked plan):

- [x] Digitise or extract the spruce–fir grouped tallies from Zhang et al. (2011) and store them
      under `tests/fixtures/forestfit_spruce_fir_grouped.csv` (columns: `bin_lower`, `bin_upper`, `count`).
- [x] Add the manuscript PSP tally (`examples/hps_baf12/4000002_PSP1_v1_p1.csv`) to the fixtures with
      expanded stand-table counts so grouped regression tests can consume it directly.
- [ ] Wire regression tests that load those fixtures, run `fit_hps_inventory`, and assert RSS/parameter
      tolerances for both the LS fallback and the grouped Newton path (flagged).
- [ ] Introduce a configuration toggle (CLI flag + FitConfig option) that allows us to switch between
      LS and grouped-MLE prior to making the grouped path the default.
