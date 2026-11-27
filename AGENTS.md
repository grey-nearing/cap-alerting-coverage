# AGENTS.md — Machine-Readable Engineering Specification

Target repository: `cap-alerting-coverage` (Apache-2.0)
Package import name: `cap_alerting_coverage` (`src/cap_alerting_coverage/`)
CLI entrypoints: `cap-gap`, `cap-alerting-coverage`, `python3 -m cap_alerting_coverage.cli`

This document is written for autonomous coding and verification agents. For human narrative context, read `README.md` and `docs/`.

---

## 1. Pipeline DAG & Execution Tiers

```text
[Tier 2: Raw Upstream Extraction — optional, requires external credentials/stores & [tier2] extra]
  ingestion/schema.sql + alerthub_ingestor.py + idmc_ingestor.py + translator.py
      │
      ├──► 01_build_catalogs.py ──► gadm41_adm1.parquet, catalog_without_gs_*
      │          │
      │          └──► 09_build_groundsource_catalog_b.py ──► catalog_with_gs_*
      │
      ├──► 02_extract_cap_alerts.py ──► cap_gid1_daily_2020_2025.parquet,
      │                                  cap_gid0_daily_2020_2025.parquet,
      │                                  cap_country_onboarding_2021_2025.csv,
      │                                  cap_country_hazard_counts_2021_2025.csv
      │
      └──► 03a_extract_ffm.py         ──► ffm_gid1_daily_2021_2025.parquet
           03b_extract_geoglows_v2.py ──► geoglows_v2_retro_gid1_daily_2021_2025.npz,
                                          geoglows_v2_oper_gid1_multilead_2024_2025.npz
           03c_extract_dhm.py         ──► dhm_10d_gid1_multilead_2021_2025.npz
           03d_extract_glofas_v5.py   ──► glofas_v5_gid1_daily_2021_2025.npz

================================================================================
[Tier 1: Deterministic Manuscript Reproduction from Zenodo Archive — CPU-only]
  Inputs: $CAP_GAP_ZENODO_DIR/{01_event_catalogs,02_admin1_exceedance_matrices,
                               03_summary_results,04_validation_deidentified}
          + data_inputs/{Supplementary_Data_1_*, ne_110m_admin_0_countries.geojson}
      │
      └──► cap-gap reproduce-tier1  [scripts/04_score_benchmark.py]
               ├──► scoring.run_full_scoring()
               │        ──► pr_curves_summary.csv, cap_continuous_pr.csv, dhm_ffm_2d_grid.csv,
               │            lead_window_sweep.csv, single_lead_decay.csv, sensitivity_audits.csv,
               │            ffm_gs_slices_pr.csv
               └──► manuscript_outputs.generate_manuscript_tables_and_figures()
                        ├──► gauge_verification.run_gauge_verification()
                        │        ──► table2_gauge_vs_catalogs.csv (partial pass-through),
                        │            stream_gauge_nse_summary.csv (Tier-1 pass-through)
                        ├──► validation_audit.run_validation_audit()
                        │        ──► ed_table4_validation_summary.csv,
                        │            validation_confusion_counts.csv,
                        │            validation_spatial_precision.csv
                        └──► table1_summary.csv (exact McNemar), ed_table1_cohort_d_35.csv,
                             ed_table3_sensitivity_matrix.csv, ed_table5_full_benchmark.csv,
                             ed_table6a_annual_onboarding.csv, ed_table6b_late_onboarders.csv,
                             country_coverage_and_gains_150.csv, cap_country_hazard_counts_2021_2025.csv,
                             reproduction_summary.json, 13 publication PNG figures (300 DPI)

  Figure-only re-render:
      └──► cap-gap build-figures  [scripts/05_generate_manuscript_outputs.py]
               └──► manuscript_outputs.render_figures_from_results() ──► 13 PNG figures
```

---

## 2. Module Ownership Map

| Path | Responsibility |
|---|---|
| `src/cap_alerting_coverage/config.py` | Date grid (`2020-12-15`..`2025-12-31`, `D_TOTAL=1843`), return periods `[1.01,2,5,10,20,25,50,100]`, FFM thresholds, `LEGACY_EXCLUDED_19`, `ISO_UNIFY`, `default_data_inputs_dir`, `ProjectPaths` env/arg resolution |
| `src/cap_alerting_coverage/scoring.py` | Tier-1 vectorized benchmark evaluation (`score_boolean_matrix`), valid-date alignment, warning episode collapsing, PR curves, Pareto front, lead-time sweeps, sensitivity audits |
| `src/cap_alerting_coverage/manuscript_outputs.py` | Table 1 (with exact McNemar *p*), ED Tables 1/3/5/6a/6b, per-country summary, and all 13 matplotlib figures (Figs 1–4, ED Figs 1–5; `ed-fig-1.png` panel `a` driven dynamically from `ed_table4_validation_summary.csv`) |
| `src/cap_alerting_coverage/gauge_verification.py` | Table 2 & multi-model stream-gauge NSE compilation (Tier-1 pass-through for raw gauge columns + recomputed DHM catalog cols; Tier-2 full raw gauge parquet support) |
| `src/cap_alerting_coverage/validation_audit.py` | Deterministic N=400 de-identified confusion matrix, spatial precision, Wald/Wilson/CP CIs, and ED Table 4 (requires explicit de-identified input file; no hard-coded fallback counts) |
| `src/cap_alerting_coverage/keywords.py` | Multilingual flood/storm exclusion (`EXCL_RE`), permissive (`PERM_RE`), restrictive (`REST_RE`), multi-hazard regexes, OASIS `score64` (`1..64`) |
| `src/cap_alerting_coverage/clustering.py` | `collapse_warning_episodes(max_gap_days=2)`, `compute_waterfall_population(pop_rule="priority_fallback"\|"max"\|"max_with_killed_fallback")`, `cluster_gid1_episodes(gap_days=7)`, `merge_strictly_nested_catalog_b` |
| `src/cap_alerting_coverage/stats_utils.py` | Exact binomial McNemar, Clopper–Pearson, Wald, two-proportion z-test, seeded bootstrap utilities (`n_resamples=1000, seed=42`), Bulletin 17C LP3 fitting |
| `src/cap_alerting_coverage/schemas.py` | Strict input schema validators & privacy/non-redistribution column/cell blockers |
| `src/cap_alerting_coverage/zenodo_fetch.py` | `compute_sha256`, `safe_extract_zip` (zip-slip & symlink guards), `verify_zenodo_checksums`, `download_zenodo_archive` |
| `src/cap_alerting_coverage/cli.py` | Subcommands: `reproduce-tier1`, `verify-parity`, `build-figures`, `verify-zenodo`, `fetch-zenodo`, `audit-validation`, plus `--version` |
| `src/cap_alerting_coverage/ingestion/*` | Tier-2 SQLite/PostgreSQL DDL (`schema.sql` loaded via `importlib.resources`), SQLAlchemy helpers (`db.py`), ISO-3/OID resolver (`country_resolver.py`), Alert Hub/IDMC ingestors, provider-agnostic translator |
| `src/cap_alerting_coverage/pipeline/*` | Tier-2 upstream extractors (`build_catalogs`, `extract_cap_alerts`, `extract_ffm`, `extract_geoglows_v2`, `extract_dhm`, `extract_glofas_v5`, `build_groundsource_catalog_b`, `blosc_decode`; lazy optional imports for `fsspec`/`zstandard`) |

---

## 3. Strict Input & Output Schemas

### 3.1 `01_event_catalogs/*.parquet`
* Files: `catalog_without_gs_all_185_2021_2025.parquet`, `catalog_without_gs_2021_2025.parquet`, `catalog_with_gs_all_185_2021_2025.parquet`, `catalog_with_gs_2021_2025.parquet`
* Required columns:
  * `event_id: str`
  * `gid_1: str` (non-null GADM v4.1 Admin-1 ID)
  * `country_iso3: str` (non-null ISO-3 after `ISO_UNIFY`)
  * `start_date: datetime64[ms]` (timestamp unit normalized via `pd.to_datetime`)
  * `end_date: datetime64[ms]` (`end_date >= start_date`)
  * `has_idmc: bool`, `has_montandon: bool`, `has_groundsource: bool`
  * `waterfall_pop: float64` (`>= 0.0`, non-null)
* Additional cohort boolean columns present in staged catalogs: `in_cohort_d_35`, `in_eval_150`, `in_audited_185`, `in_orig_excl_19`.
* Optional index: `gadm41_adm1_index.parquet` with `(gid_1: str, country_iso3: str)` (no geometry column).

### 3.2 `02_admin1_exceedance_matrices/`
* `cap_gid1_daily_2020_2025.parquet`:
  * `gid_1: str`, `country_iso3: str`, `alert_date: timestamp[us]` (normalized via `pd.to_datetime`)
  * `perm_max_sev: int8` (`0..4`), `rest_max_sev: int8` (`0..4`)
  * `perm_max_score64: int16` (`0..64`), `rest_max_score64: int16` (`0..64`)
  * `n_perm: int32`, `n_rest: int32`
  * **Invariant:** `(max_sev >= s) == (max_score64 >= 16*(s-1)+1)` for `s in {1, 2, 3, 4}`.
* `cap_gid0_daily_2020_2025.parquet`: same columns without `gid_1`.
* `ffm_gid1_daily_2021_2025.parquet`: `gid_1: str`, `country_iso3: str`, `valid_date: timestamp[ms]` (normalized via `pd.to_datetime`), `max_prob: float32` (`[0.0, 1.0]`).
* Discrete riverine tier cubes (`.npz`, `uint8` values in `0..8` where `1..8` map to `[1.01, 2, 5, 10, 20, 25, 50, 100]` yr RP):
  * `dhm_10d_gid1_multilead_2021_2025.npz`: `tiers` shape `(N_gid1, 1843, 11)`, `gid_1`, `country_iso3`, `issue_dates`. (Optional `dhm_0d_gid1_daily_2021_2025.npz` equals `dhm_10d[:, :, 0]`.)
  * `geoglows_v2_retro_gid1_daily_2021_2025.npz`: `tiers` shape `(N_gid1, 1843)`, `gid_1`, `country_iso3`, `valid_dates`.
  * `geoglows_v2_oper_gid1_multilead_2024_2025.npz`: `tiers` shape `(N_gid1, 640, 4, 15)` (`products = ["EnsMedian", "EnsMean", "EnsP75", "HRES"]`), `gid_1`, `country_iso3`, `issue_dates`.
  * `glofas_v5_gid1_daily_2021_2025.npz`: `tiers` shape `(N_gid1, 1843)`, `gid_1`, `country_iso3`, `valid_dates`.
* **Edge case:** GADM v4.1 assigns `gid_1 == "?"` to one unnamed unit in `UKR` (`22` Catalog B events, `2` CAP daily rows, `1,843` FFM rows, `0` Catalog A events); retain verbatim for exact index alignment.

### 3.3 `03_summary_results/` & `04_validation_deidentified/`
* `SHA256SUMS.txt` indexes `51` uploadable files (`LICENSE.txt`, `zenodo.json`, `DATA_DICTIONARY.md`, `Supplementary_Data_1_Global_Alerting_Authorities_Audit_185_Jurisdictions.csv`, plus `01_`–`04_`).
* `03_summary_results/` contains the 16 `PARITY_TABLES` CSVs plus 3 static upstream audit artifacts (`catalog_a_subnational_resolution_by_country.csv`, `montandon_nonhuman_unit_audit.csv`, `montandon_and_demographics_audit_summary.json`).
* `04_validation_deidentified/validation_n400_deidentified.csv`: exactly 400 rows × 30 strictly boolean/categorical/integer columns defined in `schemas.DEIDENTIFIED_VALIDATION_COLUMNS`. Zero raw CAP text, URLs, emails, XML tags, UUIDs, exact dates, or reviewer notes; any non-numeric/non-boolean cell `> 64` characters triggers `ValueError`.

---

## 4. Exact Verification Commands

Run all commands from repository root (use an external virtualenv or `/tmp` caches to keep the tree spotless):

```bash
# 1. Static lint (must exit 0)
ruff check src tests scripts

# 2. Offline unit + synthetic Tier-1 integration + clean-room privacy tests (must exit 0)
PYTHONPATH=src pytest -m "not zenodo" -v

# 3. Real Zenodo archive verification & parity test (when CAP_GAP_ZENODO_DIR is available)
CAP_GAP_ZENODO_DIR=/path/to/zenodo_archive PYTHONPATH=src pytest -m zenodo -v

# 4. End-to-end Tier-1 CLI run + full parity check (16 summary tables + 2 validation tables)
cap-gap reproduce-tier1 \
  --zenodo-dir /path/to/zenodo_archive \
  --results-dir /tmp/cap_gap_results \
  --figures-dir /tmp/cap_gap_figures \
  --verify-checksums
cap-gap verify-parity \
  --actual-dir /tmp/cap_gap_results \
  --reference-dir /path/to/zenodo_archive \
  --atol 1e-3
```

---

## 5. Mathematical Invariants (Do Not Break)

1. **Population Waterfall Default:** `compute_waterfall_population` defaults to `pop_rule="priority_fallback"` (`num_affected > 0` → `displacements > 0` → `num_killed`). Keep `pop_rule="max"` (pure `max(aff, disp)`) and `pop_rule="max_with_killed_fallback"` strictly opt-in for sensitivity audits.
2. **Valid-Date Forecast Alignment:** Multi-lead forecasts issued at day `t_issue` with lead `L` are ALWAYS placed on valid day `t_valid = t_issue + L`. Never score on issue date.
3. **Warning Episode Collapsing:** Default `max_gap_days=2` (consecutive active warning days with gap `<= 3` days merge into one episode).
4. **Strict Catalog Nesting:** Every Catalog A event appears in Catalog B with identical `event_id`, `gid_1`, `start_date`, `end_date`, and `waterfall_pop`; unmatched Groundsource episodes have `waterfall_pop == 0.0`.
5. **Exact McNemar Test:** Always use two-sided exact binomial test (`scipy.stats.binomtest(b, b + c, 0.5)`), never asymptotic chi-squared approximations for small discordant counts.
6. **Deterministic Bootstrap Utilities:** `stats_utils.bootstrap_proportion_ci` and `paired_bootstrap_diff_ci` default to `n_resamples = 1000`, `seed = 42` (`numpy.random.default_rng(42)`). *(Provided as utility helpers; benchmark summary CSVs emit exact counts, proportions, and McNemar p-values.)*

---

## 6. Forbidden Actions & Clean-Room Guardrails

* **NEVER** commit raw binary datasets (`.parquet`, `.gpkg`, `.npz`, `.npy`, `.grib*`, `.nc`, `.sqlite`, `.ndjson.gz`) into Git.
* **NEVER** commit raw CAP XML/headlines/descriptions/URLs/senders, raw third-party disaster records, raw GADM geometries, or raw stream-gauge hydrographs.
* **NEVER** hardcode API keys, bearer JWTs, database passwords, or machine-specific absolute paths.
* **NEVER** modify `LICENSE` away from canonical Apache-2.0 or weaken `tests/test_clean_room_privacy.py` / `schemas.FORBIDDEN_PUBLIC_COLUMNS`.

---

## 7. Common Failure Modes & Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Required dataset '...'` | `CAP_GAP_ZENODO_DIR` unset or pointing to wrong subfolder | Set `--zenodo-dir` or `CAP_GAP_ZENODO_DIR` to the Zenodo archive root containing `01_event_catalogs/`, `02_admin1_exceedance_matrices/`, etc. |
| `FileNotFoundError: Checksum manifest not found` | `--verify-checksums` or `verify-zenodo` run on dir lacking `SHA256SUMS.txt` | Point `--zenodo-dir` at the verified Zenodo archive root containing `SHA256SUMS.txt`. |
| `FileNotFoundError: De-identified validation table not found` | `audit-validation` or `run_validation_audit` called without valid input file | Pass `--input-csv` or `--zenodo-dir` (no hard-coded fallback counts exist). |
| `AssertionError: Expected 185 jurisdictions` | Running on synthetic subset without `--allow-subset` | Pass `--allow-subset` (CLI) or `strict_185=False` (Python API) for synthetic fixtures; keep `strict_185=True` for real runs. |
| `ValueError: Privacy/governance violation` | Table contains raw text/URL/geometry column or cell `> 64` chars | Drop forbidden columns before exporting public tables; verify against `schemas.py`. |
| `ImportError` / `ModuleNotFoundError` for `sqlalchemy`, `fsspec`, or `zstandard` | Running Tier-2 DB/extraction modules without `[tier2]` extra | Install with `pip install -e ".[tier2]"`. Tier 1 never requires `[tier2]`. |
| `FileNotFoundError` on `Supplementary_Data_1_*.csv` or `ne_110m_admin_0_countries.geojson` | Running outside repo with custom data location | Install package wheel cleanly (ships `_data_inputs/`) or pass `--data-inputs-dir` / `CAP_GAP_DATA_INPUTS_DIR`. |
| `test_clean_room_privacy.py` failure | Stray raw binary, secret, host home path, or non-canonical `LICENSE` in tracked files | Remove stray artifacts or forbidden patterns and re-run `pytest tests/test_clean_room_privacy.py`. |
