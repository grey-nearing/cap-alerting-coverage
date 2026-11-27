# Reproducibility Guide

This guide documents how to reproduce every number, statistical test, table, and figure in *"Global flood models can close nearly two-thirds of the warning gap"* under **Tier 1** (from the companion Zenodo derived benchmark archive) and how to reconstruct derived intermediate datasets from raw upstream sources under **Tier 2**.

---

## Environment Requirements

* **Python:** 3.10, 3.11, 3.12, or 3.13 (tested across all four versions in CI)
* **Tier 1 Hardware:** Any standard laptop or workstation (4+ GB RAM, CPU-only). The staged Zenodo archive is **28.1 MB** (`50` uploadable files verified in `SHA256SUMS.txt`, plus `SHA256SUMS.txt` itself and an optional staging helper `UPLOAD_INSTRUCTIONS.md` on disk), and full Tier-1 scoring, sensitivity audits, all summary tables, and all 13 300-DPI publication figures complete in **~2.6 minutes** (`~137 s` scoring/tables + `~21 s` figure rendering).
* **Tier 2 Hardware:** Multi-core Linux workstation with 32+ GB RAM and high-bandwidth object-storage access (for scanning global Zarr/NetCDF/Parquet archives of DHM, FFM, GloFAS v5, GEOGLOWS v2, GADM v4.1, and CAP XML/JSON streams).

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

# Tier 1 + tests & linter (zero Tier-2 dependencies required)
pip install -e ".[dev]"

# Optional: interactive notebook execution support
pip install -e ".[dev,notebooks]"

# Optional: Tier-2 ingestion & upstream extraction dependencies
pip install -e ".[dev,tier2]"
```

---

## Tier 1: Reproducing All Manuscript Outputs from Zenodo

### Step 1: Obtain and Verify the Zenodo Archive

Download the companion archive (`10.5281/zenodo.24583418` — *pre-reserved archive DOI; unpublished draft pending final release*) or point to an unpacked local mirror:

```bash
export CAP_GAP_ZENODO_DIR=/path/to/zenodo_archive

# Optional download via Zenodo REST API once the record is published:
cap-gap fetch-zenodo --dest-dir "$CAP_GAP_ZENODO_DIR" --record-id <PUBLISHED_RECORD_ID>

# Verify every file against SHA256SUMS.txt (50 checksummed files):
cap-gap verify-zenodo --zenodo-dir "$CAP_GAP_ZENODO_DIR"
```

Expected Zenodo directory layout (`50` uploadable checksummed files + `SHA256SUMS.txt`, `28.1 MB` total):

```text
zenodo_archive/
├── SHA256SUMS.txt                 # 50-file SHA-256 manifest
├── LICENSE.txt                    # CC-BY-4.0 + upstream terms carve-out & attributions
├── zenodo.json                    # Zenodo deposit metadata
├── DATA_DICTIONARY.md             # Column-level schema reference
├── 01_event_catalogs/
│   ├── gadm41_adm1_index.parquet
│   ├── catalog_without_gs_all_185_2021_2025.{parquet,csv}
│   ├── catalog_without_gs_2021_2025.{parquet,csv}
│   ├── catalog_with_gs_all_185_2021_2025.{parquet,csv}
│   └── catalog_with_gs_2021_2025.{parquet,csv}
├── 02_admin1_exceedance_matrices/
│   ├── README.md
│   ├── cap_gid1_daily_2020_2025.parquet
│   ├── cap_gid0_daily_2020_2025.parquet
│   ├── cap_country_onboarding_2021_2025.csv
│   ├── cap_country_hazard_counts_2021_2025.csv
│   ├── cap_geocoding_audit_by_country.csv
│   ├── ffm_gid1_daily_2021_2025.parquet
│   ├── dhm_10d_gid1_multilead_2021_2025.npz
│   ├── dhm_0d_gid1_daily_2021_2025.npz
│   ├── geoglows_v2_retro_gid1_daily_2021_2025.npz
│   ├── geoglows_v2_oper_gid1_multilead_2024_2025.npz
│   └── glofas_v5_gid1_daily_2021_2025.npz
├── 03_summary_results/
│   ├── table1_summary.csv, table2_gauge_vs_catalogs.csv, stream_gauge_nse_summary.csv
│   ├── ed_table1_cohort_d_35.csv, ed_table3_sensitivity_matrix.csv, ed_table5_full_benchmark.csv
│   ├── ed_table6a_annual_onboarding.csv, ed_table6b_late_onboarders.csv, country_coverage_and_gains_150.csv
│   ├── pr_curves_summary.csv, cap_continuous_pr.csv, dhm_ffm_2d_grid.csv, ffm_gs_slices_pr.csv
│   ├── lead_window_sweep.csv, single_lead_decay.csv, sensitivity_audits.csv
│   ├── catalog_a_subnational_resolution_by_country.csv       # static upstream audit
│   ├── montandon_nonhuman_unit_audit.csv                     # static de-identified unit-level aggregate
│   └── montandon_and_demographics_audit_summary.json         # static upstream audit summary
└── 04_validation_deidentified/
    ├── README.md, SHA256SUMS.txt, tabulate_validation.py
    ├── validation_n400_deidentified.{csv,parquet}
    ├── validation_confusion_counts.csv
    └── validation_spatial_precision.csv
```

### Step 2: Run Full Tier-1 Scoring, Tables, and Figures

```bash
cap-gap reproduce-tier1 \
  --zenodo-dir "$CAP_GAP_ZENODO_DIR" \
  --results-dir ./out_results \
  --figures-dir ./out_figures \
  --verify-checksums
```

*(Equivalent script wrapper: `python3 scripts/04_score_benchmark.py --zenodo-dir ...`)*

#### Tier-1 Recomputed vs. Pass-Through & Static Files

| Output File (in `./out_results/`) | Tier-1 Provenance | Manuscript Target |
|---|---|---|
| `table1_summary.csv` | **Recomputed** | **Table 1** (Catalog A & Catalog B headline recall, precision, hybrid gap closure, exact McNemar *p*) |
| `table2_gauge_vs_catalogs.csv` | **Partial pass-through** *(in-situ gauge columns from Zenodo; DHM Catalog A/B columns recomputed)* | **Table 2** (Stream-gauge exceedance precision vs. Catalog A/B across return periods) |
| `stream_gauge_nse_summary.csv` | **Pass-through** *(raw GRDC hydrographs excluded from redistribution; recomputable in Tier 2)* | **Methods §5** (Unfiltered & area-validated multi-model stream-gauge NSE comparison) |
| `ed_table1_cohort_d_35.csv` | **Recomputed** | **Extended Data Table 1** (35 excluded non-CAP jurisdictions & event counts) |
| `ed_table3_sensitivity_matrix.csv` | **Recomputed** *(identical alias of `sensitivity_audits.csv`)* | **Extended Data Table 3** (Sensitivity matrix across spatial/temporal/population variants) |
| `ed_table4_validation_summary.csv` | **Recomputed** *(from `04_validation_deidentified/validation_n400_deidentified.csv`)* | **Extended Data Table 4** (De-identified `N=400` CAP validation summary) |
| `validation_confusion_counts.csv` | **Recomputed** *(verified against `04_validation_deidentified/`)* | Full confusion matrices across label versions, subsets, and predictors |
| `validation_spatial_precision.csv` | **Recomputed** *(verified against `04_validation_deidentified/`)* | Country-level pairing spatial precision across label versions and subsets |
| `ed_table5_full_benchmark.csv` | **Recomputed** | **Extended Data Table 5** (Full 0–7 day threshold sweep across all systems & benchmarks) |
| `ed_table6a_annual_onboarding.csv` | **Recomputed** | **Extended Data Table 6a** (Annual CAP onboarding trajectory, 2021–2025) |
| `ed_table6b_late_onboarders.csv` | **Recomputed** | **Extended Data Table 6b** (2022–2025 late-onboarder pre/post parity audit) |
| `country_coverage_and_gains_150.csv` | **Recomputed** | Per-jurisdiction coverage, population coverage, and hybrid gains (Figures 2 & 4, ED Figure 5) |
| `pr_curves_summary.csv` | **Recomputed** | All PR curve operating points (Figure 3, ED Figures 2–4) |
| `cap_continuous_pr.csv` | **Recomputed** | Continuous OASIS `score64` CAP PR sweep |
| `dhm_ffm_2d_grid.csv` | **Recomputed** | Joint DHM × FFM Pareto grid (`RP × τ`) |
| `lead_window_sweep.csv` | **Recomputed** | Cumulative lead-time sweep (`0–0d` through `0–15d`; DHM horizon caps at `min(N, 10)`) |
| `single_lead_decay.csv` | **Recomputed** | Single-lead skill decay (`L = 0..10d`) |
| `sensitivity_audits.csv` | **Recomputed** | Full sensitivity audit suite (Admin-0, 185 countries, post-onboarding, episode gap, etc.) |
| `ffm_gs_slices_pr.csv` | **Recomputed** | FFM and DHM+FFM stratified performance across Catalog A / Catalog B / Groundsource-only |
| `cap_country_hazard_counts_2021_2025.csv` | **Copied for standalone `build-figures`** | Multi-hazard CAP country counts for Figure 1b |
| `reproduction_summary.json` | **Recomputed** | Compact run-summary event and population checksums |

*(Three additional static audit files reside in Zenodo `03_summary_results/` — `catalog_a_subnational_resolution_by_country.csv`, `montandon_nonhuman_unit_audit.csv`, and `montandon_and_demographics_audit_summary.json` — and are verified by schema/privacy tests rather than recomputed from raw third-party disaster records under Tier 1.)*

#### Generated 300-DPI PNG Figures (in `./out_figures/`)

* `fig-1a.png`, `fig-1b.png` — Global CAP alerting capability categories map & multi-hazard country coverage bar chart
* `fig-2a.png`, `fig-2b.png`, `fig-2c.png`, `fig-2_composite.png` — 5-year CAP event coverage, 5-year CAP affected-population coverage, and post-onboarding event coverage choropleth maps (Catalog A) & 3-panel composite
* `fig-3.png` — Evaluation window schematic (`a`) and Catalog A / Catalog B precision–recall curves (`b–e`)
* `fig-4.png` — Per-country hybrid coverage gain choropleth (`CAP ∪ DHM ∪ FFM` over CAP, Catalog A & Catalog B)
* `ed-fig-1.png` — Manual `N = 400` validation audit precision–recall (`a`, driven dynamically from `ed_table4_validation_summary.csv`) and Admin-1 vs. Admin-0 spatial attribution sensitivity (`b`)
* `ed-fig-2.png` — CAP + global flood model hybrid precision–recall curves (`2×2` grid: 2021–2025 & 2024–2025 × Catalog A & Catalog B)
* `ed-fig-3.png` — Cumulative lead-window scaling (`0–0d` to `0–15d`) and single-lead forecast skill decay (`L = 0..10d`)
* `ed-fig-4.png` — Standalone FFM and DHM + FFM performance across Groundsource stratified slices (`2×3` grid)
* `ed-fig-5.png` — Catalog B 5-year and post-onboarding CAP coverage choropleth maps (`a`, `b`, with 2022–2025 late onboarders hatched)
* *(Static schematic `figures/Figure_5_early_warning_impact_chain.svg` is versioned directly in the repository.)*

### Step 3: Verify Parity Against Reference Summary Tables

```bash
cap-gap verify-parity \
  --actual-dir ./out_results \
  --reference-dir "$CAP_GAP_ZENODO_DIR" \
  --atol 1e-3
```

Pointing `--reference-dir` at the Zenodo root verifies all 16 tables in `03_summary_results/` (explicitly flagging `[PASS-THROUGH]` and `[PARTIAL PASS-THROUGH]` tables) plus `validation_confusion_counts.csv` and `validation_spatial_precision.csv` in `04_validation_deidentified/`.

You can also run the full pytest Zenodo integration suite:

```bash
CAP_GAP_ZENODO_DIR="$CAP_GAP_ZENODO_DIR" pytest -m zenodo -v
```

### Step 4: Standalone Figure Re-Rendering & Validation Audit

```bash
# Re-render all 13 figures from a reproduced results directory (no --zenodo-dir needed):
cap-gap build-figures \
  --results-dir ./out_results \
  --figures-dir ./out_figures

# Or re-render directly from Zenodo's precomputed 03_summary_results/ (requires --zenodo-dir):
cap-gap build-figures \
  --zenodo-dir "$CAP_GAP_ZENODO_DIR" \
  --results-dir "$CAP_GAP_ZENODO_DIR/03_summary_results" \
  --figures-dir ./out_figures

# Equivalent script wrapper:
python3 scripts/05_generate_manuscript_outputs.py \
  --results-dir ./out_results \
  --figures-dir ./out_figures

# Tabulate the de-identified N=400 validation CSV directly (also writes sibling confusion & spatial tables):
cap-gap audit-validation \
  --zenodo-dir "$CAP_GAP_ZENODO_DIR" \
  --output-csv ./out_results/ed_table4_validation_summary.csv
```

---

## Tier 2: Reconstructing Derived Matrices from Raw Upstream Sources

Tier-2 modules live under `src/cap_alerting_coverage/ingestion/` and `src/cap_alerting_coverage/pipeline/`, with numbered script wrappers in `scripts/`. All paths and credentials are configured via CLI flags or environment variables — zero credentials or machine-specific paths are hardcoded.

> **Upstream Access Note:** While public open-data archives (Groundsource v2 on Zenodo `24553440`, GADM v4.1, Copernicus GloFAS v5, GEOGLOWS v2 on AWS Open Data, IDMC GIDD, and authenticated IFRC Montandon STAC) can be fetched by external researchers under their respective provider terms, certain raw upstream inputs are **non-public**: historical Filtered Alert Hub subscription CAP archives, raw Google Distributed Hydrological Model (DHM) and Google Flash Flood Model (FFM) Zarr/raster archives, and raw GRDC stream-gauge hydrographs. For full end-to-end public verification, use Tier 1 above.

### Environment Variables

| Variable | Tier | Purpose |
|---|---|---|
| `CAP_GAP_ZENODO_DIR` | Tier 1 | Root of unpacked Zenodo derived benchmark archive |
| `CAP_GAP_ZENODO_RECORD_ID` | Tier 1 | Published Zenodo numeric record ID for `fetch-zenodo` (optional override once minted) |
| `CAP_GAP_RESULTS_DIR` | Tier 1/2 | Output directory for summary CSV tables |
| `CAP_GAP_FIGURES_DIR` | Tier 1/2 | Output directory for rendered 300-DPI PNG figures |
| `CAP_GAP_WORK_DIR` | Tier 2 | Working directory for intermediate Parquet/NPZ files |
| `CAP_GAP_DATA_INPUTS_DIR` | Tier 1/2 | Override directory containing `data_inputs/` reference files |
| `CAP_GAP_REPO_ROOT` | Tier 1/2 | Override repository root resolution |
| `CAP_GAP_SUP1_CSV` | Tier 1/2 | Explicit override path to `Supplementary_Data_1_*.csv` |
| `CAP_GAP_GEOJSON_PATH` | Tier 1/2 | Explicit override path to Natural Earth 1:110m GeoJSON |
| `CAP_GAP_NB_INLINE_IMAGES` | Notebook | Set to `"1"` to embed figure previews inline inside `notebooks/reproduce_manuscript_results.ipynb` |
| `CAP_GAP_DB_URL` | Tier 2 | SQLAlchemy database URL for CAP alert store (SQLite or PostgreSQL) |
| `CAP_GAP_SPANNER_PROJECT_ID`, `CAP_GAP_SPANNER_INSTANCE_ID`, `CAP_GAP_SPANNER_DATABASE_ID` | Tier 2 | Optional Cloud Spanner coordinates if not using `CAP_GAP_DB_URL` |
| `CAP_GAP_SENDER_MAP_JSON` | Tier 2 | Optional local JSON mapping private sender identifiers to ISO-3 codes |
| `MONTANDON_STAC_BEARER_TOKEN`, `MONTANDON_STAC_API_SEARCH` | Tier 2 | Bearer token & STAC search URL for IFRC Montandon API (never commit) |
| `IDMC_GIDD_CLIENT_ID` | Tier 2 | Client ID for IDMC GIDD API (`idmc_ingestor.py`) |
| `CAP_GAP_VERTEX_PROJECT`, `CAP_GAP_VERTEX_LOCATION`, `CAP_GAP_LLM_MODEL` | Tier 2 | Optional Vertex AI Gemini settings for `VertexGeminiBackend` (`translator.py`) |
| `CAP_GAP_DHM_ZARR_10D_URI`, `CAP_GAP_HYDRO_ATLAS_PARQUET`, `CAP_GAP_DHM_RP_PARQUET`, `CAP_GAP_DHM_GAUGE_IDS_NPY`, `CAP_GAP_HYDROBASINS_GPQ` | Tier 2 | Upstream paths/URIs for `03c_extract_dhm.py` |
| `CAP_GAP_FFM_ZARR_URI`, `CAP_GAP_FFM_SITES_PARQUET` | Tier 2 | Upstream paths/URIs for `03a_extract_ffm.py` |
| `CAP_GAP_GEOGLOWS_META_NPZ`, `CAP_GAP_GEOGLOWS_SHARDS_DIR` | Tier 2 | Upstream paths for `03b_extract_geoglows_v2.py` |
| `CAP_GAP_GLOFAS_V5_ZARR_URI` | Tier 2 | Upstream Zarr URI for `03d_extract_glofas_v5.py` |

### Tier-2 Pipeline Sequence

Run `python3 scripts/<script>.py --help` on any script to inspect all CLI flags:

1. **Database & Ingestion (`src/cap_alerting_coverage/ingestion/`):**
   * Initialize schema with `cap_alerting_coverage.ingestion.db.init_sql_schema(db_url)` (`schema.sql`, SQLite/PostgreSQL DDL).
   * Ingest CAP archives via `alerthub_ingestor.py` (NDJSON/XML streams) and IDMC events via `idmc_ingestor.py`.
   * Optionally translate non-English CAP alert fields via `translator.py` (plug in any `TranslationBackend`).
2. **Step 01 — Event Catalogs & GADM Lookup (`scripts/01_build_catalogs.py`):**
   * Builds `gadm41_adm1.parquet` from GADM v4.1 (`gadm_410-levels.gpkg`) and deduplicates IDMC + IFRC Montandon events into Catalog A (`catalog_without_gs_*`).
3. **Step 09 — Strictly Nested Groundsource Augmentation (`scripts/09_build_groundsource_catalog_b.py`):**
   * Maps Groundsource v2 polygons to GADM Admin-1 units (`point_on_surface` intersects or `<= 0.15°`), clusters contiguous/overlapping episodes per `gid_1` (`<= 7d`), and merges strictly onto Catalog A (`±7d`, same `gid_1`) to produce Catalog B (`catalog_with_gs_*`).
4. **Step 02 — CAP Admin-1 & Admin-0 Daily Matrices (`scripts/02_extract_cap_alerts.py`):**
   * Applies multilingual inclusion/exclusion keyword filters (`keywords.py`), OASIS `score64` scoring, `>= 5%` GADM polygon overlap, `0.03°` centroid rejection, and `< 80%` whole-country rejection to emit `cap_gid1_daily_2020_2025.parquet`, `cap_gid0_daily_2020_2025.parquet`, onboarding dates, and multi-hazard counts.
5. **Steps 03a–03d — Global Flood Model Daily Exceedance Cubes:**
   * `scripts/03a_extract_ffm.py`: Aggregates Google FFM daily probabilities to `ffm_gid1_daily_2021_2025.parquet`.
   * `scripts/03b_extract_geoglows_v2.py`: Aggregates GEOGLOWS v2 retrospective (`0d`) and operational (`0–14d`, 4 products) Zarr stores to discrete `uint8` return-period tier NPZ cubes.
   * `scripts/03c_extract_dhm.py`: Aggregates Google Distributed Hydrological Model (DHM) operational Zarr chunks (`0–10d`) to `dhm_10d_gid1_multilead_2021_2025.npz` (using pure-Python `blosc_decode.py` when native `blosc` is unavailable).
   * `scripts/03d_extract_glofas_v5.py`: Fits 1981–2025 Bulletin 17C Log-Pearson Type III return-period thresholds on GloFAS v5 annual maxima (`mean_Q >= 10 m³/s`) and aggregates daily exceedances to `glofas_v5_gid1_daily_2021_2025.npz`.
6. **Steps 04 & 05 — Scoring, Tables & Figures:**
   * `python3 scripts/04_score_benchmark.py` (wrapper for `cap-gap reproduce-tier1`: runs full scoring, manuscript tables, and figures).
   * `python3 scripts/05_generate_manuscript_outputs.py` (wrapper for `cap-gap build-figures`: re-renders all 13 PNG figures from precomputed summary CSVs).
