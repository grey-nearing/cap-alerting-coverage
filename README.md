# Global Forecasts Can Close Most of the Flood Warning Gap

[![CI](https://github.com/grey-nearing/cap-alerting-coverage/actions/workflows/ci.yml/badge.svg)](https://github.com/grey-nearing/cap-alerting-coverage/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Reproducible analysis code, benchmark verification suite, and figure/table generation pipeline accompanying the manuscript:

> **Global forecasts can close most of the flood warning gap**
>
> Justin Ginnetti, Juliet Rothenberg, Shmulik Fronman, Michelle Ng, Oleg Zlydenko, Martin Gauch, Daniel Klotz, Patrick Ebel, Asher Metzger, Hadas Fester, Avishag Shapira, Dem Gerolemou, Assaf Shmuel, Ido Zemach, Rotem Mayo, Roy Nadler, Yonatan Nakar, Assaf Mauda, Matheus Vervloet, Karan Ruparell, Frederik Kratzert, Ben Feinstein, Deborah Cohen, Aviya Goldstein, Gila Loike, Dai Yamazaki, Lisie Lillianfeld, Emily Reinstein, Adam Boulanger, and Grey Nearing.

Companion Zenodo derived benchmark archive: `10.5281/zenodo.24583418` *(pre-reserved archive DOI; unpublished draft pending final release)*.

---

## Overview

This repository quantifies the global subnational coverage gap in official **Common Alerting Protocol (CAP)** flood warnings across **150 evaluated jurisdictions** (selected from an audit of **185 jurisdictions**, excluding 35 jurisdictions with mature non-CAP automated national warning platforms) over **2021–2025**, and benchmarks the extent to which operational global flood forecasting models can close that gap:

* **Official CAP alerts** (Filtered Alert Hub and IFRC Alert Hub archives, geocoded to GADM v4.1 Admin-1 units via `≥ 5%` overlap and `< 80%` whole-country rejection).
* **Global riverine flood models**: Google Distributed Hydrological Model (DHM, 0–7 day operational & 0-day), Copernicus GloFAS v5 (0-day reanalysis), and GEOGLOWS v2 (0-day retrospective & 0–7 day operational ensemble/HRES, April 2024–December 2025).
* **Global flash flood model**: Google Flash Flood Model (FFM, 0–24 h operational probabilities).
* **Two nested event benchmarks**:
  * **Catalog A (Impact-verified disasters, `N = 5,894` events in 150 countries; `8,908` in 185 audited jurisdictions, `8,911` across all territories):** IDMC GIDD + IFRC Montandon (EM-DAT, GDACS, IFRC GO) deduplicated per GADM Admin-1 unit and 7-day window.
  * **Catalog B (Expanded media-verified benchmark, `N = 27,127` events in 150 countries; `38,590` in 185 audited jurisdictions, `38,975` across all territories):** Catalog A augmented with strictly nested Groundsource v2 flood episodes (`±7d` merge within exact GADM Admin-1 units).

### Key Findings (150 Evaluated Jurisdictions, 2021–2025; CAP & River `[−7 d, end]`, 0–24 h FFM `[−1 d, end]`)

| System | Catalog A Event Recall (`N=5,894`) | Catalog A Pop. Recall (`52.66M`) | Catalog A Gap Closed | Catalog B Event Recall (`N=27,127`) | Catalog B Gap Closed |
|---|---:|---:|---:|---:|---:|
| **Official CAP (Broad Flood-and-Storm)** | `38.6%` (`2,277`) | `59.4%` | — | `45.6%` (`12,371`) | — |
| **Official CAP (Explicit Flood-Coded)** | `7.2%` (`423`) | `28.9%` | — | `10.7%` (`2,895`) | — |
| **CAP ∪ Google DHM (0–7d, Q2yr)** | `60.8%` (`3,584`) | `84.6%` | `36.1%` | `62.4%` (`16,927`) | `30.9%` |
| **CAP ∪ Google FFM (0–24h, τ=0.50)** | `67.3%` (`3,964`) | `80.1%` | `46.6%` | `77.5%` (`21,034`) | `58.7%` |
| **CAP ∪ DHM (0–7d) ∪ FFM (0–24h) [Hybrid Full]** | **`76.5%`** (`4,506`) | **`89.4%`** | **`61.6%`** | **`83.2%`** (`22,576`) | **`69.2%`** |

### Two-Tier Reproducibility Architecture

To respect upstream licensing and third-party redistribution restrictions while guaranteeing full scientific transparency, this repository implements a clean two-tier reproducibility workflow:

1. **Tier 1 — Full Manuscript Reproduction from Zenodo (~3.5 min, CPU-only, ~28 MB archive):**
   Download the derived, non-raw Zenodo archive (event catalogs, Admin-1 daily exceedance matrices, discrete `uint8` return-period cubes, de-identified `N = 400` validation labels, and reference summary CSVs) and deterministically recompute every benchmark metric, exact McNemar test, sensitivity audit, manuscript table, and all 11 single-composite publication figures in vector PDF, vector SVG, and 300-DPI PNG (Figures 1–5, Extended Data Figures 1–5, Supplementary Figure 1). *(Because raw GRDC/USGS/Caravan gauge hydrographs cannot be redistributed, Tier-1 passes through anonymized `stream_gauge_nse_summary.csv`, `stream_gauge_nse_ecdf_quantiles.csv`, `stream_gauge_precision_iqr.csv`, and the gauge-side columns of `table2_gauge_vs_catalogs.csv` while recomputing all catalog-side columns dynamically; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).)*
2. **Tier 2 — Full Upstream Extraction & Ingestion (Reference Pipeline):**
   Portable, parameterized modules under `src/cap_alerting_coverage/ingestion/` and `src/cap_alerting_coverage/pipeline/` document and execute extraction from raw upstream sources (CAP XML/JSON feeds, IDMC GIDD API, IFRC Montandon STAC, Groundsource v2, GADM v4.1, DHM, FFM, GloFAS v5, and GEOGLOWS v2) with zero hardcoded credentials or environment-specific paths.

---

## Quickstart (Tier 1)

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Verify Installation Offline (Synthetic End-to-End CI Suite)

```bash
pytest -m "not zenodo"
```

This runs all unit tests (multilingual keyword filters, CAP polygon geocoding, episode clustering, LP3 return-period fitting, exact McNemar & bootstrap utilities, clean-room privacy guardrails) plus a complete synthetic Tier-1 end-to-end reproduction.

### 3. Reproduce Manuscript Tables & Figures from the Zenodo Archive

Point `CAP_GAP_ZENODO_DIR` (or `--zenodo-dir`) to the unpacked Zenodo archive directory:

```bash
# Optional: fetch from Zenodo once the record is published (pass the minted record ID)
cap-gap fetch-zenodo --dest-dir ./zenodo_archive --record-id <PUBLISHED_RECORD_ID>

# Verify SHA-256 checksums of all staged Zenodo files
cap-gap verify-zenodo --zenodo-dir ./zenodo_archive

# Recompute all scoring matrices, tables, and 300-DPI figures
cap-gap reproduce-tier1 \
  --zenodo-dir ./zenodo_archive \
  --results-dir ./out_results \
  --figures-dir ./out_figures \
  --verify-checksums

# Verify exact numerical parity against reference 03_summary_results/ & 04_validation_deidentified/
cap-gap verify-parity \
  --actual-dir ./out_results \
  --reference-dir ./zenodo_archive
```

You can also run the interactive walkthrough in [notebooks/reproduce_manuscript_results.ipynb](notebooks/reproduce_manuscript_results.ipynb) (install `pip install -e ".[notebooks]"`) or re-render figures directly from precomputed CSVs with `cap-gap build-figures`.

### Quickstart (Tier 2 — Optional Upstream Extraction)

```bash
pip install -e ".[dev,tier2]"
python3 scripts/01_build_catalogs.py --help
```

Tier-2 ingestion and extraction require external upstream datasets and credentials (some raw upstream archives, such as historical Filtered Alert Hub feeds and raw Google DHM/FFM model Zarr stores, are non-public; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md)).

---

## Repository Structure

```text
cap-alerting-coverage/
├── LICENSE                         # Apache-2.0 software license
├── CITATION.cff                    # Machine-readable citation metadata (30 authors)
├── pyproject.toml                  # Package metadata, CLI entrypoints, ruff & pytest config
├── README.md                       # Human overview & quickstart
├── AGENTS.md                       # Machine-readable guide for autonomous coding agents
├── llms.txt                        # Concise index for LLM/agent tooling
├── data_inputs/                    # Verified static reference inputs + data_inputs/LICENSE
├── figures/                        # Static schematic (Figure 5 early warning impact chain SVG)
├── docs/
│   ├── REPRODUCIBILITY.md          # Detailed Tier-1 & Tier-2 reproduction runbook
│   ├── DATA_GOVERNANCE.md          # Upstream provenance, licenses, & non-redistribution boundaries
│   └── METHODS_ALIGNMENT.md        # Transparent alignment notes between code & manuscript text
├── notebooks/
│   └── reproduce_manuscript_results.ipynb  # Clean Tier-1 reproduction notebook
├── scripts/                        # Numbered CLI wrappers mirroring the full pipeline DAG (01–09)
├── src/cap_alerting_coverage/
│   ├── config.py                   # Constants, cohorts, thresholds, and portable path resolution
│   ├── scoring.py                  # Exact benchmark scoring, PR curves, Pareto front, lead sweeps, audits
│   ├── manuscript_outputs.py       # Manuscript & Extended Data tables, McNemar & country-clustered inference, figures
│   ├── figure_style.py             # Shared Nature typography, colorblind-safe palette, Robinson CRS & PDF/SVG/PNG export
│   ├── figure_fleet.py             # All 11 single-composite publication figures (Figs. 1–5, ED Figs. 1–5, Supp. Fig. 1)
│   ├── gauge_verification.py       # Stream-gauge NSE/ECDF/IQR summaries & Extended Data Table 4 compilation
│   ├── validation_audit.py         # Deterministic de-identified N=400 validation tabulation
│   ├── keywords.py                 # Multilingual flood/storm regex classifiers & OASIS score64
│   ├── clustering.py               # Episode collapsing, population waterfall, GID_1 clustering
│   ├── stats_utils.py              # Exact McNemar, Fisher exact, country Wilcoxon, DEFF, clustered bootstrap CIs, LP3
│   ├── schemas.py                  # Strict I/O schema & non-redistribution privacy validators
│   ├── zenodo_fetch.py             # Zenodo archive downloader, safe ZIP extractor & SHA-256 verifier
│   ├── cli.py                      # Unified `cap-gap` / `cap-alerting-coverage` CLI
│   ├── ingestion/                  # Tier-2 portable SQL DDL, Alert Hub/IDMC ingestors, translator
│   └── pipeline/                   # Tier-2 upstream extraction modules (01, 02, 03a–d, 09, blosc)
└── tests/                          # Unit tests, synthetic Tier-1 integration, Zenodo parity, privacy guardrails
```

---

## Documentation Index

* **[Reproducibility Guide](docs/REPRODUCIBILITY.md):** Step-by-step instructions for Tier-1 Zenodo reproduction, parity checking, and Tier-2 upstream extraction.
* **[Data Governance & Licensing](docs/DATA_GOVERNANCE.md):** Provenance, terms of use, required upstream attributions, and explicit boundaries on what is and is not redistributed.
* **[Methods-to-Code Alignment](docs/METHODS_ALIGNMENT.md):** Transparent engineering notes mapping exact algorithmic implementations to manuscript sections.
* **[Agent Guide](AGENTS.md) & [llms.txt](llms.txt):** Dedicated machine-readable specifications for coding agents (pipeline DAG, exact column schemas, test commands, and guardrails).

---

## Licensing & Citation

* **Code (`src/`, `scripts/`, `tests/`, `notebooks/`):** Licensed under the [Apache License 2.0](LICENSE).
* **Author-Curated Reference Inputs & Derived Benchmark Summaries (`data_inputs/`, Zenodo derived outputs):** Released under **CC-BY-4.0** (see [data_inputs/LICENSE](data_inputs/LICENSE)); Natural Earth 1:110m boundaries are **Public Domain**. Upstream source datasets (Copernicus GloFAS v5, GEOGLOWS v2, IDMC GIDD, IFRC Montandon, Groundsource v2, GADM v4.1, and CAP alerting feeds) remain governed by their respective upstream terms. Raw upstream CAP alert text, third-party disaster records, Groundsource geometries, GADM geometries, and raw river discharge streams are **not** redistributed; see [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md).

If you use this software or benchmark archive, please cite the manuscript and software record via [CITATION.cff](CITATION.cff).
