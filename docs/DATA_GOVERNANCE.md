# Data Governance, Provenance & Licensing

This repository and its companion Zenodo benchmark archive adhere to strict upstream data-governance, licensing, and privacy boundaries.

---

## Licensing Summary

| Artifact Category | Location | License |
|---|---|---|
| Software package, CLI, tests, CI, notebook | This GitHub repository (`src/`, `scripts/`, `tests/`, `notebooks/`) | **Apache-2.0** ([LICENSE](../LICENSE)) |
| Author-curated static reference tables & keyword filters | `data_inputs/Supplementary_Data_1_*.csv`, `unified_multilingual_keyword_filters.csv`, `annex_*.txt` | **CC-BY-4.0** ([data_inputs/LICENSE](../data_inputs/LICENSE)) |
| Country map boundaries (`1:110m`) | `data_inputs/ne_110m_admin_0_countries.geojson` (Natural Earth) | **Public Domain** |
| Impact chain schematic | `figures/Figure_5_early_warning_impact_chain.svg` | **CC-BY-4.0** *(adapted from Lindell & Perry 2012 PADM)* |
| Derived event catalogs, Admin-1 exceedance matrices, `uint8` tier cubes, de-identified `N=400` audit labels, `Supplementary_Data_1`, and summary CSVs | Companion Zenodo archive (`10.5281/zenodo.24583418` — *pre-reserved archive DOI; unpublished draft pending final release*) | **CC-BY-4.0** *(subject to upstream terms carve-out below)* |

---

## Upstream Datasets: Provenance & Redistribution Boundary

| Upstream Dataset | Provider / Source | Upstream License / Access Terms | Redistributed Here? | Representation in Zenodo Archive |
|---|---|---|---|---|
| **Filtered Alert Hub (FAH) CAP Archive** | Alert-Hub.Org CIC / Filtered Alert Hub (aggregates official CAP feeds registered in the WMO Register of Alerting Authorities) | Paid subscription / institutional data-access agreement; heterogeneous national CAP terms | **No** (raw CAP XML, headlines, descriptions, instructions, URLs, senders, polygons are **never** redistributed) | Derived daily Admin-1 & Admin-0 severity integers (`0..4`), OASIS `score64` integers (`0..64`), and aggregate alert counts only |
| **IFRC Alert Hub CAP Archive** | International Federation of Red Cross and Red Crescent Societies (aggregates official CAP feeds) | IFRC non-commercial humanitarian data terms; heterogeneous national CAP terms | **No** | Merged into derived Admin-1/Admin-0 severity & count matrices only |
| **IDMC Global Internal Displacement Database (GIDD)** | Internal Displacement Monitoring Centre (`https://www.internal-displacement.org/database/displacement-data/`) | **CC BY-NC-SA 3.0 IGO** (attribution, non-commercial, share-alike) | **No** (raw records not redistributed) | Derived event start/end dates, GADM `gid_1`, source presence boolean `has_idmc`, and harmonized `waterfall_pop` |
| **IFRC Montandon Global Crisis Data Bank** | International Federation of Red Cross and Red Crescent Societies (aggregates EM-DAT, GDACS, IFRC GO) | Authenticated STAC API; EM-DAT non-commercial redistribution restrictions | **No** (raw records, descriptions, EM-DAT IDs, and API tokens are **never** redistributed) | Derived event start/end dates, GADM `gid_1`, source presence boolean `has_montandon`, harmonized `waterfall_pop`, and unit-level aggregate count audit |
| **Groundsource v2** | Mayo et al., Zenodo (`doi:10.5281/zenodo.24553440`) | **CC-BY-4.0** (published separately on Zenodo) | **No** (raw polygons & news metadata not duplicated; fetch directly from Zenodo `24553440`) | Derived Admin-1 clustered episode dates and `has_groundsource` flag in Catalog B |
| **GADM v4.1 Administrative Boundaries** | GADM (`https://gadm.org/download_world.html`) | Free for academic / non-commercial use; **no redistribution of raw geometries** | **No** (raw `.gpkg`/polygons are **never** redistributed) | Non-geometric `(gid_1, country_iso3)` lookup index (`gadm41_adm1_index.parquet`) only |
| **Natural Earth 1:110m Admin-0 Countries** | Natural Earth (`https://www.naturalearthdata.com/`) | **Public Domain** | **Yes** (`data_inputs/ne_110m_admin_0_countries.geojson`, for choropleth plotting) | Included in repo `data_inputs/` |
| **Google Distributed Hydrological Model (DHM)** | Google Flood Forecasting / Flood Hub (`https://sites.research.google/floods/`) | Google Flood Forecasting model outputs (raw historical Zarr archives are not publicly licensed) | **No** (raw multi-terabyte discharge time series not redistributed) | Derived Admin-1 daily discrete `uint8` return-period exceedance tiers (`0..8`, 0–10d lead times) |
| **Google Flash Flood Model (FFM)** | Google Flood Forecasting | Google Flood Forecasting model outputs (raw gridded probability archives are not publicly licensed) | **No** (raw gridded rasters not redistributed) | Derived Admin-1 daily maximum flash-flood probability (`max_prob` in `[0, 1]`) |
| **Copernicus GloFAS v5** | ECMWF / Copernicus Emergency Management Service (`https://ewds.climate.copernicus.eu/`) | **Copernicus CEMS Open License** (attribution required) | **No** (raw gridded discharge NetCDFs not redistributed) | Derived Admin-1 daily discrete `uint8` return-period exceedance tiers (`0..8`, 0d) |
| **GEOGLOWS ECMWF Streamflow Service v2** | GEOGLOWS / BYU / ECMWF (AWS Open Data S3) | **CC-BY-4.0** | **No** (raw reach-level Zarr discharge streams not redistributed) | Derived Admin-1 daily discrete `uint8` return-period exceedance tiers (`0..8`, retro 0d & oper 0–14d) |
| **Stream-Gauge Observations (GRDC / Caravan / USGS)** | GRDC, Caravan (`doi:10.5281/zenodo.7540792`), USGS NWIS | Heterogeneous (GRDC prohibits raw discharge redistribution) | **No** (raw gauge hydrographs are **never** redistributed) | Aggregated return-period precision & multi-model NSE summary tables (`table2_gauge_vs_catalogs.csv`, `stream_gauge_nse_summary.csv`) |

---

## Upstream Terms Carve-Out & Required Attributions

The **CC-BY-4.0** license on `data_inputs/` and the companion Zenodo archive applies strictly to author-curated metadata, keyword filter lists, and derived aggregate benchmark matrices created by the manuscript authors. All upstream source datasets remain governed by their original provider licenses and terms of use (including non-commercial provisions where applicable, such as IDMC GIDD, EM-DAT via IFRC Montandon, IFRC Alert Hub, and GADM v4.1). Reuse of derived outputs that incorporate upstream information must retain the following attributions:

* **Copernicus GloFAS v5:** *Contains modified Copernicus Emergency Management Service information (2026).* Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
* **GEOGLOWS ECMWF Streamflow Service v2:** GEOGLOWS / Brigham Young University / ECMWF, licensed under CC-BY-4.0.
* **IDMC Global Internal Displacement Database (GIDD):** Internal Displacement Monitoring Centre (IDMC), licensed under CC BY-NC-SA 3.0 IGO (`https://www.internal-displacement.org/database/displacement-data/`).
* **IFRC Montandon:** *IFRC (2026). Montandon – Global Crisis Data Bank* (incorporating disaster records from EM-DAT, GDACS, and IFRC GO).
* **Groundsource v2:** Mayo et al., Zenodo (`doi:10.5281/zenodo.24553440`), licensed under CC-BY-4.0.
* **GADM v4.1:** Database of Global Administrative Areas (`https://gadm.org/`), academic / non-commercial use; no raw geometries are redistributed here.
* **Natural Earth 1:110m:** *Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com* (Public Domain).
* **Figure 5 Schematic (`figures/Figure_5_early_warning_impact_chain.svg`):** Adapted from the Protective Action Decision Model (PADM) of Lindell, M. K. & Perry, R. W. (2012), *Risk Analysis* 32(4), 616–632.

---

## Privacy & Non-Redistribution Guardrails

This repository enforces automated data-governance checks both at runtime and in GitHub Actions CI:

1. **Schema-Level Column Blocking (`src/cap_alerting_coverage/schemas.py`):**
   * Every derived public table validator explicitly rejects raw text, geometry, URL, email, or raw streamflow columns (`raw_xml`, `headline`, `description`, `instruction`, `translated_*`, `source_url`, `url`, `sender`, `polygon`, `geometry`, `article_text`, `raw_discharge`, etc.), including strictly aggregating `montandon_nonhuman_unit_audit.csv` to unit-level counts (`23` unit rows across `223` excluded records; zero raw `description` or `item_id`).
   * No contact email addresses or SHA-256 email-hash lookup tables are embedded in public source code (`country_resolver.py` resolves via ISO-3166 numeric/alpha codes, public WMO OIDs, and public institutional domains, with optional local mapping via `CAP_GAP_SENDER_MAP_JSON`).
   * GADM v4.1's unnamed Ukraine polygon (`gid_1 == "?"`, appearing in `22` Catalog B events, `2` CAP rows, and `1,843` FFM rows; `0` Catalog A events) is preserved verbatim as a non-geometric code for exact 1-to-1 alignment.
2. **De-identified `N = 400` Validation Audit (`04_validation_deidentified/validation_n400_deidentified.csv`):**
   * Contains strictly 400 rows × 30 boolean/categorical/integer columns (review-round verdicts, area stratum, lead-window bin, pseudonymous `alert_group_id`, severity enum, and pipeline filter booleans).
   * Zero CAP headlines, descriptions, instructions, area descriptions, translations, senders, UUIDs, exact dates, country codes, or reviewer notes are included.
   * Runtime validators reject any non-numeric/non-boolean cell exceeding 64 characters or containing URLs, `@`, or XML tags.
3. **CI Clean-Room Guardrail Suite (`tests/test_clean_room_privacy.py`):**
   * Automatically fails CI if `LICENSE` drifts from canonical Apache-2.0, or if any raw binary dataset extension (`.parquet`, `.gpkg`, `.npz`, `.npy`, `.grib*`, `.nc`, `.sqlite`, `.ndjson.gz`), hardcoded JWT/token, local host home path, or embedded hash map enters the tracked Git tree.
4. **Credential Isolation:**
   * All Tier-2 credentials and endpoints (`MONTANDON_STAC_BEARER_TOKEN`, `IDMC_GIDD_CLIENT_ID`, `CAP_GAP_DB_URL`, `CAP_GAP_VERTEX_PROJECT`, `CAP_GAP_SPANNER_*`, etc.) are read exclusively from environment variables and never persisted to disk or logs.
