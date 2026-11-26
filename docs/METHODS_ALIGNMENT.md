# Methods-to-Code Alignment Notes

This document transparently maps the algorithmic implementation in this repository to the Methods and Extended Data sections of *"Global flood models can close nearly two-thirds of the warning gap"*, documenting exact implementation details and methodological sensitivities.

---

## 1. Jurisdiction Cohorts, Study Window & Figure 1b Definition

* **Audited Universe (`N = 185`):** Loaded dynamically at runtime from `data_inputs/Supplementary_Data_1_Global_Alerting_Authorities_Audit_185_Jurisdictions.csv` (see `scoring.py` and `manuscript_outputs.py`; legacy sensitivity cohort `LEGACY_EXCLUDED_19` and `ISO_UNIFY` live in `src/cap_alerting_coverage/config.py`).
  * **Cohort A (Category 1 — Active Flood CAP):** 70 jurisdictions.
  * **Cohort B (Category 2 — Non-Flood CAP Only):** 45 jurisdictions.
  * **Cohort C (Category 3 — Zero 2021–2025 CAP Alerts):** 35 jurisdictions.
  * **Evaluated Universe (`N = 150`):** Cohorts A + B + C (`70 + 45 + 35 = 150`).
  * **Cohort D (Excluded Mature Non-CAP Automated Systems):** 35 jurisdictions (listed in Extended Data Table 1).
  * *Reference table note (`Supplementary_Data_1`):* `data_inputs/Supplementary_Data_1_Global_Alerting_Authorities_Audit_185_Jurisdictions.csv` is preserved verbatim as the published reference audit table. Its static per-country count columns reflect an earlier catalog snapshot (`8,272` disasters / `3,534` CAP-alerted = `42.7%` across the 150 evaluated jurisdictions), whereas the pipeline reads **only** jurisdiction identifiers, cohort/category labels, and alerting-authority metadata (`ISO3`, `Country_or_Territory`, `Cohort_and_Category`, `WMO_Registered_Alerting_Authorities`, `Official_Public_Warning_Portal_and_Non_CAP_Channels`, `WMO_and_AlertHub_CAP_Feed_Status_and_URL`) and dynamically computes all per-country event and coverage counts (`5,894` / `2,277` = `38.6%` in Catalog A) into `country_coverage_and_gains_150.csv`.
  * *Figure 1b hazard-bar definition:* In `manuscript_outputs.py`, the `"Floods (Category 1)"` bar plots the curated institutional Category 1 cohort count (`70` jurisdictions from `Supplementary_Data_1`) rather than raw keyword-count thresholds from `cap_country_hazard_counts_2021_2025.csv` (`n_flood_rest > 0` holds for `57` of 150 jurisdictions; `n_flood_perm > 0` holds for `113` of 150). The other six hazard bars count jurisdictions in the 150-country evaluation set with `> 0` CAP alerts in `cap_country_hazard_counts_2021_2025.csv` (`Storms = 102`, `Heat = 64`, `Cold = 60`, `Wildfire = 18`, `Earthquake = 9`, `Volcano = 5`).
* **Study Period:** `2021-01-01` to `2025-12-31` inclusive (`1,826` days). Daily matrices span a 17-day leading buffer starting `2020-12-15` (`D_TOTAL = 1,843` daily steps) to eliminate boundary truncation for pre-onset lead windows up to 15 days.
* **GEOGLOWS v2 Operational Window:** `2024-04-01` to `2025-12-31` (`640` issue dates, `0–14` day lead times), scored head-to-head against DHM and CAP on the shared valid window `2024-04-16` to `2025-12-31`.
* **ISO-3 Unification:** Territorial and legacy codes are harmonized consistently across GADM, CAP, IDMC, and Montandon via `ISO_UNIFY` in `config.py` (`XXK`/`XKO`/`KOS` → `XKX`, `AZO` → `PRT`, `PRI` → `USA`).

---

## 2. Event Catalog Construction & Population Waterfall

* **Catalog A (IDMC + IFRC Montandon, `N = 5,894` in 150 countries; `8,908` in 185 audited jurisdictions, `8,911` total including 3 non-audited territory events):**
  * Implemented in `src/cap_alerting_coverage/pipeline/build_catalogs.py` and `src/cap_alerting_coverage/clustering.py`.
  * Records are spatially linked to GADM v4.1 Admin-1 (`gid_1`) units and deduplicated per `gid_1` whenever event intervals overlap or fall within `<= 7` days (symmetric chaining per `gid_1`; multi-province records split population equally across matched units).
  * **Population Harmonization (`compute_waterfall_population`):** Three explicit population rules are implemented in `clustering.py`:
    1. **`pop_rule="priority_fallback"` (default, baseline benchmark):** Uses `num_affected` if `> 0`, else `displacements` if `> 0`, else `num_killed`. Produces an exact Catalog A 150-country total of `52,661,289.42` people (CAP population coverage `59.4%`, Hybrid Full population coverage `90.0%`, population gap closed `75.4%`).
    2. **`pop_rule="max"` (pure max sensitivity):** Uses `max(num_affected, displacements)` with no `num_killed` fallback. Relative to `priority_fallback`, this alters `467` of `5,894` events (`90` merged events increase by `+1,618,113`; `377` killed-only Montandon events lose `-1,519`), yielding `54,277,882.75` (`+1,616,593`, `+3.07%`), shifting CAP population coverage from `59.4%` to `58.6%` (`-0.76 pp`), Hybrid Full coverage from `90.0%` to `90.2%` (`+0.22 pp`), and population gap closed from `75.4%` to `76.4%`.
    3. **`pop_rule="max_with_killed_fallback"`:** Uses `max(num_affected, displacements)`, falling back to `num_killed` when both are zero, yielding `54,279,402.09` (`+1,618,113`, `+3.07%`).
    * Every event-count metric (`38.6%` CAP recall, `78.7%` Hybrid Full recall, `65.3%` / `72.9%` event gap closed) is invariant to `pop_rule`.
* **Catalog B (Catalog A + Groundsource v2, `N = 27,127` in 150 countries; `38,590` in 185 audited jurisdictions, `38,975` total across all territories):**
  * Implemented in `src/cap_alerting_coverage/pipeline/build_groundsource_catalog_b.py` and `clustering.py`.
  * **Spatial & Temporal Clustering:** Groundsource v2 flood polygons are mapped to GADM v4.1 `gid_1` units via `point_on_surface()` intersection (with nearest-unit fallback within `0.15°`), clustered per `gid_1` whenever consecutive events occur within `<= 7` days, and merged onto Catalog A when an identical `gid_1` matches within `±7` days of `[start_date, end_date]` (`3,646` Catalog A parents corroborated; `21,233` standalone Groundsource-only episodes in the 150 countries with `waterfall_pop = 0`, guaranteeing strict nesting `Catalog A ⊂ Catalog B`). Administrative-unit clustering (`gid_1` + `<= 7d`) is used directly rather than unconstrained 150 km ST-DBSCAN.
  * **GADM `gid_1 == "?"` Edge Case:** GADM v4.1 assigns the literal string `"?"` to one unnamed polygon in Ukraine (`country_iso3 == "UKR"`), appearing in `22` Groundsource-only Catalog B events, `2` CAP daily rows, and `1,843` FFM rows (`0` Catalog A events); it is preserved verbatim for exact 1-to-1 GADM index alignment.

---

## 3. CAP Alert Filtering, Translation & Subnational Geocoding

* **Multilingual Keyword Classification (`src/cap_alerting_coverage/keywords.py`):**
  * Combines curated exclusion regexes (`EXCL_RE`), broad flood-and-storm inclusion regexes (`PERM_RE`), and explicit flood-coded regexes (`REST_RE`), evaluated over raw and translated alert `event`, `headline`, and `description` fields.
  * `EXCL_RE` runs strictly on the `event` field and takes precedence over inclusion terms (`keywords.py` applies its compiled regexes directly rather than parsing the reference CSV/TXT tables in `data_inputs/` at runtime).
  * Beyond the 12 core languages consolidated in Extended Data Table 2, the implemented regexes also cover additional multilingual terms encountered in the global archive (including German, Italian, Persian, Turkish, Thai, Vietnamese, Tagalog, and Hindi).
  * Tier-2 translation (`src/cap_alerting_coverage/ingestion/translator.py`) uses a provider-agnostic LLM translation interface (`TranslationBackend`, with an optional `VertexGeminiBackend` configured via environment variables) over non-English alerts with strict `<cap_alert>` delimiter isolation and JSON schema validation.
* **Subnational Polygon Geocoding & Centroid Rejection (`src/cap_alerting_coverage/pipeline/extract_cap_alerts.py`):**
  * CAP polygons and circles are mapped to GADM v4.1 Admin-1 units using a pure geometric overlap rule: an Admin-1 unit is attributed if the intersection area satisfies `overlap >= min(0.05 * poly_area, 0.05 * adm1_area)`, while whole-country blanket alerts covering `>= 80%` of the parent nation's total Admin-1 area are excluded from subnational Admin-1 attribution (and retained in the Admin-0 sensitivity audit). Gridded population weighting (WorldPop / GPW v4) is not used in the spatial attribution step.
  * *Point-only centroid rejection (`0.03°` tolerance) & Tier-1 vs. Tier-2 Poland sensitivity:* Point-only CAP alerts within `0.03°` (~3.3 km; `abs(lon - cx) < 0.03 and abs(lat - cy) < 0.03`) of a country centroid are rejected as whole-country placeholders (`extract_cap_alerts.py` and `build_catalogs.py`). The published Zenodo Tier-1 matrix (`cap_gid1_daily_2020_2025.parquet`) was built with country centroids computed by dissolving GADM Admin-1 polygons (`364,679` CAP `gid_1`-days, `65,874` CAP episodes, `45.604%` Catalog B CAP recall, `85.236%` Catalog B Hybrid recall), under which a recurring placeholder coordinate in Poland (`51.91°N, 19.14°E`) fell outside the `0.03°` Admin-1-dissolve box. By contrast, public Tier-2 (`build_catalogs.py`) computes country centroids directly from GADM Admin-0 (`ADM_0` dissolve), which rejects that placeholder coordinate and changes **only Poland (`POL`)**: `365,744` CAP `gid_1`-days, `66,151` CAP episodes (`+277`; Catalog A precision `3.809%` → `3.793%`), Catalog B CAP recall `45.604%` → `45.785%` (`+49` POL events), and Catalog B Hybrid recall `85.236%` → `85.273%`, while Catalog A CAP recall (`38.6%`), Catalog A Hybrid recall (`78.7%`), Catalog A gap closed (`65.3%`), and all population metrics remain completely unchanged.
* **OASIS Severity Scoring (`compute_cap_score64`):**
  * Combines CAP `severity`, `certainty`, and `urgency` enumerations (`1..4` each) into a 64-level monotonic score `16 * (sev - 1) + 4 * (cert - 1) + (urg - 1) + 1` (`1..64`), where discrete severity levels `>= 1, 2, 3, 4` correspond identically to `score64 >= 1, 17, 33, 49`.

---

## 4. Global Flood Model Exceedances & Return-Period Provenance

* **Valid-Date Forecast Alignment (`src/cap_alerting_coverage/scoring.py`):**
  * For every multi-lead riverine forecast (DHM `0–10d`, GEOGLOWS v2 operational `0–14d`), an exceedance issued on day `t_issue` at lead `L` days is aligned to its target valid date `t_valid = t_issue + L` before scoring against event windows `[start_date - N, end_date]`. In the lead-window sweep (`lead_window_sweep.csv`), the DHM operational forecast horizon scales as `min(N, 10)` alongside the pre-onset evaluation window `N`.
* **Episode Collapsing (`collapse_warning_episodes`):**
  * Consecutive warning days within an Admin-1 unit separated by at most 2 non-warning days (`d_{k+1} - d_k <= 3`, `max_gap_days = 2`) are collapsed into a single warning episode when computing episode-level precision and false-alarm ratios.
* **Return-Period Thresholds (`[1.01, 2, 5, 10, 20, 25, 50, 100]` yr):**
  * **Google Distributed Hydrological Model (DHM, `extract_dhm.py`):** Consumes precomputed reach-level `return_periods.parquet` fitted via Log-Pearson Type III (exact `gammaincinv`/`gammainccinv` over 45 October–September water years `WY1981–WY2025` on the 4-member ensemble median 0-day hindcast; `< 0.5%` median difference across `Q2–Q20` vs. calendar-year Wilson–Hilferty LP3).
  * **Copernicus GloFAS v5 (`extract_glofas_v5.py`):** Fits Bulletin 17C Wilson–Hilferty Log-Pearson Type III (LP3) distributions in-pipeline over 45 calendar-year annual maxima (`1981-01-02..2025-12-31`, `ddof=1` `log10` moments, skew clipped to `[-2, 2]`, `stats_utils.fit_lp3_return_periods`) for land grid cells with `1981–2025 mean_Q >= 10.0 m³/s`. The staged Zenodo GloFAS v5 cube additionally applied a non-binding `uparea >= 20 km²` candidate pre-filter before the `mean_Q >= 10.0 m³/s` filter (physically identical, as `mean_Q >= 10 m³/s` on `< 20 km²` would require `> 15,700 mm/yr` runoff).
  * **GEOGLOWS v2 (`extract_geoglows_v2.py`):** Consumes precomputed reach-level `rp8` thresholds fitted via **Gumbel (GEV Type I)** in linear discharge (not LP3) on the pre-Rev-4 RAPID-era `1940–present` retrospective archive (`22.8%–25.7%` lower than current Rev-4 `river-route` `daily.zarr`, inflating 2021–2025 retrospective reach-day exceedances by `1.43×` at `Q2`, `2.04×` at `Q5`, `4.27×` at `Q20`, and `12.93×` at `Q100`; `29.33%` of active reaches have `Q1.01 <= 0`, disabling `Q1.01` on those reaches while leaving `Q2..Q100` unaffected).
* **Stream-Gauge Verification (`gauge_verification.py`, Tier-1 vs. Tier-2 Boundary):**
  * Evaluates DHM exceedance precision against observed stream-gauge discharge return periods (Table 2) and compares catchment-area-validated Nash–Sutcliffe Efficiency (NSE) across DHM, GloFAS v5, and GEOGLOWS v2 on their shared gauge universe (Methods §5).
  * Because raw GRDC/USGS/Caravan stream-gauge hydrographs and `dhm_gauged_metrics.parquet` cannot be publicly redistributed, **Tier 1** copies `stream_gauge_nse_summary.csv` verbatim from Zenodo `03_summary_results/` and reads the three in-situ gauge columns (`gauge_n`, `gauge_precision_mean_pct`, `gauge_precision_median_pct`) of `table2_gauge_vs_catalogs.csv` from Zenodo while dynamically recomputing all Catalog A and Catalog B DHM precision/recall columns from `pr_curves_summary.csv`. Full raw gauge recomputation is supported in **Tier 2** when `gauge_metrics_parquet` and `nse_benchmark_csv` are supplied locally.

---

## 5. Statistical Inference

* **Paired Event Comparisons:** Exact two-sided binomial McNemar test (`scipy.stats.binomtest(b, b + c, 0.5)`) on discordant event pairs between CAP and model/hybrid systems (`stats_utils.mcnemar_exact_p`, invoked in `manuscript_outputs.py` for Table 1).
* **Confidence Intervals:**
  * Exact Clopper–Pearson, Wilson score, and Wald 95% confidence intervals are computed in `validation_audit.py` and `stats_utils.py` for binomial validation proportions (`ed_table4_validation_summary.csv`, `validation_confusion_counts.csv`, and `validation_spatial_precision.csv`).
  * Deterministic non-parametric bootstrap utilities (`bootstrap_proportion_ci` and `paired_bootstrap_diff_ci`, defaulting to `n_resamples = 1000`, `seed = 42`) are provided in `stats_utils.py` for ad-hoc uncertainty analysis; the primary benchmark summary CSVs (`pr_curves_summary.csv`, `table1_summary.csv`) report exact event counts, proportions, and exact McNemar *p*-values rather than bootstrap CI columns.

---

## 6. De-Identified `N = 400` CAP Validation Audit

* The de-identified 400-row × 30-column audit table (`04_validation_deidentified/validation_n400_deidentified.csv`) enables deterministic recomputation of all confusion matrices (`validation_confusion_counts.csv`), spatial pairing precision (`validation_spatial_precision.csv`), and Extended Data Table 4 (`ed_table4_validation_summary.csv`) via `cap-gap audit-validation` (`src/cap_alerting_coverage/validation_audit.py`).
* **Sampling Structure & Label Versions:**
  * The 400 rows represent 400 `(event, matched alert)` pairs spanning 245 unique CAP alerts drawn from legacy country-level `±15`-day matches stratified by country area (`> 500,000 km²` vs. `<= 500,000 km²`).
  * Under the re-reviewed `v4_reviewed` label set (default in `validation_audit.py`) and the `v3` label set:
    * **Country-only pairing spatial precision (`spatial_match == "Yes"`):** `86/400 = 21.50%` (`95%` Clopper–Pearson CI `17.57%–25.85%`) in `v4_reviewed`; `39/400 = 9.75%` (`7.03%–13.09%`) in `v3`. (Because this evaluates country-level spatial pairing precision over the full sampled pool rather than thematic recall, conditional recall columns are left `NaN` in `ed_table4_validation_summary.csv`.)
    * **Exclusion filter (`exclusion_keep_all_sampled`):** Thematic precision `220/400 = 55.00%` (`49.98%–59.95%` CP CI) in `v4_reviewed`; `206/400 = 51.50%` (`46.48%–56.50%`) in `v3`; conditional recall `100.0%` by definition (`98.34%–100.00%` CP CI in `v4_reviewed`; degenerate Wald CI `100.00%–100.00%`).
    * **Inclusion filter (`filter02_broad_flood_storm`):** Thematic precision `208/235 = 88.51%` (`83.73%–92.29%` CP CI) and conditional recall `208/220 = 94.55%` (`90.67%–97.15%` CP CI) in `v4_reviewed`; `197/235 = 83.83%` precision and `197/206 = 95.63%` recall in `v3`.
    * **Alternative `sheet_keyword_filter_status_Both` predictor:** `190/289 = 65.74%` precision / `86.36%` recall in `v4_reviewed`; `181/289 = 62.63%` precision / `87.86%` recall in `v3`.
  * **Figure–Table Consistency:** `ed-fig-1.png` panel `a` in `manuscript_outputs.py` plots directly from `ed_table4_validation_summary.csv` using exact Clopper–Pearson 95% confidence intervals rather than hard-coded constants, ensuring exact parity between the plotted figure and the generated table.
