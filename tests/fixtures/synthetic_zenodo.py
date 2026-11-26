"""Deterministic synthetic Zenodo-layout fixture generator for offline CI testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cap_alerting_coverage.config import D_TOTAL, DATES
from cap_alerting_coverage.zenodo_fetch import compute_sha256


def build_synthetic_zenodo_archive(root_dir: Path) -> Path:
    """Create a minimal, deterministic, schema-valid Zenodo archive tree under ``root_dir``.

    Includes 6 jurisdictions matching the real cohort taxonomy:
      - USA, BRA (Cohort A / Category 1: early & late onboarders)
      - KEN (Cohort B / Category 2: non-flood CAP only)
      - TCD (Cohort C / Category 3: no CAP alerts)
      - JPN, BGD (Cohort D: excluded non-CAP warning systems)
    """
    cat_dir = root_dir / "01_event_catalogs"
    mat_dir = root_dir / "02_admin1_exceedance_matrices"
    res_dir = root_dir / "03_summary_results"
    val_dir = root_dir / "04_validation_deidentified"
    for d in (cat_dir, mat_dir, res_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic Supplementary Data 1 (subset of 6 jurisdictions)
    sup1 = pd.DataFrame(
        [
            {
                "ISO3": "USA",
                "Country_or_Territory": "United States",
                "Cohort_and_Category": "Cohort A: Category 1 (Active Flood CAP)",
                "WMO_Registered_Alerting_Authorities": "NOAA NWS",
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": "https://www.weather.gov/",
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": "operating",
            },
            {
                "ISO3": "BRA",
                "Country_or_Territory": "Brazil",
                "Cohort_and_Category": "Cohort A: Category 1 (Active Flood CAP)",
                "WMO_Registered_Alerting_Authorities": "INMET",
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": "https://portal.inmet.gov.br/",
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": "operating",
            },
            {
                "ISO3": "KEN",
                "Country_or_Territory": "Kenya",
                "Cohort_and_Category": "Cohort B: Category 2 (Non-Flood CAP Only)",
                "WMO_Registered_Alerting_Authorities": "KMD",
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": "https://meteo.go.ke/",
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": "operating",
            },
            {
                "ISO3": "TCD",
                "Country_or_Territory": "Chad",
                "Cohort_and_Category": "Cohort C: Category 3 (No CAP Alerts)",
                "WMO_Registered_Alerting_Authorities": "ANAM",
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": "None",
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": "No CAP Feed",
            },
            {
                "ISO3": "JPN",
                "Country_or_Territory": "Japan",
                "Cohort_and_Category": "Cohort D: Excluded (Non-CAP Automated Warning System)",
                "WMO_Registered_Alerting_Authorities": "JMA",
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": "https://www.jma.go.jp/",
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": "No CAP Feed",
            },
            {
                "ISO3": "BGD",
                "Country_or_Territory": "Bangladesh",
                "Cohort_and_Category": "Cohort D: Excluded (Non-CAP Automated Warning System)",
                "WMO_Registered_Alerting_Authorities": "BWDB",
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": "http://www.ffwc.gov.bd/",
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": "No CAP Feed",
            },
        ]
    )
    sup1.to_csv(root_dir / "supplementary_data_1_subset.csv", index=False)

    # 2. GADM Admin-1 index (8 units across the 6 countries)
    adm1_index = pd.DataFrame(
        [
            {"gid_1": "USA.1_1", "country_iso3": "USA"},
            {"gid_1": "USA.2_1", "country_iso3": "USA"},
            {"gid_1": "BRA.1_1", "country_iso3": "BRA"},
            {"gid_1": "BRA.2_1", "country_iso3": "BRA"},
            {"gid_1": "KEN.1_1", "country_iso3": "KEN"},
            {"gid_1": "TCD.1_1", "country_iso3": "TCD"},
            {"gid_1": "JPN.1_1", "country_iso3": "JPN"},
            {"gid_1": "BGD.1_1", "country_iso3": "BGD"},
        ]
    )
    adm1_index.to_parquet(cat_dir / "gadm41_adm1_index.parquet", index=False)
    gid1_arr = adm1_index["gid_1"].to_numpy(dtype=str)
    iso3_arr = adm1_index["country_iso3"].to_numpy(dtype=str)
    n_g = len(gid1_arr)

    # 3. Synthetic Catalog A & strictly nested Catalog B (covering both 2021-2025 & Apr 2024-Dec 2025)
    cat_a = pd.DataFrame(
        [
            {
                "event_id": "USA.1_1_1",
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "start_date": pd.Timestamp("2021-06-15"),
                "end_date": pd.Timestamp("2021-06-17"),
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 1000.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "BRA.1_1_2",
                "gid_1": "BRA.1_1",
                "country_iso3": "BRA",
                "start_date": pd.Timestamp("2022-03-10"),
                "end_date": pd.Timestamp("2022-03-12"),
                "has_idmc": True,
                "has_montandon": True,
                "has_groundsource": False,
                "waterfall_pop": 2500.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "BRA.2_1_3",
                "gid_1": "BRA.2_1",
                "country_iso3": "BRA",
                "start_date": pd.Timestamp("2024-06-20"),
                "end_date": pd.Timestamp("2024-06-22"),
                "has_idmc": False,
                "has_montandon": True,
                "has_groundsource": False,
                "waterfall_pop": 1500.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "KEN.1_1_4",
                "gid_1": "KEN.1_1",
                "country_iso3": "KEN",
                "start_date": pd.Timestamp("2024-08-10"),
                "end_date": pd.Timestamp("2024-08-11"),
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 3000.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "TCD.1_1_5",
                "gid_1": "TCD.1_1",
                "country_iso3": "TCD",
                "start_date": pd.Timestamp("2025-07-05"),
                "end_date": pd.Timestamp("2025-07-07"),
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 2000.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "JPN.1_1_6",
                "gid_1": "JPN.1_1",
                "country_iso3": "JPN",
                "start_date": pd.Timestamp("2023-07-15"),
                "end_date": pd.Timestamp("2023-07-16"),
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 800.0,
                "in_cohort_d_35": True,
                "in_eval_150": False,
                "in_audited_185": True,
                "in_orig_excl_19": True,
            },
            {
                "event_id": "BGD.1_1_7",
                "gid_1": "BGD.1_1",
                "country_iso3": "BGD",
                "start_date": pd.Timestamp("2024-09-01"),
                "end_date": pd.Timestamp("2024-09-04"),
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 5000.0,
                "in_cohort_d_35": True,
                "in_eval_150": False,
                "in_audited_185": True,
                "in_orig_excl_19": True,
            },
        ]
    )
    cat_a.to_parquet(cat_dir / "catalog_without_gs_all_185_2021_2025.parquet", index=False)
    cat_a[cat_a["in_eval_150"]].to_parquet(cat_dir / "catalog_without_gs_2021_2025.parquet", index=False)

    gs_extra = pd.DataFrame(
        [
            {
                "event_id": "GS_USA.2_1_1",
                "gid_1": "USA.2_1",
                "country_iso3": "USA",
                "start_date": pd.Timestamp("2021-09-01"),
                "end_date": pd.Timestamp("2021-09-01"),
                "has_idmc": False,
                "has_montandon": False,
                "has_groundsource": True,
                "waterfall_pop": 0.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "GS_KEN.1_1_2",
                "gid_1": "KEN.1_1",
                "country_iso3": "KEN",
                "start_date": pd.Timestamp("2025-04-12"),
                "end_date": pd.Timestamp("2025-04-13"),
                "has_idmc": False,
                "has_montandon": False,
                "has_groundsource": True,
                "waterfall_pop": 0.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
            {
                "event_id": "GS_TCD.1_1_3",
                "gid_1": "TCD.1_1",
                "country_iso3": "TCD",
                "start_date": pd.Timestamp("2024-10-02"),
                "end_date": pd.Timestamp("2024-10-02"),
                "has_idmc": False,
                "has_montandon": False,
                "has_groundsource": True,
                "waterfall_pop": 0.0,
                "in_cohort_d_35": False,
                "in_eval_150": True,
                "in_audited_185": True,
                "in_orig_excl_19": False,
            },
        ]
    )
    cat_b_parent = cat_a.copy()
    cat_b_parent.loc[0, "has_groundsource"] = True
    cat_b = pd.concat([cat_b_parent, gs_extra], ignore_index=True)
    cat_b.to_parquet(cat_dir / "catalog_with_gs_all_185_2021_2025.parquet", index=False)
    cat_b[cat_b["in_eval_150"]].to_parquet(cat_dir / "catalog_with_gs_2021_2025.parquet", index=False)

    # 4. Synthetic CAP Admin-1 & Admin-0 matrices (strict score64 subsumption)
    cap1 = pd.DataFrame(
        [
            # Matches USA.1_1_1 (2021-06-15) pre-onset with Extreme flood-coded alert
            {
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "alert_date": pd.Timestamp("2021-06-13"),
                "perm_max_sev": np.int8(4),
                "rest_max_sev": np.int8(4),
                "perm_max_score64": np.int16(64),
                "rest_max_score64": np.int16(64),
                "n_perm": np.int32(2),
                "n_rest": np.int32(2),
            },
            # Matches BRA.2_1_3 (2024-06-20) with Moderate broad storm alert
            {
                "gid_1": "BRA.2_1",
                "country_iso3": "BRA",
                "alert_date": pd.Timestamp("2024-06-18"),
                "perm_max_sev": np.int8(2),
                "rest_max_sev": np.int8(0),
                "perm_max_score64": np.int16(24),
                "rest_max_score64": np.int16(0),
                "n_perm": np.int32(1),
                "n_rest": np.int32(0),
            },
            # False-alarm alert episode in USA.2_1 far from any flood
            {
                "gid_1": "USA.2_1",
                "country_iso3": "USA",
                "alert_date": pd.Timestamp("2022-01-10"),
                "perm_max_sev": np.int8(3),
                "rest_max_sev": np.int8(0),
                "perm_max_score64": np.int16(40),
                "rest_max_score64": np.int16(0),
                "n_perm": np.int32(1),
                "n_rest": np.int32(0),
            },
        ]
    )
    cap1.to_parquet(mat_dir / "cap_gid1_daily_2020_2025.parquet", index=False)

    cap0 = cap1.groupby(["country_iso3", "alert_date"], as_index=False).agg(
        perm_max_sev=("perm_max_sev", "max"),
        rest_max_sev=("rest_max_sev", "max"),
        perm_max_score64=("perm_max_score64", "max"),
        rest_max_score64=("rest_max_score64", "max"),
        n_perm=("n_perm", "sum"),
        n_rest=("n_rest", "sum"),
    )
    cap0.to_parquet(mat_dir / "cap_gid0_daily_2020_2025.parquet", index=False)

    # 5. Synthetic FFM daily probabilities (detects KEN.1_1_4 and GS_TCD.1_1_3)
    ffm = pd.DataFrame(
        [
            {
                "gid_1": "KEN.1_1",
                "country_iso3": "KEN",
                "valid_date": pd.Timestamp("2024-08-09"),
                "max_prob": np.float32(0.75),
            },
            {
                "gid_1": "TCD.1_1",
                "country_iso3": "TCD",
                "valid_date": pd.Timestamp("2024-10-01"),
                "max_prob": np.float32(0.62),
            },
            {
                "gid_1": "USA.2_1",
                "country_iso3": "USA",
                "valid_date": pd.Timestamp("2021-08-31"),
                "max_prob": np.float32(0.55),
            },
        ]
    )
    ffm.to_parquet(mat_dir / "ffm_gid1_daily_2021_2025.parquet", index=False)

    # 6. Synthetic riverine tier cubes (DHM 10D, GeoGLOWS Retro 0D, GeoGLOWS Oper, GloFAS v5 0D)
    dhm_tiers = np.zeros((n_g, D_TOTAL, 11), dtype=np.uint8)
    # BRA.1_1_2 (2022-03-10 -> day idx 450): predicted 3 days ahead (issue 447, lead 3 -> valid 450)
    idx_bra1 = int((pd.Timestamp("2022-03-10") - DATES[0]).days)
    dhm_tiers[2, idx_bra1 - 3, 3] = 4  # RP >= 10yr
    dhm_tiers[2, idx_bra1, 0] = 3  # RP >= 5yr on 0d
    # TCD.1_1_5 (2025-07-05): predicted at lead 2
    idx_tcd = int((pd.Timestamp("2025-07-05") - DATES[0]).days)
    dhm_tiers[5, idx_tcd - 2, 2] = 2  # RP >= 2yr
    # False alarm in KEN.1_1
    dhm_tiers[4, 200, 0] = 2

    date_strs = np.asarray(DATES.strftime("%Y-%m-%d").to_numpy(), dtype=str)
    np.savez_compressed(
        mat_dir / "dhm_10d_gid1_multilead_2021_2025.npz",
        tiers=dhm_tiers,
        gid_1=gid1_arr,
        country_iso3=iso3_arr,
        issue_dates=date_strs,
    )

    geo_retro = np.zeros((n_g, D_TOTAL), dtype=np.uint8)
    geo_retro[0, int((pd.Timestamp("2021-06-15") - DATES[0]).days)] = 2
    geo_retro[5, idx_tcd] = 3
    np.savez_compressed(
        mat_dir / "geoglows_v2_retro_gid1_daily_2021_2025.npz",
        tiers=geo_retro,
        gid_1=gid1_arr,
        country_iso3=iso3_arr,
        valid_dates=date_strs,
    )

    oper_dates = pd.date_range("2024-04-01", periods=640, freq="D")
    geo_oper = np.zeros((n_g, 640, 4, 15), dtype=np.uint8)
    tcd_oper_day = int((pd.Timestamp("2025-07-05") - oper_dates[0]).days)
    geo_oper[5, tcd_oper_day - 2, 0, 2] = 2  # EnsMedian
    geo_oper[5, tcd_oper_day - 2, 3, 2] = 3  # HRES
    np.savez_compressed(
        mat_dir / "geoglows_v2_oper_gid1_multilead_2024_2025.npz",
        tiers=geo_oper,
        gid_1=gid1_arr,
        country_iso3=iso3_arr,
        issue_dates=np.asarray(oper_dates.strftime("%Y-%m-%d").to_numpy(), dtype=str),
        products=np.asarray(["EnsMedian", "EnsMean", "EnsP75", "HRES"], dtype=str),
    )

    glofas_tiers = np.zeros((n_g, D_TOTAL), dtype=np.uint8)
    glofas_tiers[2, idx_bra1] = 2
    np.savez_compressed(
        mat_dir / "glofas_v5_gid1_daily_2021_2025.npz",
        tiers=glofas_tiers,
        gid_1=gid1_arr,
        country_iso3=iso3_arr,
        valid_dates=date_strs,
    )

    # 7. Auxiliary onboarding, hazard counts, gauge summary, and de-identified N=400 validation CSV
    pd.DataFrame(
        [
            {
                "country_iso3": "USA",
                "first_any_cap_dt": "2021-01-01T00:00:00Z",
                "last_any_cap_dt": "2025-12-31T00:00:00Z",
                "first_flood_perm_dt": "2021-01-05T00:00:00Z",
                "n_any_total": 100,
                "n_2021": 20,
                "n_2022": 20,
                "n_2023": 20,
                "n_2024": 20,
                "n_2025": 20,
                "onboard_year_2021_2025": 2021,
            },
            {
                "country_iso3": "BRA",
                "first_any_cap_dt": "2023-06-01T00:00:00Z",
                "last_any_cap_dt": "2025-12-31T00:00:00Z",
                "first_flood_perm_dt": "2023-06-10T00:00:00Z",
                "n_any_total": 40,
                "n_2021": 0,
                "n_2022": 0,
                "n_2023": 10,
                "n_2024": 15,
                "n_2025": 15,
                "onboard_year_2021_2025": 2023,
            },
            {
                "country_iso3": "KEN",
                "first_any_cap_dt": "2022-05-01T00:00:00Z",
                "last_any_cap_dt": "2025-12-01T00:00:00Z",
                "first_flood_perm_dt": "",
                "n_any_total": 15,
                "n_2021": 0,
                "n_2022": 5,
                "n_2023": 5,
                "n_2024": 3,
                "n_2025": 2,
                "onboard_year_2021_2025": 2022,
            },
        ]
    ).to_csv(res_dir / "cap_country_onboarding_2021_2025.csv", index=False)

    pd.DataFrame(
        [
            {
                "country_iso3": "USA",
                "n_any_total": 100,
                "n_storms": 50,
                "n_cold": 20,
                "n_heat": 15,
                "n_wildfire": 10,
                "n_earthquake": 5,
                "n_volcano": 2,
                "n_flood_perm": 40,
                "n_flood_rest": 15,
            },
            {
                "country_iso3": "BRA",
                "n_any_total": 40,
                "n_storms": 25,
                "n_cold": 3,
                "n_heat": 5,
                "n_wildfire": 2,
                "n_earthquake": 0,
                "n_volcano": 0,
                "n_flood_perm": 20,
                "n_flood_rest": 5,
            },
            {
                "country_iso3": "KEN",
                "n_any_total": 15,
                "n_storms": 5,
                "n_cold": 0,
                "n_heat": 10,
                "n_wildfire": 0,
                "n_earthquake": 0,
                "n_volcano": 0,
                "n_flood_perm": 0,
                "n_flood_rest": 0,
            },
        ]
    ).to_csv(res_dir / "cap_country_hazard_counts_2021_2025.csv", index=False)

    pd.DataFrame(
        [
            {"rp_yr": rp, "gauge_n": 100, "gauge_precision_mean_pct": 50.0, "gauge_precision_median_pct": 52.0}
            for rp in (2, 5, 10, 20, 25, 50, 100)
        ]
    ).to_csv(res_dir / "table2_gauge_vs_catalogs.csv", index=False)

    pd.DataFrame(
        [
            {
                "unfiltered_total_gauges": 100,
                "unfiltered_valid_nse": 95,
                "unfiltered_median_nse": 0.73,
                "unfiltered_pct_pos_nse": 90.5,
                "area_validated_n_dhm": 80,
                "area_validated_median_nse_dhm": 0.76,
                "area_validated_pct_pos_dhm": 92.5,
                "area_validated_n_glofas": 80,
                "area_validated_median_nse_glofas": 0.35,
                "area_validated_pct_pos_glofas": 73.8,
                "area_validated_n_geoglows": 70,
                "area_validated_median_nse_geoglows": 0.09,
                "area_validated_pct_pos_geoglows": 59.0,
                "common_3model_n": 70,
                "common_3model_median_nse_dhm": 0.76,
                "common_3model_pct_pos_dhm": 93.2,
                "common_3model_median_nse_glofas": 0.34,
                "common_3model_pct_pos_glofas": 74.3,
                "common_3model_median_nse_geoglows": 0.09,
                "common_3model_pct_pos_geoglows": 59.0,
            }
        ]
    ).to_csv(res_dir / "stream_gauge_nse_summary.csv", index=False)

    # Synthetic de-identified N=400 validation table using the exact 30-column public schema
    val_rows = []
    for idx in range(400):
        sid = idx + 1
        rnd = 1 if sid <= 200 else 2
        stratum = "large_gt_500k_km2" if (idx % 2 == 0) else "small_medium_le_500k_km2"
        v3_them = "Yes" if idx < 206 else "No"
        v4_them = "Yes" if idx < 220 else "No"
        v3_spat = "Yes" if idx < 39 else "No"
        v4_spat = "Yes" if idx < 86 else "No"
        broad_f = (idx < 208) or (220 <= idx < 247)  # 208 TP + 27 FP = 235 predicted
        sheet_both = (idx < 190) or (220 <= idx < 319)  # 289 Both
        val_rows.append(
            {
                "sample_id": sid,
                "sample_round": rnd,
                "country_area_stratum": stratum,
                "cohort_d_excluded_country": False,
                "event_year": 2023,
                "alert_year": 2023,
                "event_in_study_period_2021_2025": True,
                "alert_lead_window": "alert_0_to_7d_before_onset",
                "alert_group_id": (idx % 245) + 1,
                "alert_shared_with_other_rows": idx >= 245,
                "alert_cap_severity": "Moderate",
                "event_hazard_type": "Riverine Flood" if idx % 2 == 0 else "Flash Flood",
                "keyword_filter_status_sheet": "Both" if sheet_both else "Exclusion-Only",
                "spatial_match_v1": v3_spat if rnd == 1 else "",
                "thematic_match_v1": v3_them if rnd == 1 else "",
                "spatial_match_v3": v3_spat,
                "thematic_match_v3": v3_them,
                "alert_type_v3": "Hydrological" if v3_them == "Yes" else "",
                "spatial_match_v4": v4_spat,
                "thematic_match_v4": v4_them,
                "alert_type_v4": "Hydrological" if v4_them == "Yes" else "N/A",
                "reviewer2_reviewed_v4": sid <= 10,
                "reviewer2_verdict_v4": "agree" if sid <= 10 else "",
                "occurrences_in_v4_v6": 2 if sid <= 3 else 1,
                "reviewer2_reviewed_v4_v6": sid <= 10,
                "filter02_event_excluded": idx >= 330,
                "filter02_broad_flood_storm": broad_f,
                "filter02_broad_flood_storm_with_sheet_translation": broad_f,
                "filter02_explicit_flood": idx < 120,
                "annex_inclusion_keyword_any_text": broad_f,
            }
        )
    pd.DataFrame(val_rows).to_csv(val_dir / "validation_n400_deidentified.csv", index=False)

    # 8. Write SHA256SUMS.txt covering all staged files
    manifest_lines = []
    for p in sorted(root_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel = p.relative_to(root_dir).as_posix()
            manifest_lines.append(f"{compute_sha256(p)}  {rel}")
    (root_dir / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return root_dir
