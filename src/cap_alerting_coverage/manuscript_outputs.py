"""Manuscript tables and publication-grade single-composite figure generation (Tier 1)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cap_alerting_coverage.config import D_TOTAL, DATES, FFM_MAX_PRE_DAYS, ProjectPaths, date_to_idx
from cap_alerting_coverage.figure_fleet import (
    render_ed_fig1,
    render_ed_fig2,
    render_ed_fig3,
    render_ed_fig4,
    render_ed_fig5,
    render_fig1,
    render_fig2,
    render_fig3,
    render_fig4,
    render_fig5,
    render_supp_fig1,
)
from cap_alerting_coverage.figure_style import load_world_robinson, setup_nature_style
from cap_alerting_coverage.gauge_verification import run_gauge_verification
from cap_alerting_coverage.scoring import (
    load_benchmark_data,
    load_universe_sets,
    make_w_matrix,
)
from cap_alerting_coverage.stats_utils import (
    country_clustered_coverage_inference,
    mcnemar_exact_p,
)
from cap_alerting_coverage.validation_audit import run_validation_audit

# Public backward-compatible alias
setup_publication_style = setup_nature_style


def build_event_detection_flags(
    cat_df: pd.DataFrame,
    w_bool: np.ndarray,
    gid_to_idx: dict[str, int],
    b0_s: str,
    b1_s: str,
    lead_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean row mask inside ``[b0_s, b1_s]`` and binary per-event detection array."""
    b0 = date_to_idx(b0_s)
    b1 = date_to_idx(b1_s)
    w_min = max(0, b0 - lead_n)
    n_g = w_bool.shape[0]
    prefix = np.zeros((n_g, D_TOTAL + 1), dtype=np.int32)
    np.cumsum(w_bool, axis=1, out=prefix[:, 1:])

    sd = (pd.to_datetime(cat_df["start_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    ed = (pd.to_datetime(cat_df["end_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    gi = cat_df["gid_1"].map(gid_to_idx)
    valid = gi.notna().to_numpy() & (sd >= b0) & (sd <= b1)
    g_arr = gi[valid].astype(np.int32).to_numpy()
    s_arr = sd[valid]
    e_arr = np.maximum(s_arr, np.minimum(ed[valid], b1))
    w0_arr = np.maximum(w_min, s_arr - lead_n)
    det = (prefix[g_arr, e_arr + 1] - prefix[g_arr, w0_arr]) > 0
    return valid, det


def _clean_auth_name(iso: str, raw_auth: str) -> str:
    s = re.sub(r"\s*\(recId=[^)]+\)", "", str(raw_auth)).strip()
    s = s.replace("Direction de la Météorologie Natinale", "Direction Générale de la Météorologie (DGM)")
    repls = {
        "FJI": "Fiji Meteorological Service (FMS); Suva Water Supplies",
        "LUX": "Administration de la gestion de l'eau (AGE); Administration de l'Aéroport de Luxembourg",
        "ZAF": "South African Weather Service (SAWS); Department of Water and Sanitation",
        "HUN": "HungaroMet (Hungarian Meteorological Service); General Directorate of Water Management (OVF / VITUKI)",
    }
    return repls.get(iso, s)


def _clean_pipe_list(raw_val: str) -> str:
    return re.sub(r"\s*\|\s*", "; ", str(raw_val)).strip()


def build_country_clustered_inference_table(
    data: dict[str, Any],
    gid_to_idx: dict[str, int],
    eval_150: set[str],
    n_g: int,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute country-level paired Wilcoxon tests, DEFF, and clustered 95% CIs for 2021–2025 systems."""
    w_cap = make_w_matrix(data, "CAP_Perm", 1, 7, n_g)
    w_dhm_oper = make_w_matrix(data, "DHM_Oper", 2, 7, n_g)
    w_dhm_0d = make_w_matrix(data, "DHM_0d", 2, 7, n_g)
    w_ffm = make_w_matrix(data, "FFM", 0.50, 7, n_g)
    w_glofas = make_w_matrix(data, "GloFAS_v5_0d", 2, 7, n_g)
    w_geoglows = make_w_matrix(data, "GeoGLOWS_Retro_0d", 2, 7, n_g)

    # Pairs of (system_name, thr_label, non_ffm_matrix_7d, include_ffm_1d)
    candidates: list[tuple[str, str, np.ndarray, bool]] = [
        ("CAP_Perm_U_GloFAS_v5_0d", "RP=2", w_cap | w_glofas, False),
        ("CAP_Perm_U_GeoGLOWS_Retro_0d", "RP=2", w_cap | w_geoglows, False),
        ("CAP_Perm_U_DHM_0d", "RP=2", w_cap | w_dhm_0d, False),
        ("CAP_Perm_U_DHM_Oper", "RP=2", w_cap | w_dhm_oper, False),
        ("CAP_Perm_U_FFM", "tau=0.50", w_cap, True),
        ("CAP_Perm_U_DHM_Oper_U_FFM050", "RP=2", w_cap | w_dhm_oper, True),
    ]

    rows: list[dict[str, Any]] = []
    for cat_k, cat_df in (("Without_GS", data["cat_without_gs"]), ("With_GS", data["cat_with_gs"])):
        v_mask, det_cap = build_event_detection_flags(cat_df, w_cap, gid_to_idx, "2021-01-01", "2025-12-31", 7)
        _, det_ffm = build_event_detection_flags(
            cat_df, w_ffm, gid_to_idx, "2021-01-01", "2025-12-31", FFM_MAX_PRE_DAYS
        )
        ev = cat_df[v_mask].copy()
        in_eval = ev["country_iso3"].isin(eval_150).to_numpy()
        if not np.any(in_eval):
            continue
        countries = ev.loc[in_eval, "country_iso3"].to_numpy()
        pop_w = ev.loc[in_eval, "waterfall_pop"].to_numpy(dtype=np.float64)
        base_det = det_cap[in_eval]

        for sys_name, thr_lbl, w_other, use_ffm in candidates:
            _, det_other = build_event_detection_flags(cat_df, w_other, gid_to_idx, "2021-01-01", "2025-12-31", 7)
            det_cand = (det_other | det_ffm) if use_ffm else det_other
            inf = country_clustered_coverage_inference(
                country_ids=countries,
                baseline=base_det,
                candidate=det_cand[in_eval],
                weights=pop_w,
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                {
                    "catalog": cat_k,
                    "benchmark": "2021_2025",
                    "system": sys_name,
                    "thr_label": thr_lbl,
                    **{
                        k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) and k != "wilcoxon_p" else v)
                        for k, v in inf.items()
                    },
                }
            )
    return pd.DataFrame(rows)


def build_table1_summary(df_pr: pd.DataFrame) -> pd.DataFrame:
    """Assemble Table 1 summary rows with exact two-sided binomial McNemar p-values."""
    t1_specs = [
        ("CAP_Perm", "All", "Official CAP Baseline (Broad Flood-and-Storm)"),
        ("CAP_Rest", "All", "Official CAP Explicit Flood-Coded Only"),
        ("FFM", "tau=0.50", "Google FFM (0–24h, τ=0.50)"),
        ("DHM_0d", "RP=2", "Google DHM (0d Nowcast, Q2yr)"),
        ("DHM_Oper", "RP=2", "Google DHM Operational (0–7d valid-date aligned, Q2yr)"),
        ("GloFAS_v5_0d", "RP=2", "Copernicus GloFAS v5 (0d, Q2yr)"),
        ("GeoGLOWS_Retro_0d", "RP=2", "GeoGLOWS v2 Retro (0d, Q2yr)"),
        ("GeoGLOWS_Oper_EnsMedian", "RP=2", "GeoGLOWS v2 Oper EnsMedian (0–7d valid-date aligned, Q2yr)"),
        ("GeoGLOWS_Oper_HRES", "RP=2", "GeoGLOWS v2 Control HRES (0–7d valid-date aligned, Q2yr)"),
        ("DHM_Oper_U_FFM050", "RP=2", "Google Combined (DHM 0–7d Q2yr + FFM τ=0.50)"),
        ("CAP_Perm_U_GloFAS_v5_0d", "RP=2", "+ CAP ∪ GloFAS v5 (0d, Q2yr)"),
        ("CAP_Perm_U_GeoGLOWS_Retro_0d", "RP=2", "+ CAP ∪ GeoGLOWS v2 Retro (0d, Q2yr)"),
        ("CAP_Perm_U_GeoGLOWS_Oper_EnsMedian", "RP=2", "+ CAP ∪ GeoGLOWS v2 EnsMedian (0–7d, Q2yr)"),
        ("CAP_Perm_U_GeoGLOWS_Oper_HRES", "RP=2", "+ CAP ∪ GeoGLOWS v2 Control HRES (0–7d, Q2yr)"),
        ("CAP_Perm_U_DHM_0d", "RP=2", "+ CAP ∪ Google DHM (0d, Q2yr)"),
        ("CAP_Perm_U_DHM_Oper", "RP=2", "+ CAP ∪ Google DHM (0–7d valid-date aligned, Q2yr)"),
        ("CAP_Perm_U_FFM", "tau=0.50", "+ CAP ∪ Google FFM (0–24h, τ=0.50)"),
        ("CAP_Perm_U_DHM_Oper_U_FFM050", "RP=2", "+ CAP ∪ DHM (0–7d) ∪ FFM (0–24h) [Hybrid Full]"),
    ]
    rows = []
    for cat_k in ("Without_GS", "With_GS"):
        for bm_k in ("2021_2025", "2024_2025"):
            sub = df_pr[(df_pr["catalog"] == cat_k) & (df_pr["benchmark"] == bm_k) & (df_pr["lead_N"] == 7)]
            base_matches = sub[(sub["system"] == "CAP_Perm") & (sub["thr_label"] == "All")]
            if base_matches.empty:
                continue
            base_row = base_matches.iloc[0]
            base_tp = int(base_row["tp_ev"])
            n_ev = int(base_row["n_events"])
            unalerted = n_ev - base_tp
            for sk, tl, lbl in t1_specs:
                m = sub[(sub["system"] == sk) & (sub["thr_label"] == tl)]
                if m.empty:
                    continue
                r = m.iloc[0]
                tp_ev = int(r["tp_ev"])
                net_vs_cap = tp_ev - base_tp
                exact_pp_gain = 100.0 * net_vs_cap / max(1, n_ev)
                gap_closed = (100.0 * net_vs_cap / max(1, unalerted)) if sk.startswith("CAP_Perm_U_") else np.nan
                p_val = mcnemar_exact_p(net_vs_cap, 0) if (sk.startswith("CAP_Perm_U_") and net_vs_cap > 0) else np.nan
                rows.append(
                    {
                        "catalog": cat_k,
                        "benchmark": bm_k,
                        "system": sk,
                        "label": lbl,
                        "thr_label": tl,
                        "n_events": n_ev,
                        "tp_ev": tp_ev,
                        "tp_pre": int(r["tp_pre"]),
                        "recall_pct": r["recall_pct"],
                        "pre_recall_pct": r["pre_recall_pct"],
                        "pop_recall_pct": r["pop_recall_pct"],
                        "n_episodes": int(r["n_episodes"]),
                        "tp_ep": int(r["tp_ep"]),
                        "precision_pct": r["precision_pct"],
                        "f1_pct": r["f1_pct"],
                        "net_gain_over_cap": net_vs_cap if sk.startswith("CAP_Perm_U_") else np.nan,
                        "pp_gain_over_cap": round(exact_pp_gain, 3) if sk.startswith("CAP_Perm_U_") else np.nan,
                        "gap_closed_pct": round(gap_closed, 2) if np.isfinite(gap_closed) else np.nan,
                        "mcnemar_p": p_val,
                    }
                )
    return pd.DataFrame(rows)


def render_all_publication_figures(
    paths: ProjectPaths,
    supp: pd.DataFrame,
    c_df: pd.DataFrame,
    ed6b_df: pd.DataFrame,
    haz: pd.DataFrame,
    df_pr: pd.DataFrame,
    df_fine: pd.DataFrame,
    df_leads: pd.DataFrame,
    df_single: pd.DataFrame,
    df_ffm_slices: pd.DataFrame,
    df_aud: pd.DataFrame,
    ed4_df: pd.DataFrame,
    t2_df: pd.DataFrame | None = None,
    ecdf_df: pd.DataFrame | None = None,
    iqr_df: pd.DataFrame | None = None,
) -> list[Path]:
    """Render all 11 single-composite publication figures in PDF, SVG, and 300-DPI PNG."""
    world_rob, frame_gdf = load_world_robinson(paths.countries_geojson)
    out_dir = paths.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # Main Figures 1-5
    written.extend(render_fig1(world_rob, frame_gdf, supp, haz, out_dir))
    written.extend(render_fig2(world_rob, frame_gdf, supp, c_df, ed6b_df, out_dir))
    written.extend(render_fig3(df_pr, df_fine, out_dir))
    written.extend(render_fig4(world_rob, frame_gdf, supp, c_df, out_dir))
    written.extend(render_fig5(out_dir))
    # Extended Data Figures 1-5
    written.extend(render_ed_fig1(df_aud, ed4_df, out_dir))
    written.extend(render_ed_fig2(df_pr, df_fine, out_dir))
    written.extend(render_ed_fig3(df_leads, df_single, out_dir))
    written.extend(render_ed_fig4(df_ffm_slices, df_fine, out_dir))
    if t2_df is not None and not t2_df.empty:
        written.extend(render_ed_fig5(t2_df, ecdf_df, iqr_df, out_dir))
    # Supplementary Figure 1
    written.extend(render_supp_fig1(world_rob, frame_gdf, supp, c_df, ed6b_df, out_dir))
    return written


def generate_manuscript_tables_and_figures(
    paths: ProjectPaths,
    scoring_state: dict[str, Any] | None = None,
    strict_185: bool = True,
    render_figures: bool = True,
) -> dict[str, Any]:
    """Generate all manuscript & Extended Data CSV tables and 11 single-composite figures."""
    t0 = time.time()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    supp = pd.read_csv(paths.supplementary_data_1_csv, keep_default_na=False)
    supp["ISO3"] = supp["ISO3"].str.strip()
    cat1_isos = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort A")]["ISO3"])
    cat2_isos = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort B")]["ISO3"])
    cat3_isos = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort C")]["ISO3"])
    cohort_d = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"])
    eval_150 = cat1_isos | cat2_isos | cat3_isos
    if strict_185:
        if not (len(cat1_isos) == 70 and len(cat2_isos) == 45 and len(cat3_isos) == 35 and len(cohort_d) == 35):
            raise AssertionError("Jurisdiction cohort sizes do not match 70 / 45 / 35 / 35.")

    if scoring_state is not None:
        gid1_list = scoring_state["gid1_list"]
        gid_to_idx = scoring_state["gid_to_idx"]
        data = scoring_state["data"]
        df_pr = scoring_state["df_pr"]
        df_aud = scoring_state["df_audit"]
        df_leads = scoring_state["df_leads"]
        df_single = scoring_state["df_single"]
        df_ffm_slices = scoring_state["df_ffm_slices"]
        df_fine = scoring_state.get("df_fine_ffm", pd.DataFrame())
    else:
        gid1_list, iso3_list, gid_to_idx, _ = load_universe_sets(paths, strict_185=strict_185)
        data = load_benchmark_data(paths, gid1_list, iso3_list, gid_to_idx)
        df_pr = pd.read_csv(paths.results_dir / "pr_curves_summary.csv")
        df_aud = pd.read_csv(paths.results_dir / "sensitivity_audits.csv")
        df_leads = pd.read_csv(paths.results_dir / "lead_window_sweep.csv")
        df_single = pd.read_csv(paths.results_dir / "single_lead_decay.csv")
        df_ffm_slices = pd.read_csv(paths.results_dir / "ffm_gs_slices_pr.csv")
        fine_p = paths.results_dir / "fine_ffm_pr_curves.csv"
        df_fine = pd.read_csv(fine_p) if fine_p.exists() else pd.DataFrame()

    n_g = len(gid1_list)

    onb = pd.read_csv(
        paths.resolve_data_file(
            "cap_country_onboarding_2021_2025.csv", "03_summary_results", "02_admin1_exceedance_matrices"
        ),
        keep_default_na=False,
    )
    haz = pd.read_csv(
        paths.resolve_data_file(
            "cap_country_hazard_counts_2021_2025.csv", "03_summary_results", "02_admin1_exceedance_matrices"
        ),
        keep_default_na=False,
    )
    haz.to_csv(paths.results_dir / "cap_country_hazard_counts_2021_2025.csv", index=False)

    # 1. Table 1, ED Table 3 (sensitivity), ED Table 5 (full benchmark)
    t1_df = build_table1_summary(df_pr)
    t1_df.to_csv(paths.results_dir / "table1_summary.csv", index=False)
    df_aud.to_csv(paths.results_dir / "ed_table3_sensitivity_matrix.csv", index=False)
    df_pr[df_pr["lead_N"] == 7].to_csv(paths.results_dir / "ed_table5_full_benchmark.csv", index=False)

    # 2. Stream-gauge verification (Extended Data Table 5 & Extended Data Fig. 5 summaries)
    gm_path = _optional_resolve(
        paths, "dhm_gauged_metrics.parquet", ("02_admin1_exceedance_matrices", "03_summary_results")
    )
    nse_path = _optional_resolve(paths, "dhm_vs_glofas_geoglows_benchmark_nse.csv", ("03_summary_results",))
    ref_t2 = _optional_resolve(paths, "table2_gauge_vs_catalogs.csv", ("03_summary_results",))
    ref_nse = _optional_resolve(paths, "stream_gauge_nse_summary.csv", ("03_summary_results",))
    ref_ecdf = _optional_resolve(paths, "stream_gauge_nse_ecdf_quantiles.csv", ("03_summary_results",))
    ref_iqr = _optional_resolve(paths, "stream_gauge_precision_iqr.csv", ("03_summary_results",))
    t2_df, nse_summary = None, None
    if gm_path is not None or ref_t2 is not None:
        t2_df, nse_summary = run_gauge_verification(
            pr_curves_csv=paths.results_dir / "pr_curves_summary.csv",
            results_dir=paths.results_dir,
            gauge_metrics_parquet=gm_path,
            nse_benchmark_csv=nse_path,
            reference_table2_csv=ref_t2,
            reference_nse_summary_csv=ref_nse,
            reference_nse_ecdf_csv=ref_ecdf,
            reference_precision_iqr_csv=ref_iqr,
        )

    # 3. De-identified N=400 validation summary
    val_csv = _optional_resolve(paths, "validation_n400_deidentified.csv", ("04_validation_deidentified",))
    if val_csv is None:
        val_csv = _optional_resolve(paths, "validation_audit_deidentified_n400.csv", ("04_validation_deidentified",))
    ed4_df = run_validation_audit(
        deidentified_csv=val_csv,
        output_csv=paths.results_dir / "ed_table4_validation_summary.csv",
    )

    # 4. Country-level coverage, hybrid gains, and post-onboarding tables (Catalog A & Catalog B)
    w_cap7 = make_w_matrix(data, "CAP_Perm", 1, 7, n_g)
    w_dhm7 = make_w_matrix(data, "DHM_Oper", 2, 7, n_g)
    w_ffm = make_w_matrix(data, "FFM", 0.50, 7, n_g)
    w_hyb_daily = w_cap7 | w_dhm7 | w_ffm

    cat_a = data["cat_without_gs"]
    cat_b = data["cat_with_gs"]
    va_cap, da_cap = build_event_detection_flags(cat_a, w_cap7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, da_dhm = build_event_detection_flags(cat_a, w_dhm7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, da_ffm = build_event_detection_flags(cat_a, w_ffm, gid_to_idx, "2021-01-01", "2025-12-31", FFM_MAX_PRE_DAYS)
    da_mod = da_dhm | da_ffm
    da_hyb = da_cap | da_mod

    vb_cap, db_cap = build_event_detection_flags(cat_b, w_cap7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, db_dhm = build_event_detection_flags(cat_b, w_dhm7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, db_ffm = build_event_detection_flags(cat_b, w_ffm, gid_to_idx, "2021-01-01", "2025-12-31", FFM_MAX_PRE_DAYS)
    db_mod = db_dhm | db_ffm
    db_hyb = db_cap | db_mod

    ev_a = cat_a[va_cap].copy()
    ev_a["cov_cap"] = da_cap
    ev_a["cov_mod"] = da_mod
    ev_a["cov_hyb"] = da_hyb

    ev_b = cat_b[vb_cap].copy()
    ev_b["cov_cap"] = db_cap
    ev_b["cov_mod"] = db_mod
    ev_b["cov_hyb"] = db_hyb

    onb_map = onb.set_index("country_iso3").to_dict("index")
    c_rows = []
    ed6b_rows = []
    for _, sr in supp.iterrows():
        iso = sr["ISO3"]
        coh = sr["Cohort_and_Category"]
        o_info = onb_map.get(iso, {})
        first_any = str(o_info.get("first_any_cap_dt") or "")[:10]
        first_fl = str(o_info.get("first_flood_perm_dt") or "")[:10]
        onb_yr = int(o_info.get("onboard_year_2021_2025") or 0)

        sa = ev_a[ev_a["country_iso3"] == iso]
        sb = ev_b[ev_b["country_iso3"] == iso]
        na, nb = len(sa), len(sb)
        ca, cb = int(sa["cov_cap"].sum()), int(sb["cov_cap"].sum())
        ha, hb = int(sa["cov_hyb"].sum()), int(sb["cov_hyb"].sum())
        pa_tot = float(sa["waterfall_pop"].sum())
        pa_cov = float(sa.loc[sa["cov_cap"], "waterfall_pop"].sum())

        pct_a = (100.0 * ca / na) if na > 0 else np.nan
        pct_ha = (100.0 * ha / na) if na > 0 else np.nan
        pct_pa = (100.0 * pa_cov / pa_tot) if pa_tot > 0 else np.nan
        pct_b = (100.0 * cb / nb) if nb > 0 else np.nan
        pct_hb = (100.0 * hb / nb) if nb > 0 else np.nan

        if first_any and na > 0:
            cutoff = pd.Timestamp(first_any) - pd.Timedelta(days=7)
            sa_post = sa[pd.to_datetime(sa["start_date"]) >= cutoff]
            n_post_a = len(sa_post)
            c_post_a = int(sa_post["cov_cap"].sum())
            pct_post_a = (100.0 * c_post_a / n_post_a) if n_post_a > 0 else np.nan
        else:
            n_post_a, c_post_a, pct_post_a = na, ca, pct_a

        if first_any and nb > 0:
            cutoff_b = pd.Timestamp(first_any) - pd.Timedelta(days=7)
            sb_post = sb[pd.to_datetime(sb["start_date"]) >= cutoff_b]
            n_post_b = len(sb_post)
            c_post_b = int(sb_post["cov_cap"].sum())
            pct_post_b = (100.0 * c_post_b / n_post_b) if n_post_b > 0 else np.nan
        else:
            n_post_b, c_post_b, pct_post_b = nb, cb, pct_b

        c_rows.append(
            {
                "ISO3": iso,
                "Country": sr["Country_or_Territory"],
                "Cohort_and_Category": coh,
                "first_any_cap_date": first_any,
                "first_flood_cap_date": first_fl,
                "onboard_year": onb_yr,
                "n_events_wo_gs": na,
                "cap_cov_wo_gs": ca,
                "cap_pct_wo_gs": round(pct_a, 2) if np.isfinite(pct_a) else np.nan,
                "hyb_cov_wo_gs": ha,
                "hyb_pct_wo_gs": round(pct_ha, 2) if np.isfinite(pct_ha) else np.nan,
                "pp_gain_wo_gs": round(pct_ha - pct_a, 2) if (np.isfinite(pct_ha) and np.isfinite(pct_a)) else np.nan,
                "pop_tot_wo_gs": round(pa_tot, 1),
                "pop_cov_wo_gs": round(pa_cov, 1),
                "pop_pct_wo_gs": round(pct_pa, 2) if np.isfinite(pct_pa) else np.nan,
                "post_onb_n_wo_gs": n_post_a,
                "post_onb_cap_wo_gs": c_post_a,
                "post_onb_pct_wo_gs": round(pct_post_a, 2) if np.isfinite(pct_post_a) else np.nan,
                "n_events_w_gs": nb,
                "cap_cov_w_gs": cb,
                "cap_pct_w_gs": round(pct_b, 2) if np.isfinite(pct_b) else np.nan,
                "hyb_cov_w_gs": hb,
                "hyb_pct_w_gs": round(pct_hb, 2) if np.isfinite(pct_hb) else np.nan,
                "pp_gain_w_gs": round(pct_hb - pct_b, 2) if (np.isfinite(pct_hb) and np.isfinite(pct_b)) else np.nan,
                "post_onb_n_w_gs": n_post_b,
                "post_onb_cap_w_gs": c_post_b,
                "post_onb_pct_w_gs": round(pct_post_b, 2) if np.isfinite(pct_post_b) else np.nan,
            }
        )

        if iso in cat1_isos and onb_yr >= 2022:
            ed6b_rows.append(
                {
                    "ISO3": iso,
                    "Country": sr["Country_or_Territory"],
                    "First_Any_CAP_Alert": first_any,
                    "First_Flood_CAP_Alert": first_fl,
                    "Onboard_Year": onb_yr,
                    "Events_5yr_WoGS": na,
                    "CAP_Cov_5yr_WoGS": ca,
                    "Cov_5yr_Pct": round(pct_a, 1) if np.isfinite(pct_a) else np.nan,
                    "Pre_Onboard_Events": na - n_post_a,
                    "Post_Onboard_Events": n_post_a,
                    "Post_Onboard_CAP_Cov": c_post_a,
                    "Post_Onboard_Pct": round(pct_post_a, 1) if np.isfinite(pct_post_a) else np.nan,
                    "Diff_pp": round(pct_post_a - pct_a, 1)
                    if (np.isfinite(pct_post_a) and np.isfinite(pct_a))
                    else np.nan,
                    "Events_5yr_WGS": nb,
                    "CAP_Cov_5yr_WGS": cb,
                    "Cov_5yr_Pct_WGS": round(pct_b, 1) if np.isfinite(pct_b) else np.nan,
                    "Pre_Onboard_Events_WGS": nb - n_post_b,
                    "Post_Onboard_Events_WGS": n_post_b,
                    "Post_Onboard_CAP_Cov_WGS": c_post_b,
                    "Post_Onboard_Pct_WGS": round(pct_post_b, 1) if np.isfinite(pct_post_b) else np.nan,
                    "Diff_pp_WGS": round(pct_post_b - pct_b, 1)
                    if (np.isfinite(pct_post_b) and np.isfinite(pct_b))
                    else np.nan,
                }
            )

    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(paths.results_dir / "country_coverage_and_gains_150.csv", index=False)

    c_idx = c_df.set_index("ISO3")
    ed1_rows = []
    for _, sr in supp[supp["Cohort_and_Category"].str.startswith("Cohort D")].sort_values("ISO3").iterrows():
        iso = sr["ISO3"]
        cr = c_idx.loc[iso]
        ed1_rows.append(
            {
                "ISO3": iso,
                "Country_or_Territory": sr["Country_or_Territory"],
                "WMO_Registered_Alerting_Authorities": _clean_auth_name(iso, sr["WMO_Registered_Alerting_Authorities"]),
                "Official_Public_Warning_Portal_and_Non_CAP_Channels": _clean_pipe_list(
                    sr["Official_Public_Warning_Portal_and_Non_CAP_Channels"]
                ),
                "WMO_and_AlertHub_CAP_Feed_Status_and_URL": _clean_pipe_list(
                    sr["WMO_and_AlertHub_CAP_Feed_Status_and_URL"]
                ),
                "Cat_A_Disasters_2021_2025": int(cr["n_events_wo_gs"]),
                "Cat_B_Events_2021_2025": int(cr["n_events_w_gs"]),
            }
        )
    ed1_df = pd.DataFrame(ed1_rows)
    ed1_df.to_csv(paths.results_dir / "ed_table1_cohort_d_35.csv", index=False)

    if ed6b_rows:
        ed6b_df = (
            pd.DataFrame(ed6b_rows)
            .sort_values(["Onboard_Year", "First_Any_CAP_Alert"], ascending=[False, False])
            .reset_index(drop=True)
        )
    else:
        ed6b_df = pd.DataFrame(
            columns=[
                "ISO3",
                "Country",
                "First_Any_CAP_Alert",
                "First_Flood_CAP_Alert",
                "Onboard_Year",
                "Events_5yr_WoGS",
                "CAP_Cov_5yr_WoGS",
                "Cov_5yr_Pct",
                "Pre_Onboard_Events",
                "Post_Onboard_Events",
                "Post_Onboard_CAP_Cov",
                "Post_Onboard_Pct",
                "Diff_pp",
                "Events_5yr_WGS",
                "CAP_Cov_5yr_WGS",
                "Cov_5yr_Pct_WGS",
                "Pre_Onboard_Events_WGS",
                "Post_Onboard_Events_WGS",
                "Post_Onboard_CAP_Cov_WGS",
                "Post_Onboard_Pct_WGS",
                "Diff_pp_WGS",
            ]
        )
    ed6b_df.to_csv(paths.results_dir / "ed_table6b_late_onboarders.csv", index=False)

    ev_a150 = ev_a[ev_a["country_iso3"].isin(eval_150)].copy()
    ev_a150["year"] = pd.to_datetime(ev_a150["start_date"]).dt.year
    ev_b150 = ev_b[ev_b["country_iso3"].isin(eval_150)].copy()
    ev_b150["year"] = pd.to_datetime(ev_b150["start_date"]).dt.year
    c1_onb_yrs = c_df[c_df["ISO3"].isin(cat1_isos)].set_index("ISO3")["onboard_year"].to_dict()
    c2_onb_yrs = c_df[c_df["ISO3"].isin(cat2_isos)].set_index("ISO3")["onboard_year"].to_dict()
    ed6a_rows = []
    cum_cap = 0
    for yr in range(2021, 2026):
        n1_yr = [k for k, v in c1_onb_yrs.items() if (yr == 2021 and 0 < v <= 2021) or (yr > 2021 and v == yr)]
        n2_yr = [k for k, v in c2_onb_yrs.items() if (yr == 2021 and 0 < v <= 2021) or (yr > 2021 and v == yr)]
        cum_cap += len(n1_yr) + len(n2_yr)
        sub_ya = ev_a150[ev_a150["year"] == yr]
        sub_yb = ev_b150[ev_b150["year"] == yr]
        nya, cya = len(sub_ya), int(sub_ya["cov_cap"].sum())
        nyb, cyb = len(sub_yb), int(sub_yb["cov_cap"].sum())
        ed6a_rows.append(
            {
                "Year": yr,
                "New_Cat1_Flood_CAP": len(n1_yr),
                "New_Cat1_ISOs": ",".join(sorted(n1_yr)),
                "New_Cat2_NonFlood_CAP": len(n2_yr),
                "Cumulative_Active_CAP_in_Window": cum_cap,
                "Events_WoGS": nya,
                "CAP_Cov_WoGS": cya,
                "CAP_Cov_Pct_WoGS": round(100.0 * cya / max(1, nya), 1),
                "Events_WGS": nyb,
                "CAP_Cov_WGS": cyb,
                "CAP_Cov_Pct_WGS": round(100.0 * cyb / max(1, nyb), 1),
            }
        )
    ed6a_df = pd.DataFrame(ed6a_rows)
    ed6a_df.to_csv(paths.results_dir / "ed_table6a_annual_onboarding.csv", index=False)

    clustered_df = build_country_clustered_inference_table(
        data=data,
        gid_to_idx=gid_to_idx,
        eval_150=eval_150,
        n_g=n_g,
    )
    clustered_df.to_csv(paths.results_dir / "country_clustered_inference.csv", index=False)

    summary_payload: dict[str, Any] = {
        "n_eval_jurisdictions": len(eval_150),
        "n_cohort_d": len(cohort_d),
        "ffm_max_pre_days": FFM_MAX_PRE_DAYS,
        "cat_a_eval_events": int(len(ev_a150)),
        "cat_b_eval_events": int(len(ev_b150)),
        "cat_a_eval_pop": float(ev_a150["waterfall_pop"].sum()),
        "cat_b_eval_pop": float(ev_b150["waterfall_pop"].sum()),
    }
    # Operational alert-volume & Admin-1 alert-day burden summary (2021-2025)
    t_b0 = date_to_idx("2021-01-01")
    t_b1 = date_to_idx("2025-12-31")
    cap_days_yr = w_cap7[:, t_b0 : t_b1 + 1].sum(axis=1) / 5.0
    hyb_days_yr = w_hyb_daily[:, t_b0 : t_b1 + 1].sum(axis=1) / 5.0
    eval_gids_a = {gid_to_idx[g] for g in ev_a150["gid_1"].unique() if g in gid_to_idx}
    eval_gids_b = {gid_to_idx[g] for g in ev_b150["gid_1"].unique() if g in gid_to_idx}
    burden_info: dict[str, Any] = {}
    for cat_k, gset in (("Without_GS", eval_gids_a), ("With_GS", eval_gids_b)):
        sub_t1 = t1_df[(t1_df["catalog"] == cat_k) & (t1_df["benchmark"] == "2021_2025")]
        r_cap = sub_t1[(sub_t1["system"] == "CAP_Perm") & (sub_t1["thr_label"] == "All")]
        r_hyb = sub_t1[(sub_t1["system"] == "CAP_Perm_U_DHM_Oper_U_FFM050") & (sub_t1["thr_label"] == "RP=2")]
        if not r_cap.empty and not r_hyb.empty and gset:
            idx_arr = np.array(sorted(gset), dtype=np.int32)
            c_ep, c_tp = int(r_cap["n_episodes"].iloc[0]), int(r_cap["tp_ep"].iloc[0])
            h_ep, h_tp = int(r_hyb["n_episodes"].iloc[0]), int(r_hyb["tp_ep"].iloc[0])
            c_unv, h_unv = c_ep - c_tp, h_ep - h_tp
            burden_info[cat_k] = {
                "n_flood_prone_gid1": int(len(idx_arr)),
                "cap_episodes": c_ep,
                "hybrid_episodes": h_ep,
                "episode_change_pct": round(100.0 * (h_ep - c_ep) / max(1, c_ep), 2),
                "cap_unverified_episodes": c_unv,
                "hybrid_unverified_episodes": h_unv,
                "unverified_episode_change_pct": round(100.0 * (h_unv - c_unv) / max(1, c_unv), 2),
                "cap_mean_alert_days_per_yr": round(float(np.mean(cap_days_yr[idx_arr])), 2),
                "cap_median_alert_days_per_yr": round(float(np.median(cap_days_yr[idx_arr])), 2),
                "hybrid_mean_alert_days_per_yr": round(float(np.mean(hyb_days_yr[idx_arr])), 2),
                "hybrid_median_alert_days_per_yr": round(float(np.median(hyb_days_yr[idx_arr])), 2),
            }
    if burden_info:
        summary_payload["alert_burden_2021_2025"] = burden_info
    if not clustered_df.empty:
        hyb_rows = clustered_df[clustered_df["system"] == "CAP_Perm_U_DHM_Oper_U_FFM050"]
        if not hyb_rows.empty:
            summary_payload["country_clustered_hybrid"] = {
                str(r["catalog"]): {
                    "n_countries": int(r["n_countries"]),
                    "n_improved": int(r["n_improved"]),
                    "median_pp_gain": float(r["median_pp_gain"]),
                    "wilcoxon_p": float(r["wilcoxon_p"]),
                    "baseline_deff": float(r["baseline_deff"]),
                    "candidate_deff": float(r["candidate_deff"]),
                    "baseline_ci_pct": [float(r["baseline_ci_low_pct"]), float(r["baseline_ci_high_pct"])],
                    "candidate_ci_pct": [float(r["candidate_ci_low_pct"]), float(r["candidate_ci_high_pct"])],
                    "gap_closed_ci_pct": [float(r["gap_closed_ci_low_pct"]), float(r["gap_closed_ci_high_pct"])],
                }
                for _, r in hyb_rows.iterrows()
            }
    (paths.results_dir / "reproduction_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    figures_written: list[Path] = []
    if render_figures:
        ecdf_p = paths.results_dir / "stream_gauge_nse_ecdf_quantiles.csv"
        iqr_p = paths.results_dir / "stream_gauge_precision_iqr.csv"
        ecdf_df = pd.read_csv(ecdf_p) if ecdf_p.exists() else None
        iqr_df = pd.read_csv(iqr_p) if iqr_p.exists() else None
        figures_written = render_all_publication_figures(
            paths=paths,
            supp=supp,
            c_df=c_df,
            ed6b_df=ed6b_df,
            haz=haz,
            df_pr=df_pr,
            df_fine=df_fine,
            df_leads=df_leads,
            df_single=df_single,
            df_ffm_slices=df_ffm_slices,
            df_aud=df_aud,
            ed4_df=ed4_df,
            t2_df=t2_df,
            ecdf_df=ecdf_df,
            iqr_df=iqr_df,
        )

    print(
        f"Manuscript tables & {len(figures_written)} figure files generated in {time.time() - t0:.1f}s.",
        flush=True,
    )
    return {
        "table1": t1_df,
        "table2": t2_df,
        "nse_summary": nse_summary,
        "ed_table1": ed1_df,
        "ed_table4": ed4_df,
        "ed_table6a": ed6a_df,
        "ed_table6b": ed6b_df,
        "country_summary": c_df,
        "country_clustered_inference": clustered_df,
        "figures": figures_written,
    }


def render_figures_from_results(paths: ProjectPaths) -> list[Path]:
    """Re-render all 11 single-composite publication figures (PDF, SVG, PNG) from summary CSVs."""
    df_pr = pd.read_csv(paths.results_dir / "pr_curves_summary.csv")
    fine_p = paths.results_dir / "fine_ffm_pr_curves.csv"
    if not fine_p.exists():
        resolved_fine = _optional_resolve(paths, "fine_ffm_pr_curves.csv", ("03_summary_results",))
        fine_p = resolved_fine if resolved_fine is not None else fine_p
    df_fine = pd.read_csv(fine_p) if fine_p.exists() else pd.DataFrame()
    df_leads = pd.read_csv(paths.results_dir / "lead_window_sweep.csv")
    df_single = pd.read_csv(paths.results_dir / "single_lead_decay.csv")
    df_ffm_slices = pd.read_csv(paths.results_dir / "ffm_gs_slices_pr.csv")
    df_aud = pd.read_csv(paths.results_dir / "sensitivity_audits.csv")
    c_df = pd.read_csv(paths.results_dir / "country_coverage_and_gains_150.csv")
    ed6b_df = pd.read_csv(paths.results_dir / "ed_table6b_late_onboarders.csv")
    supp = pd.read_csv(paths.supplementary_data_1_csv, keep_default_na=False)

    haz_local = paths.results_dir / "cap_country_hazard_counts_2021_2025.csv"
    if haz_local.exists():
        haz = pd.read_csv(haz_local, keep_default_na=False)
    else:
        haz = pd.read_csv(
            paths.resolve_data_file(
                "cap_country_hazard_counts_2021_2025.csv",
                "03_summary_results",
                "02_admin1_exceedance_matrices",
            ),
            keep_default_na=False,
        )

    ed4_local = paths.results_dir / "ed_table4_validation_summary.csv"
    if ed4_local.exists():
        ed4_df = pd.read_csv(ed4_local)
    else:
        val_csv = _optional_resolve(paths, "validation_n400_deidentified.csv", ("04_validation_deidentified",))
        if val_csv is None:
            val_csv = _optional_resolve(paths, "validation_audit_deidentified_n400.csv", ("04_validation_deidentified",))
        ed4_df = run_validation_audit(deidentified_csv=val_csv)

    t2_local = paths.results_dir / "table2_gauge_vs_catalogs.csv"
    t2_p = t2_local if t2_local.exists() else _optional_resolve(paths, "table2_gauge_vs_catalogs.csv", ("03_summary_results",))
    t2_df = pd.read_csv(t2_p) if t2_p is not None and t2_p.exists() else None

    ecdf_local = paths.results_dir / "stream_gauge_nse_ecdf_quantiles.csv"
    ecdf_p = ecdf_local if ecdf_local.exists() else _optional_resolve(paths, "stream_gauge_nse_ecdf_quantiles.csv", ("03_summary_results",))
    ecdf_df = pd.read_csv(ecdf_p) if ecdf_p is not None and ecdf_p.exists() else None

    iqr_local = paths.results_dir / "stream_gauge_precision_iqr.csv"
    iqr_p = iqr_local if iqr_local.exists() else _optional_resolve(paths, "stream_gauge_precision_iqr.csv", ("03_summary_results",))
    iqr_df = pd.read_csv(iqr_p) if iqr_p is not None and iqr_p.exists() else None

    return render_all_publication_figures(
        paths=paths,
        supp=supp,
        c_df=c_df,
        ed6b_df=ed6b_df,
        haz=haz,
        df_pr=df_pr,
        df_fine=df_fine,
        df_leads=df_leads,
        df_single=df_single,
        df_ffm_slices=df_ffm_slices,
        df_aud=df_aud,
        ed4_df=ed4_df,
        t2_df=t2_df,
        ecdf_df=ecdf_df,
        iqr_df=iqr_df,
    )


def _optional_resolve(paths: ProjectPaths, filename: str, subdirs: tuple[str, ...]) -> Path | None:
    try:
        return paths.resolve_data_file(filename, *subdirs)
    except FileNotFoundError:
        return None
