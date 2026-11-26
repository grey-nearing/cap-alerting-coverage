"""Portable in-situ stream-gauge verification (Table 2 and NSE benchmark summaries)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TABLE2_RETURN_PERIODS: tuple[int, ...] = (2, 5, 10, 20, 25, 50, 100)


def build_shared_gauge_nse_benchmark(
    dhm_gauged_metrics: pd.DataFrame,
    glofas_nse_df: pd.DataFrame,
    geoglows_nse_df: pd.DataFrame,
    min_obs_days: int = 365,
) -> pd.DataFrame:
    """Assemble the area-validated (>=365 overlapping days, <=10% area diff) 3-model NSE table."""
    v_gf = glofas_nse_df[
        (glofas_nse_df["is_valid_10pct"].astype(bool)) & (glofas_nse_df["valid_obs_days"] >= min_obs_days)
    ].copy()
    v_gg = geoglows_nse_df[
        (geoglows_nse_df["is_valid_10pct"].astype(bool)) & (geoglows_nse_df["valid_obs_days"] >= min_obs_days)
    ].copy()

    dhm_cols = [c for c in ("NSE", "NSE_ABOVE_MEAN") if c in dhm_gauged_metrics.columns]
    if "gauge_id" in dhm_gauged_metrics.columns:
        common_gf = v_gf.merge(dhm_gauged_metrics[["gauge_id", *dhm_cols]], on="gauge_id", how="inner")
    else:
        common_gf = v_gf.merge(dhm_gauged_metrics[dhm_cols], left_on="gauge_id", right_index=True, how="inner")

    common_all = common_gf.merge(v_gg[["gauge_id", "nse_geoglows_v2"]], on="gauge_id", how="left")
    return common_all.rename(columns={"NSE": "nse_dhm", "NSE_ABOVE_MEAN": "nse_above_mean_dhm"})


def summarize_stream_gauge_nse(
    dhm_gauged_metrics: pd.DataFrame,
    nse_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ``stream_gauge_nse_summary.csv`` matching manuscript Methods Section 5."""
    common3 = nse_benchmark.dropna(subset=["nse_dhm", "nse_glofas_v5", "nse_geoglows_v2"])
    dhm_valid = dhm_gauged_metrics["NSE"].dropna()
    row = {
        "unfiltered_total_gauges": int(len(dhm_gauged_metrics)),
        "unfiltered_valid_nse": int(len(dhm_valid)),
        "unfiltered_median_nse": round(float(dhm_valid.median()), 3) if len(dhm_valid) else np.nan,
        "unfiltered_pct_pos_nse": round(float((dhm_valid > 0).mean() * 100.0), 1) if len(dhm_valid) else np.nan,
        "area_validated_n_dhm": int(nse_benchmark["nse_dhm"].notna().sum()),
        "area_validated_median_nse_dhm": round(float(nse_benchmark["nse_dhm"].median()), 3),
        "area_validated_pct_pos_dhm": round(float((nse_benchmark["nse_dhm"].dropna() > 0).mean() * 100.0), 1),
        "area_validated_n_glofas": int(nse_benchmark["nse_glofas_v5"].notna().sum()),
        "area_validated_median_nse_glofas": round(float(nse_benchmark["nse_glofas_v5"].median()), 3),
        "area_validated_pct_pos_glofas": round(float((nse_benchmark["nse_glofas_v5"].dropna() > 0).mean() * 100.0), 1),
        "area_validated_n_geoglows": int(nse_benchmark["nse_geoglows_v2"].notna().sum()),
        "area_validated_median_nse_geoglows": round(float(nse_benchmark["nse_geoglows_v2"].dropna().median()), 3),
        "area_validated_pct_pos_geoglows": round(
            float((nse_benchmark["nse_geoglows_v2"].dropna() > 0).mean() * 100.0), 1
        ),
        "common_3model_n": int(len(common3)),
        "common_3model_median_nse_dhm": round(float(common3["nse_dhm"].median()), 3) if len(common3) else np.nan,
        "common_3model_pct_pos_dhm": round(float((common3["nse_dhm"] > 0).mean() * 100.0), 1)
        if len(common3)
        else np.nan,
        "common_3model_median_nse_glofas": round(float(common3["nse_glofas_v5"].median()), 3)
        if len(common3)
        else np.nan,
        "common_3model_pct_pos_glofas": round(float((common3["nse_glofas_v5"] > 0).mean() * 100.0), 1)
        if len(common3)
        else np.nan,
        "common_3model_median_nse_geoglows": round(float(common3["nse_geoglows_v2"].median()), 3)
        if len(common3)
        else np.nan,
        "common_3model_pct_pos_geoglows": round(float((common3["nse_geoglows_v2"] > 0).mean() * 100.0), 1)
        if len(common3)
        else np.nan,
    }
    return pd.DataFrame([row])


def build_table2_gauge_vs_catalogs(
    df_pr: pd.DataFrame,
    dhm_gauged_metrics: pd.DataFrame | None = None,
    precomputed_gauge_precision: pd.DataFrame | None = None,
    return_periods: Sequence[int] = TABLE2_RETURN_PERIODS,
) -> pd.DataFrame:
    """Build ``table2_gauge_vs_catalogs.csv`` comparing gauge precision against Catalog A & B.

    Accepts either raw ``dhm_gauged_metrics`` (with ``return_period_{rp}_window_2_precision``
    columns) or a pre-aggregated ``precomputed_gauge_precision`` table containing
    ``rp_yr``, ``gauge_n``, ``gauge_precision_mean_pct``, and ``gauge_precision_median_pct``
    (e.g., from Tier-1 Zenodo summary inputs).
    """
    if dhm_gauged_metrics is None and precomputed_gauge_precision is None:
        raise ValueError("Provide either dhm_gauged_metrics or precomputed_gauge_precision.")

    pre_idx = precomputed_gauge_precision.set_index("rp_yr") if precomputed_gauge_precision is not None else None
    rows = []
    for rp in return_periods:
        if dhm_gauged_metrics is not None:
            col = f"return_period_{rp}_window_2_precision"
            s = dhm_gauged_metrics[col].dropna() * 100.0
            g_mean = float(s.mean())
            g_med = float(s.median())
            g_n = int(len(s))
        else:
            assert pre_idx is not None
            if rp not in pre_idx.index:
                continue
            gr = pre_idx.loc[rp]
            g_mean = float(gr["gauge_precision_mean_pct"])
            g_med = float(gr["gauge_precision_median_pct"])
            g_n = int(gr["gauge_n"])

        tl = f"RP={rp}"
        d_wo_df = df_pr[
            (df_pr["catalog"] == "Without_GS")
            & (df_pr["benchmark"] == "2021_2025")
            & (df_pr["lead_N"] == 7)
            & (df_pr["system"] == "DHM_Oper")
            & (df_pr["thr_label"] == tl)
        ]
        d_w_df = df_pr[
            (df_pr["catalog"] == "With_GS")
            & (df_pr["benchmark"] == "2021_2025")
            & (df_pr["lead_N"] == 7)
            & (df_pr["system"] == "DHM_Oper")
            & (df_pr["thr_label"] == tl)
        ]
        if d_wo_df.empty or d_w_df.empty:
            continue
        d_wo = d_wo_df.iloc[0]
        d_w = d_w_df.iloc[0]
        rows.append(
            {
                "rp_yr": int(rp),
                "gauge_n": g_n,
                "gauge_precision_mean_pct": round(g_mean, 3),
                "gauge_precision_median_pct": round(g_med, 3),
                "dhm_oper_without_gs_n_episodes": int(d_wo["n_episodes"]),
                "dhm_oper_without_gs_prec_pct": round(float(d_wo["precision_pct"]), 3),
                "dhm_oper_without_gs_rec_pct": round(float(d_wo["recall_pct"]), 3),
                "dhm_oper_with_gs_n_episodes": int(d_w["n_episodes"]),
                "dhm_oper_with_gs_prec_pct": round(float(d_w["precision_pct"]), 3),
                "dhm_oper_with_gs_rec_pct": round(float(d_w["recall_pct"]), 3),
                "diff_gauge_vs_without_gs_pp": round(g_mean - float(d_wo["precision_pct"]), 2),
                "diff_gauge_vs_with_gs_pp": round(g_mean - float(d_w["precision_pct"]), 2),
            }
        )
    return pd.DataFrame(rows)


def run_gauge_verification(
    pr_curves_csv: str | Path,
    results_dir: str | Path,
    gauge_metrics_parquet: str | Path | None = None,
    nse_benchmark_csv: str | Path | None = None,
    reference_table2_csv: str | Path | None = None,
    reference_nse_summary_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Generate Table 2 and stream-gauge NSE summary cleanly in either Tier-1 or Tier-2 mode."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_pr = pd.read_csv(pr_curves_csv)

    gm_df = (
        pd.read_parquet(gauge_metrics_parquet)
        if gauge_metrics_parquet is not None and Path(gauge_metrics_parquet).exists()
        else None
    )
    nse_df = (
        pd.read_csv(nse_benchmark_csv) if nse_benchmark_csv is not None and Path(nse_benchmark_csv).exists() else None
    )
    pre_t2 = (
        pd.read_csv(reference_table2_csv)
        if reference_table2_csv is not None and Path(reference_table2_csv).exists()
        else None
    )

    nse_summary: pd.DataFrame | None = None
    if gm_df is not None and nse_df is not None:
        nse_summary = summarize_stream_gauge_nse(gm_df, nse_df)
        nse_summary.to_csv(out_dir / "stream_gauge_nse_summary.csv", index=False)
    elif reference_nse_summary_csv is not None and Path(reference_nse_summary_csv).exists():
        nse_summary = pd.read_csv(reference_nse_summary_csv)
        nse_summary.to_csv(out_dir / "stream_gauge_nse_summary.csv", index=False)

    t2_df = build_table2_gauge_vs_catalogs(
        df_pr,
        dhm_gauged_metrics=gm_df,
        precomputed_gauge_precision=pre_t2 if gm_df is None else None,
    )
    t2_df.to_csv(out_dir / "table2_gauge_vs_catalogs.csv", index=False)
    return t2_df, nse_summary
