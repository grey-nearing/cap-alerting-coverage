"""Manuscript tables and publication-grade figure generation (Tier 1)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from cap_alerting_coverage.config import D_TOTAL, DATES, LEAD_WINDOWS_DAYS, ProjectPaths, date_to_idx
from cap_alerting_coverage.gauge_verification import run_gauge_verification
from cap_alerting_coverage.scoring import (
    bin_envelope,
    extract_pareto_frontier,
    load_benchmark_data,
    load_universe_sets,
    make_w_matrix,
)
from cap_alerting_coverage.stats_utils import mcnemar_exact_p
from cap_alerting_coverage.validation_audit import run_validation_audit


def setup_publication_style() -> None:
    """Apply clean publication matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_world_gdf(geojson_path: str | Path) -> gpd.GeoDataFrame:
    """Load Natural Earth 1:110m countries GeoJSON and normalize ISO-3 codes + French Guiana."""
    world = gpd.read_file(geojson_path)
    world = world[world["NAME"] != "Antarctica"].copy()
    world["iso3"] = world.apply(lambda r: r["ADM0_A3"] if r["ISO_A3"] == "-99" else r["ISO_A3"], axis=1)
    world.loc[world["NAME"] == "Norway", "iso3"] = "NOR"
    world.loc[world["NAME"] == "France", "iso3"] = "FRA"
    world["iso3"] = world["iso3"].replace({"KOS": "XKX", "SOL": "SOM", "SDS": "SSD", "PSX": "PSE"})

    fr_rows = world[world["iso3"] == "FRA"]
    if not fr_rows.empty:
        fr_idx = fr_rows.index[0]
        fr_geom = fr_rows.loc[fr_idx, "geometry"]
        if fr_geom.geom_type == "MultiPolygon":
            guf_parts = [g for g in fr_geom.geoms if g.centroid.x < -20.0]
            met_parts = [g for g in fr_geom.geoms if g.centroid.x >= -20.0]
            if guf_parts and met_parts:
                world.loc[fr_idx, "geometry"] = MultiPolygon(met_parts) if len(met_parts) > 1 else met_parts[0]
                guf_row = world.loc[fr_idx].copy()
                guf_row["NAME"] = "French Guiana"
                guf_row["ISO_A3"] = "GUF"
                guf_row["ADM0_A3"] = "GUF"
                guf_row["iso3"] = "GUF"
                guf_row["geometry"] = MultiPolygon(guf_parts) if len(guf_parts) > 1 else guf_parts[0]
                world = pd.concat([world, gpd.GeoDataFrame([guf_row], crs=world.crs)], ignore_index=True)
    return world


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


def _tier_label(pct: float) -> str:
    if pct > 80.0:
        return ">80%"
    if pct > 50.0:
        return "51–80%"
    if pct > 20.0:
        return "21–50%"
    if pct > 0.0:
        return "1–20%"
    return "0%"


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


def _draw_conceptual_timeline_panel(ax: plt.Axes) -> None:
    """Figure 3a: Evaluation timeline centered on flood onset."""
    ax.set_xlim(-8.8, 6.1)
    ax.set_ylim(-0.35, 4.48)
    ax.axvspan(-7.0, 0.0, color="#dbeafe", alpha=0.55, zorder=0)
    ax.axvspan(0.0, 3.0, color="#fed7aa", alpha=0.45, zorder=0)
    for xv, col, ls, lw in ((-7.0, "#2563eb", ":", 1.2), (0.0, "#b91c1c", "--", 1.5), (3.0, "#c2410c", ":", 1.2)):
        ax.axvline(xv, color=col, linestyle=ls, linewidth=lw, zorder=1)

    ax.annotate("", xy=(-7.0, 4.18), xytext=(3.0, 4.18), arrowprops=dict(arrowstyle="<->", color="#1e293b", lw=1.2))
    ax.text(
        -2.0,
        4.24,
        "Matching window (7 days before onset through flood end)",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color="#0f172a",
    )
    ax.annotate("", xy=(-7.0, 3.80), xytext=(0.0, 3.80), arrowprops=dict(arrowstyle="<->", color="#1d4ed8", lw=1.1))
    ax.text(
        -3.5, 3.86, "Pre-onset (7 days)", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#1d4ed8"
    )
    ax.text(1.5, 3.86, "Active flood", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#9a3412")

    y1 = 3.05
    ax.axhline(y1, color="#e2e8f0", lw=0.7, zorder=0)
    ax.scatter([-7.9], [y1], s=40, marker="^", color="#94a3b8", edgecolor="#475569", lw=0.7, zorder=4)
    ax.text(-7.9, y1 - 0.24, "Outside window\n(false alarm)", ha="center", va="top", fontsize=6.1, color="#475569")
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-4.55, y1 - 0.16), 3.6, 0.32, boxstyle="round,pad=0.04", fc="#fee2e2", ec="#c1121f", lw=1.0, zorder=2
        )
    )
    ax.scatter([-4.2, -3.2, -1.3], [y1] * 3, s=46, marker="^", color="#c1121f", edgecolor="black", lw=0.55, zorder=5)
    ax.text(
        -2.75,
        y1 + 0.21,
        "Alerts ≤2 quiet days apart → 1 warning episode",
        ha="center",
        va="bottom",
        fontsize=6.7,
        fontweight="bold",
        color="#991b1b",
    )

    y2 = 2.08
    ax.axhline(y2, color="#e2e8f0", lw=0.7, zorder=0)
    xs_sim = np.linspace(-7.5, 4.5, 160)
    ys_sim = y2 - 0.12 + 0.36 * np.exp(-0.5 * ((xs_sim - 1.1) / 1.45) ** 2)
    q2_lvl = y2 + 0.10
    ax.plot(xs_sim, ys_sim, color="#004e89", lw=1.5, zorder=3)
    ax.plot([-7.5, 4.5], [q2_lvl, q2_lvl], color="#004e89", ls="--", lw=0.85, alpha=0.8, zorder=2)
    ax.fill_between(xs_sim, q2_lvl, ys_sim, where=(ys_sim >= q2_lvl), color="#93c5fd", alpha=0.55, zorder=2)
    ax.text(3.2, q2_lvl + 0.04, "Q2yr threshold", fontsize=6.5, fontweight="bold", color="#004e89", va="bottom")
    ax.text(-4.8, y2 + 0.14, "Simulated river peak", fontsize=6.6, color="#1e3a8a", ha="left", va="bottom")

    y3 = 1.12
    ax.axhline(y3, color="#e2e8f0", lw=0.7, zorder=0)
    ax.scatter([-5.0], [y3 + 0.06], s=32, marker="o", color="#0077b6", edgecolor="black", lw=0.55, zorder=5)
    ax.scatter([0.0], [y3 + 0.06], s=44, marker="o", color="#0077b6", edgecolor="black", lw=0.7, zorder=5)
    ax.annotate(
        "", xy=(-0.15, y3 + 0.06), xytext=(-4.85, y3 + 0.06), arrowprops=dict(arrowstyle="->", color="#0077b6", lw=1.5)
    )
    ax.text(
        -2.7,
        y3 + 0.16,
        "Issued 5d ahead → scored on predicted flood date",
        ha="center",
        va="bottom",
        fontsize=6.6,
        fontweight="bold",
        color="#0369a1",
    )
    ax.scatter([3.0], [y3 - 0.12], s=28, marker="o", color="#9ca3af", edgecolor="#4b5563", lw=0.55, zorder=4)
    ax.scatter([5.6], [y3 - 0.12], s=32, marker="x", color="#6b7280", lw=1.2, zorder=4)
    ax.annotate(
        "",
        xy=(5.48, y3 - 0.12),
        xytext=(3.12, y3 - 0.12),
        arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.1, ls="--"),
    )
    ax.text(4.3, y3 - 0.21, "After flood end\n(excluded)", ha="center", va="top", fontsize=6.1, color="#4b5563")

    y4 = 0.18
    ax.axhline(y4, color="#e2e8f0", lw=0.7, zorder=0)
    xs_ffm = np.linspace(-7.5, 4.5, 160)
    ys_ffm = y4 - 0.12 + 0.34 * np.exp(-0.5 * ((xs_ffm + 0.65) / 0.55) ** 2)
    tau_lvl = y4 + 0.08
    ax.plot(xs_ffm, ys_ffm, color="#6a4c93", lw=1.5, zorder=3)
    ax.plot([-7.5, 4.5], [tau_lvl, tau_lvl], color="#6a4c93", ls="--", lw=0.85, alpha=0.8, zorder=2)
    ax.fill_between(xs_ffm, tau_lvl, ys_ffm, where=(ys_ffm >= tau_lvl), color="#d8b4fe", alpha=0.55, zorder=2)
    ax.text(
        -0.65,
        y4 + 0.23,
        "Flash flood probability ≥ 0.50 (0–24 h)",
        ha="center",
        va="bottom",
        fontsize=6.6,
        fontweight="bold",
        color="#581c87",
    )

    ax.set_yticks([y1, y2, y3, y4])
    ax.set_yticklabels(
        ["Official CAP alerts", "0-day river models", "0–7 day forecasts", "0–24 h flash flood model"],
        fontsize=7.4,
        fontweight="bold",
        color="#1e293b",
    )
    ax.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5])
    ax.set_xticklabels(["−7d", "−5d", "−3d", "−1d", "Onset", "+1d", "End", "End+2d"], fontsize=7.2)
    ax.set_xlabel("Days relative to recorded subnational flood onset and end", fontsize=7.6)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.08, 1.03, "a", transform=ax.transAxes, fontsize=11.0, fontweight="bold", va="bottom", ha="left")
    ax.set_title("Evaluation timeline and warning alignment", fontsize=8.4, fontweight="bold", pad=6)


def _plot_pr_panel(
    ax: plt.Axes,
    df_sub: pd.DataFrame,
    title: str,
    panel_letter: str,
    include_geoglows_oper: bool = False,
    is_cat_a: bool = False,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    if include_geoglows_oper:
        specs = [
            ("GeoGLOWS_Oper_EnsMedian", "GeoGLOWS v2 ensemble (0–7 day)", "#e66101", "-", "s", 4.0, 1.8, 5),
            ("GeoGLOWS_Oper_HRES", "GeoGLOWS v2 control (0–7 day)", "#fdb863", "-.", "^", 3.8, 1.5, 4),
            ("DHM_Oper", "Google DHM (0–7 day)", "#0077b6", "-", "o", 4.0, 1.9, 5),
            ("DHM_Oper_U_FFM_Pareto", "Google DHM (0–7 day) + FFM (0–24 h)", "#6a4c93", "-", "*", 6.0, 2.0, 6),
        ]
    else:
        specs = [
            ("GeoGLOWS_Retro_0d", "GeoGLOWS v2 (0-day)", "#d55e00", "--", "s", 3.6, 1.4, 3),
            ("GloFAS_v5_0d", "GloFAS v5 (0-day)", "#8c510a", "--", "h", 3.8, 1.4, 3),
            ("DHM_0d", "Google DHM (0-day)", "#004e89", "--", "o", 3.6, 1.5, 4),
            ("DHM_Oper", "Google DHM (0–7 day)", "#0077b6", "-", "o", 4.0, 1.9, 5),
            ("DHM_Oper_U_FFM_Pareto", "Google DHM (0–7 day) + FFM (0–24 h)", "#6a4c93", "-", "*", 6.0, 2.0, 6),
        ]

    for sys_k, label, col, ls, mk, ms, lw, zo in specs:
        sub = df_sub[df_sub["system"] == sys_k].sort_values("recall_pct")
        if sub.empty:
            continue
        ax.plot(
            sub["recall_pct"],
            sub["precision_pct"],
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=ms,
            linewidth=lw,
            label=label,
            zorder=zo,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        if sys_k == "DHM_Oper_U_FFM_Pareto":
            pt = df_sub[(df_sub["system"] == "DHM_Oper_U_FFM050") & (df_sub["thr_label"] == "RP=2")]
        else:
            pt = sub[sub["thr_label"].isin(["RP=2", "tau=0.50"])]
        if not pt.empty and np.isfinite(pt["precision_pct"].iloc[0]):
            ax.scatter(
                [float(pt["recall_pct"].iloc[0])],
                [float(pt["precision_pct"].iloc[0])],
                s=32,
                color=col,
                edgecolor="black",
                linewidth=0.8,
                zorder=zo + 2,
            )

    for cap_k, cap_lbl, col, mk, zo in (
        ("CAP_Perm", "Official CAP (flood & storm)", "#c1121f", "^", 7),
        ("CAP_Rest", "Official CAP (flood-coded only)", "#780000", "v", 7),
    ):
        csub = df_sub[df_sub["system"] == cap_k].copy()
        if not csub.empty:
            csub = csub.sort_values("recall_pct")
            ax.plot(
                csub["recall_pct"],
                csub["precision_pct"],
                color=col,
                linestyle="-",
                marker=mk,
                markersize=5.2,
                linewidth=1.8,
                label=cap_lbl,
                zorder=zo + 1,
                markeredgecolor="black",
                markeredgewidth=0.55,
            )
            all_pt = csub[csub["thr_label"] == "All"]
            if not all_pt.empty and np.isfinite(all_pt["precision_pct"].iloc[0]):
                rx, py = float(all_pt["recall_pct"].iloc[0]), float(all_pt["precision_pct"].iloc[0])
                tag = "Flood & storm CAP" if cap_k == "CAP_Perm" else "Flood-coded CAP"
                if is_cat_a:
                    tx = (rx + 5.0) if cap_k == "CAP_Perm" else (rx + 7.5)
                    ty = 15.8 if cap_k == "CAP_Perm" else 21.2
                else:
                    tx = (rx + 8.0) if cap_k == "CAP_Perm" else 3.0
                    ty = 9.5 if cap_k == "CAP_Perm" else 10.5
                ax.annotate(
                    f"{tag}\n({rx:.1f}%, {py:.1f}%)",
                    xy=(rx, py),
                    xytext=(min(tx, 72.0), ty),
                    textcoords="data",
                    fontsize=6.5,
                    fontweight="bold",
                    color=col,
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, alpha=0.94, lw=0.65),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.75, alpha=0.85),
                    zorder=9,
                )

    ax.text(
        -0.14 if show_ylabel else -0.10,
        1.03,
        panel_letter,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=5)
    ax.set_xlim(0, 100)
    if is_cat_a:
        ax.set_ylim(0, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
    else:
        ax.set_ylim(0, 75)
        ax.set_yticks([0, 15, 30, 45, 60, 75])
    ax.tick_params(labelsize=7.0)
    if show_xlabel:
        ax.set_xlabel("Flood coverage / recall (%)", fontsize=7.6)
    if show_ylabel:
        ax.set_ylabel("Warning precision (%)", fontsize=7.6)


def render_pr_and_lead_figures(
    df_pr: pd.DataFrame,
    df_leads: pd.DataFrame,
    df_single: pd.DataFrame,
    df_ffm_slices: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Render Figure 3, Extended Data Figure 2, Extended Data Figure 3, and Extended Data Figure 4."""
    setup_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- Figure 3: 5-panel timeline + PR curves ---
    panels_fig3 = [
        (1, 0, "Without_GS", "2021_2025", "b", "Catalog A: Disaster catalog (2021–2025)", False, True),
        (1, 1, "With_GS", "2021_2025", "c", "Catalog B: With Groundsource (2021–2025)", False, False),
        (2, 0, "Without_GS", "2024_2025", "d", "Catalog A: Overlap (Apr 2024–Dec 2025)", True, True),
        (2, 1, "With_GS", "2024_2025", "e", "Catalog B: Overlap (Apr 2024–Dec 2025)", True, False),
    ]
    fig3_path = figures_dir / "fig-3.png"
    fig = plt.figure(figsize=(7.2, 7.35), facecolor="white")
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.95, 1.18, 1.18], hspace=0.44, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    _draw_conceptual_timeline_panel(ax_a)

    ax_grid = {}
    for r_i, c_i, cat_k, bm_k, let, ttl, inc_g, is_a in panels_fig3:
        ax = fig.add_subplot(gs[r_i, c_i])
        ax_grid[(r_i, c_i)] = ax
        sub = df_pr[(df_pr["catalog"] == cat_k) & (df_pr["benchmark"] == bm_k) & (df_pr["lead_N"] == 7)]
        n_ev = int(sub["n_events"].iloc[0]) if not sub.empty else 0
        _plot_pr_panel(
            ax,
            sub,
            f"{ttl}\nN = {n_ev:,} events",
            let,
            include_geoglows_oper=inc_g,
            is_cat_a=is_a,
            show_ylabel=(c_i == 0),
            show_xlabel=(r_i == 2),
        )

    seen_hl: dict[str, Any] = {}
    for ax_src in (ax_grid[(1, 0)], ax_grid[(2, 0)]):
        hs, ls = ax_src.get_legend_handles_labels()
        for h, l_txt in zip(hs, ls):
            if l_txt not in seen_hl:
                seen_hl[l_txt] = h
    want_leg = [
        "Official CAP (flood & storm)",
        "Official CAP (flood-coded only)",
        "Google DHM (0–7 day)",
        "Google DHM (0–7 day) + FFM (0–24 h)",
        "GeoGLOWS v2 ensemble (0–7 day)",
        "GeoGLOWS v2 control (0–7 day)",
        "Google DHM (0-day)",
        "GeoGLOWS v2 (0-day)",
        "GloFAS v5 (0-day)",
    ]
    leg_h = [seen_hl[k] for k in want_leg if k in seen_hl]
    leg_l = [k for k in want_leg if k in seen_hl]
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.005),
            frameon=True,
            facecolor="#fcfcfc",
            edgecolor="#cccccc",
            columnspacing=0.9,
            handlelength=1.8,
            fontsize=6.4,
        )
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.125, top=0.968, hspace=0.44, wspace=0.22)
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    written.append(fig3_path)

    # --- Extended Data Figure 2: CAP Alone vs CAP + Global Flood Models (2x2) ---
    df_base7 = df_pr[df_pr["lead_N"] == 7]
    df_all_pr = pd.concat(
        [
            df_base7,
            df_ffm_slices[df_ffm_slices["system"].isin(["CAP_Perm_U_DHM_Oper_U_FFM_Grid", "DHM_Oper_U_FFM_Grid"])],
        ],
        ignore_index=True,
    )

    def _n_for(cname: str, bname: str) -> int:
        s = df_ffm_slices[
            (df_ffm_slices["catalog"] == cname)
            & (df_ffm_slices["benchmark"] == bname)
            & (df_ffm_slices["system"] == "CAP_Perm")
        ]
        return int(s["n_events"].iloc[0]) if not s.empty else 0

    n_a21, n_b21, n_gs21 = (
        _n_for("Without_GS", "2021_2025"),
        _n_for("With_GS", "2021_2025"),
        _n_for("GS_Media_Only", "2021_2025"),
    )
    n_a24, n_b24, n_gs24 = (
        _n_for("Without_GS", "2024_2025"),
        _n_for("With_GS", "2024_2025"),
        _n_for("GS_Media_Only", "2024_2025"),
    )

    ed2_path = figures_dir / "ed-fig-2.png"
    panels_ed2 = [
        (
            "Without_GS",
            "2021_2025",
            "a",
            f"Catalog A: Without Groundsource (2021–2025)\n(7-day window; N = {n_a21:,} events)",
        ),
        (
            "With_GS",
            "2021_2025",
            "b",
            f"Catalog B: With Groundsource (2021–2025)\n(7-day window; N = {n_b21:,} events)",
        ),
        (
            "Without_GS",
            "2024_2025",
            "c",
            f"Catalog A: Operational overlap (Apr 2024–Dec 2025)\n(7-day window; N = {n_a24:,} events)",
        ),
        (
            "With_GS",
            "2024_2025",
            "d",
            f"Catalog B: Operational overlap (Apr 2024–Dec 2025)\n(7-day window; N = {n_b24:,} events)",
        ),
    ]
    union_specs = [
        ("CAP_Perm_U_GloFAS_v5_0d", "CAP + GloFAS v5 (0-day)", "#8c510a", "--o", 4.0),
        ("CAP_Perm_U_GeoGLOWS_Retro_0d", "CAP + GeoGLOWS v2 (0-day)", "#e66101", "--s", 4.0),
        ("CAP_Perm_U_GeoGLOWS_Oper_EnsMedian", "CAP + GeoGLOWS v2 ensemble (0–7 day)", "#f1a340", "-s", 4.2),
        ("CAP_Perm_U_GeoGLOWS_Oper_HRES", "CAP + GeoGLOWS v2 control (0–7 day)", "#fdb863", ":^", 4.2),
        ("CAP_Perm_U_DHM_0d", "CAP + Google DHM (0-day)", "#1b7837", "--o", 4.0),
        ("CAP_Perm_U_DHM_Oper", "CAP + Google DHM (0–7 day)", "#0072b2", "-o", 4.5),
        ("CAP_Perm_U_FFM", "CAP + Google FFM (0–24 h, τ = 0.95–0.05)", "#009e73", "--D", 4.4),
    ]
    sev_markers = {"Extreme": ("v", 6.5), ">=Severe": ("^", 6.5), ">=Moderate": ("^", 6.5), "All": ("^", 7.5)}
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.6), dpi=300, sharex=True, sharey=True)
    for ax, (cname, bname, pletter, ptitle) in zip(axes.flat, panels_ed2):
        sub = df_all_pr[(df_all_pr["catalog"] == cname) & (df_all_pr["benchmark"] == bname)]
        for sys_cap, col, lbl_c in (
            ("CAP_Rest", "#7f0000", "Official CAP (flood-coded only)"),
            ("CAP_Perm", "#d62728", "Official CAP (flood & storm)"),
        ):
            sa_df = sub[sub["system"] == sys_cap].sort_values("recall_pct")
            if sa_df.empty:
                continue
            ax.plot(sa_df["recall_pct"], sa_df["precision_pct"], color=col, lw=2.0, ls="-", label=lbl_c, zorder=4)
            for _, sr in sa_df.iterrows():
                mk, sz = sev_markers.get(str(sr["thr_label"]), ("^", 6.0))
                ax.plot(
                    sr["recall_pct"], sr["precision_pct"], marker=mk, ms=sz, color=col, mec="black", mew=0.6, zorder=5
                )
        for sys_u, lbl_u, col_u, ls_u, ms_u in union_specs:
            su = sub[sub["system"] == sys_u].sort_values("recall_pct")
            if su.empty:
                continue
            ax.plot(
                su["recall_pct"],
                su["precision_pct"],
                ls_u,
                color=col_u,
                lw=1.7,
                ms=ms_u,
                alpha=0.92,
                label=lbl_u,
                zorder=4,
            )
            anch_lbl = "tau=0.50" if sys_u == "CAP_Perm_U_FFM" else "RP=2"
            sa_row = su[su["thr_label"] == anch_lbl]
            if not sa_row.empty:
                ax.plot(
                    sa_row["recall_pct"],
                    sa_row["precision_pct"],
                    marker="D" if sys_u == "CAP_Perm_U_FFM" else "o",
                    ms=7.0,
                    color=col_u,
                    mec="black",
                    mew=1.0,
                    zorder=6,
                )
        s_grid = sub[sub["system"] == "CAP_Perm_U_DHM_Oper_U_FFM_Grid"]
        if not s_grid.empty:
            s_bin = bin_envelope(s_grid, 2.5)
            ax.plot(
                s_bin["recall_pct"],
                s_bin["precision_pct"],
                "-*",
                color="#6a3d9a",
                lw=2.2,
                ms=5.5,
                label="CAP + Google (DHM 0–7 day + FFM 0–24 h) frontier",
                zorder=7,
            )
        s_canon = sub[(sub["system"] == "CAP_Perm_U_DHM_Oper_U_FFM050") & (sub["thr_label"] == "RP=2")]
        if not s_canon.empty:
            ax.plot(
                s_canon["recall_pct"],
                s_canon["precision_pct"],
                "o",
                color="#6a3d9a",
                ms=8.0,
                mec="black",
                mew=1.1,
                zorder=8,
            )
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, ls=":", lw=0.6, alpha=0.55)
        ax.set_title(ptitle, fontsize=10.8, pad=8)
        ax.text(-0.08, 1.05, pletter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")
        ax.set_xlabel("Flood coverage / recall (%)", fontsize=10.5)
        ax.set_ylabel("Warning precision (%)", fontsize=10.5)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    hl_map = dict(zip(labels, handles))
    order_want = [
        "Official CAP (flood & storm)",
        "Official CAP (flood-coded only)",
        "CAP + GloFAS v5 (0-day)",
        "CAP + GeoGLOWS v2 (0-day)",
        "CAP + GeoGLOWS v2 ensemble (0–7 day)",
        "CAP + GeoGLOWS v2 control (0–7 day)",
        "CAP + Google DHM (0-day)",
        "CAP + Google DHM (0–7 day)",
        "CAP + Google FFM (0–24 h, τ = 0.95–0.05)",
        "CAP + Google (DHM 0–7 day + FFM 0–24 h) frontier",
    ]
    f_h = [hl_map[k] for k in order_want if k in hl_map]
    f_l = [k for k in order_want if k in hl_map]
    if f_h:
        fig.legend(
            f_h,
            f_l,
            loc="lower center",
            ncol=3,
            frameon=True,
            facecolor="#fafafa",
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=9.2,
        )
    plt.tight_layout(rect=[0.01, 0.11, 0.99, 0.98], h_pad=2.2, w_pad=2.0)
    fig.savefig(ed2_path, dpi=300)
    plt.close(fig)
    written.append(ed2_path)

    # --- Extended Data Figure 3: Lead-window scaling & single-lead decay ---
    ed3_path = figures_dir / "ed-fig-3.png"
    lead_styles = {
        "All CAP (Permissive)": ("#c1121f", "^", "Official CAP (flood & storm)"),
        "DHM 0d (RP=2)": ("#004e89", "o", "Google DHM 0-day (Q2yr)"),
        "DHM Oper (RP=2)": ("#0077b6", "o", "Google DHM 0–7 day (Q2yr)"),
        "Combined DHM(RP=2) U FFM(0.5)": ("#6a4c93", "*", "Google DHM (Q2yr) + FFM (τ = 0.50)"),
        "GeoGLOWS Oper EnsMedian (RP=2)": ("#e66101", "s", "GeoGLOWS v2 ensemble (Q2yr)"),
        "GeoGLOWS Oper HRES (RP=2)": ("#fdb863", "^", "GeoGLOWS v2 control (Q2yr)"),
        "GloFAS v5 0d (RP=2)": ("#8c510a", "h", "GloFAS v5 0-day (Q2yr)"),
    }
    dec_styles = [
        ("DHM_10D", "Google DHM (single lead 0–10 d, Q2yr)", "#0077b6", "o"),
        ("GeoGLOWS_EnsMedian", "GeoGLOWS v2 ensemble (single lead 0–14 d, Q2yr)", "#e66101", "s"),
        ("GeoGLOWS_HRES", "GeoGLOWS v2 control (single lead 0–9 d, Q2yr)", "#fdb863", "^"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.6))
    for col_i, cat_k, ttl_top, let in (
        (0, "Without_GS", "Lead-window scaling: Catalog A (Apr 2024–Dec 2025)", "a"),
        (1, "With_GS", "Lead-window scaling: Catalog B (Apr 2024–Dec 2025)", "b"),
    ):
        ax = axes[0, col_i]
        sub = df_leads[(df_leads["catalog"] == cat_k) & (df_leads["benchmark"] == "2024_2025")]
        for op_lbl, (col, mk, disp_lbl) in lead_styles.items():
            ss = sub[sub["op_label"] == op_lbl].sort_values("lead_N")
            if ss.empty:
                continue
            ms = 6.5 if mk == "*" else 4.6
            ax.plot(
                ss["lead_N"],
                ss["recall_pct"],
                color=col,
                linestyle="-",
                marker=mk,
                markersize=ms,
                linewidth=1.9,
                label=disp_lbl,
            )
            ax.plot(
                ss["lead_N"],
                ss["precision_pct"],
                color=col,
                linestyle="--",
                marker=mk,
                markersize=ms * 0.85,
                linewidth=1.45,
                alpha=0.85,
                label="_nolegend_",
            )
        ax.plot([], [], color="#333333", linestyle="-", linewidth=1.9, label="Solid line: Flood recall (%)")
        ax.plot([], [], color="#333333", linestyle="--", linewidth=1.45, label="Dashed line: Warning precision (%)")
        ax.text(-0.09, 1.03, let, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
        ax.set_title(ttl_top, fontsize=10.0, pad=7)
        ax.set_xlabel("Lead window before flood onset (days through flood end)")
        ax.set_ylabel("Recall (solid) / Precision (dashed) (%)")
        ax.set_xticks(list(LEAD_WINDOWS_DAYS))
        ax.set_ylim(0, 100)

    for col_i, cat_k, ttl_c, let in (
        (0, "Without_GS", "Single-lead skill decay: Catalog A (Apr 2024–Dec 2025, Q2yr)", "c"),
        (1, "With_GS", "Single-lead skill decay: Catalog B (Apr 2024–Dec 2025, Q2yr)", "d"),
    ):
        ax = axes[1, col_i]
        sub = df_single[
            (df_single["catalog"] == cat_k) & (df_single["benchmark"] == "2024_2025") & (df_single["rp_tier"] == 2)
        ]
        for s_k, lbl, col, mk in dec_styles:
            ss = sub[sub["system"] == s_k].sort_values("lead_day")
            if ss.empty:
                continue
            ax.plot(
                ss["lead_day"],
                ss["recall_pct"],
                color=col,
                linestyle="-",
                marker=mk,
                markersize=4.8,
                linewidth=2.0,
                label=f"{lbl} — Recall",
            )
            ax.plot(
                ss["lead_day"],
                ss["precision_pct"],
                color=col,
                linestyle="--",
                marker=mk,
                markersize=4.2,
                linewidth=1.6,
                label=f"{lbl} — Precision",
            )
        ax.text(-0.09, 1.03, let, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
        ax.set_title(ttl_c, fontsize=10.0, pad=7)
        ax.set_xlabel("Forecast lead time prior to flood date (days)")
        ax.set_ylabel("Single-lead recall / precision (%)")
        ax.set_xticks(range(0, 15, 2))
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white")

    h_top, l_top = axes[0, 0].get_legend_handles_labels()
    if h_top:
        fig.legend(
            h_top,
            l_top,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.992),
            frameon=True,
            facecolor="#fcfcfc",
            edgecolor="#cccccc",
            fontsize=8.0,
        )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.91], h_pad=2.2, w_pad=1.6)
    fig.savefig(ed3_path, dpi=300)
    plt.close(fig)
    written.append(ed3_path)

    # --- Extended Data Figure 4: Standalone FFM across Groundsource slices (2x3) ---
    ed4_path = figures_dir / "ed-fig-4.png"
    panels_ed4 = [
        ("Without_GS", "2021_2025", "a", f"Catalog A: Without Groundsource (2021–2025)\n(N = {n_a21:,} disasters)"),
        ("With_GS", "2021_2025", "b", f"Catalog B: With Groundsource (2021–2025)\n(N = {n_b21:,} events)"),
        (
            "GS_Media_Only",
            "2021_2025",
            "c",
            f"Groundsource media-only clusters (2021–2025)\n(N = {n_gs21:,} unmatched clusters)",
        ),
        (
            "Without_GS",
            "2024_2025",
            "d",
            f"Catalog A: Operational overlap (Apr 2024–Dec 2025)\n(N = {n_a24:,} disasters)",
        ),
        ("With_GS", "2024_2025", "e", f"Catalog B: Operational overlap (Apr 2024–Dec 2025)\n(N = {n_b24:,} events)"),
        (
            "GS_Media_Only",
            "2024_2025",
            "f",
            f"Groundsource media-only overlap (Apr 2024–Dec 2025)\n(N = {n_gs24:,} clusters)",
        ),
    ]
    fig4, axes4 = plt.subplots(2, 3, figsize=(16.5, 9.8), dpi=300, sharex=True, sharey=True)
    for ax, (cname, bname, pletter, ptitle) in zip(axes4.flat, panels_ed4):
        sub = df_ffm_slices[(df_ffm_slices["catalog"] == cname) & (df_ffm_slices["benchmark"] == bname)]
        dhm = sub[sub["system"] == "DHM_Oper"].sort_values("recall_pct")
        ffm = sub[sub["system"] == "FFM"].sort_values("recall_pct")
        cuffm = sub[sub["system"] == "CAP_Perm_U_FFM"].sort_values("recall_pct")
        grid_recs = sub[sub["system"] == "DHM_Oper_U_FFM_Grid"].to_dict("records")
        par_df = pd.DataFrame(extract_pareto_frontier(grid_recs))
        fron = bin_envelope(par_df, 2.5) if not par_df.empty else par_df

        for sys_cap, col, lbl_c in (
            ("CAP_Rest", "#7f0000", "Official CAP (flood-coded only)"),
            ("CAP_Perm", "#d62728", "Official CAP (flood & storm)"),
        ):
            sa_df = sub[sub["system"] == sys_cap].sort_values("recall_pct")
            if not sa_df.empty:
                ax.plot(sa_df["recall_pct"], sa_df["precision_pct"], color=col, lw=1.9, ls="-", label=lbl_c, zorder=3)
        if not dhm.empty:
            ax.plot(
                dhm["recall_pct"],
                dhm["precision_pct"],
                "-o",
                color="#0072b2",
                lw=2.0,
                ms=4.2,
                label="Google DHM (0–7 day)",
                zorder=4,
            )
        if not ffm.empty:
            ax.plot(
                ffm["recall_pct"],
                ffm["precision_pct"],
                "-s",
                color="#d55e00",
                lw=2.2,
                ms=4.5,
                label="Google FFM standalone (0–24 h, τ = 0.95–0.05)",
                zorder=5,
            )
        if not cuffm.empty:
            ax.plot(
                cuffm["recall_pct"],
                cuffm["precision_pct"],
                "--D",
                color="#009e73",
                lw=2.0,
                ms=4.2,
                label="CAP + Google FFM (0–24 h, τ = 0.95–0.05)",
                zorder=5,
            )
        if not fron.empty:
            ax.plot(
                fron["recall_pct"],
                fron["precision_pct"],
                "-*",
                color="#6a3d9a",
                lw=2.2,
                ms=5.5,
                label="Google DHM (0–7 day) + FFM (0–24 h) frontier",
                zorder=6,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, ls=":", lw=0.6, alpha=0.55)
        ax.set_title(ptitle, fontsize=10.5, pad=7)
        ax.text(-0.07, 1.04, pletter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")
        if bname == "2024_2025":
            ax.set_xlabel("Flood coverage / recall (%)", fontsize=10.5)
        if cname == "Without_GS":
            ax.set_ylabel("Warning precision (%)", fontsize=10.5)

    h4, l4 = axes4[0, 0].get_legend_handles_labels()
    if h4:
        fig4.legend(
            h4,
            l4,
            loc="lower center",
            ncol=3,
            frameon=True,
            facecolor="#fafafa",
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=9.4,
        )
    plt.tight_layout(rect=[0.01, 0.09, 0.99, 0.98], h_pad=2.2, w_pad=1.8)
    fig4.savefig(ed4_path, dpi=300)
    plt.close(fig4)
    written.append(ed4_path)
    return written


def render_maps_and_validation_figures(
    paths: ProjectPaths,
    supp: pd.DataFrame,
    c_df: pd.DataFrame,
    ed6b_df: pd.DataFrame,
    haz: pd.DataFrame,
    df_aud: pd.DataFrame,
    t1_df: pd.DataFrame,
    ed4_df: pd.DataFrame,
) -> list[Path]:
    """Render Figures 1a, 1b, 2a–2c, 4, Extended Data Fig. 1, and Extended Data Fig. 5."""
    setup_publication_style()
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    world = load_world_gdf(paths.countries_geojson)

    supp_clean = supp.copy()
    supp_clean["ISO3"] = supp_clean["ISO3"].str.strip()
    cat1_isos = set(supp_clean[supp_clean["Cohort_and_Category"].str.startswith("Cohort A")]["ISO3"])
    cat2_isos = set(supp_clean[supp_clean["Cohort_and_Category"].str.startswith("Cohort B")]["ISO3"])
    cat3_isos = set(supp_clean[supp_clean["Cohort_and_Category"].str.startswith("Cohort C")]["ISO3"])
    cohort_d = set(supp_clean[supp_clean["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"])
    eval_150 = cat1_isos | cat2_isos | cat3_isos

    # --- Fig 1a: Global CAP Capability Categories ---
    lbl_c1 = f"Category 1: Active flood CAP alerts ({len(cat1_isos)} countries)"
    lbl_c2 = f"Category 2: Other CAP alerts only ({len(cat2_isos)} countries)"
    lbl_c3 = f"Category 3: No CAP alerts ({len(cat3_isos)} countries)"
    lbl_cd = f"Excluded non-CAP warning countries ({len(cohort_d)} countries)"
    lbl_no = "Outside study set / no recorded floods"
    col_1a = {lbl_c1: "#1b9e77", lbl_c2: "#e6ab02", lbl_c3: "#d95f02", lbl_cd: "#a6a6a6", lbl_no: "#ededed"}

    def _cls_1a(iso: str) -> str:
        if iso in cohort_d:
            return lbl_cd
        if iso in cat1_isos:
            return lbl_c1
        if iso in cat2_isos:
            return lbl_c2
        if iso in cat3_isos:
            return lbl_c3
        return lbl_no

    world["cls_1a"] = world["iso3"].map(_cls_1a)
    fig1a_p = paths.figures_dir / "fig-1a.png"
    fig, ax = plt.subplots(figsize=(15.0, 8.6), facecolor="white", dpi=300)
    world.plot(ax=ax, color=world["cls_1a"].map(col_1a), edgecolor="#444444", linewidth=0.35)
    cd_w = world[world["iso3"].isin(cohort_d)]
    if not cd_w.empty:
        cd_w.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=0.45, hatch="///")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    handles_1a = [
        mpatches.Patch(facecolor=col_1a[lbl_c1], edgecolor="#444444", label=lbl_c1),
        mpatches.Patch(facecolor=col_1a[lbl_c2], edgecolor="#444444", label=lbl_c2),
        mpatches.Patch(facecolor=col_1a[lbl_c3], edgecolor="#444444", label=lbl_c3),
        mpatches.Patch(facecolor=col_1a[lbl_cd], edgecolor="#555555", hatch="///", label=lbl_cd),
        mpatches.Patch(facecolor=col_1a[lbl_no], edgecolor="#bbbbbb", label=lbl_no),
    ]
    ax.legend(
        handles=handles_1a,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=10.2,
    )
    ax.text(-176, 82, "a", fontsize=22, fontweight="bold", ha="left", va="top")
    plt.tight_layout()
    fig.savefig(fig1a_p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(fig1a_p)

    # --- Fig 1b: Hazard Capability Bar Chart ---
    haz150 = haz[haz["country_iso3"].isin(eval_150)].copy()
    haz_bars = sorted(
        [
            ("Storms &\nHeavy Rain", int((haz150["n_storms"] > 0).sum())),
            ("Floods\n(Category 1)", len(cat1_isos)),
            ("Extreme\nCold & Snow", int((haz150["n_cold"] > 0).sum())),
            ("Extreme\nHeat", int((haz150["n_heat"] > 0).sum())),
            ("Wildfires", int((haz150["n_wildfire"] > 0).sum())),
            ("Earthquakes &\nTsunami", int((haz150["n_earthquake"] > 0).sum())),
            ("Volcanic\nActivity", int((haz150["n_volcano"] > 0).sum())),
        ],
        key=lambda x: -x[1],
    )
    fig1b_p = paths.figures_dir / "fig-1b.png"
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=300, facecolor="white")
    labels_h = [x[0] for x in haz_bars]
    vals_h = [x[1] for x in haz_bars]
    cols_h = ["#1b9e77" if "Floods" in lb else "#2b6cb0" for lb in labels_h]
    bars = ax.bar(labels_h, vals_h, color=cols_h, edgecolor="#222222", linewidth=0.8, width=0.62, zorder=3)
    for bar, cnt in zip(bars, vals_h):
        pct = 100.0 * cnt / max(1, len(eval_150))
        ax.annotate(
            f"{cnt}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )
    ax.set_ylabel(f"Number of countries (of {len(eval_150)} in CAP analysis)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(max(vals_h), 1) * 1.18 + 8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.text(-0.08, 1.03, "b", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
    plt.tight_layout()
    fig.savefig(fig1b_p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(fig1b_p)

    # --- Fig 2a, 2b, 2c + Composite ---
    tier_col = {
        ">80%": "#1a9850",
        "51–80%": "#91cf60",
        "21–50%": "#fee08b",
        "1–20%": "#fc8d59",
        "0%": "#d73027",
        "Excluded non-CAP": "#b5b5b5",
        "No floods / no reported pop.": "#ededed",
    }
    tier_ord = [">80%", "51–80%", "21–50%", "1–20%", "0%", "Excluded non-CAP", "No floods / no reported pop."]
    c_map_dict = c_df.set_index("ISO3").to_dict("index")
    late_cat1_isos = set(ed6b_df["ISO3"]) if not ed6b_df.empty else set()

    def _map_tier(iso: str, n_col: str, pct_col: str) -> str:
        if iso in cohort_d:
            return "Excluded non-CAP"
        info = c_map_dict.get(iso)
        if not info or info[n_col] == 0 or not np.isfinite(info[pct_col]):
            return "No floods / no reported pop."
        return _tier_label(float(info[pct_col]))

    world["t_2a"] = world["iso3"].map(lambda x: _map_tier(x, "n_events_wo_gs", "cap_pct_wo_gs"))
    world["t_2b"] = world["iso3"].map(lambda x: _map_tier(x, "pop_tot_wo_gs", "pop_pct_wo_gs"))
    world["t_2c"] = world["iso3"].map(lambda x: _map_tier(x, "post_onb_n_wo_gs", "post_onb_pct_wo_gs"))

    handles_2 = [
        mpatches.Patch(
            facecolor=tier_col[t], edgecolor="#444444", hatch="..." if t == "Excluded non-CAP" else None, label=t
        )
        for t in tier_ord
    ] + [
        mpatches.Patch(facecolor="white", edgecolor="#111111", hatch="///", label="Cat. 1 late onboarders (2022–2025)")
    ]

    def _plot_fig2_panel(ax: plt.Axes, col_tier: str, let: str, title_txt: str, hatch_late: bool = False) -> None:
        world.plot(ax=ax, color=world[col_tier].map(tier_col), edgecolor="#444444", linewidth=0.35)
        cd_sub = world[world["iso3"].isin(cohort_d)]
        if not cd_sub.empty:
            cd_sub.plot(ax=ax, facecolor="none", edgecolor="#666666", linewidth=0.4, hatch="...")
        if hatch_late and late_cat1_isos:
            lt_sub = world[world["iso3"].isin(late_cat1_isos)]
            if not lt_sub.empty:
                lt_sub.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=0.8, hatch="///")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.axis("off")
        ax.text(-176, 82, let, fontsize=18, fontweight="bold", ha="left", va="top")
        if title_txt:
            ax.set_title(title_txt, fontsize=10.8, fontweight="bold", pad=6)

    for col_k, let_k, fn_k, hatch_k in (
        ("t_2a", "a", "fig-2a.png", False),
        ("t_2b", "b", "fig-2b.png", False),
        ("t_2c", "c", "fig-2c.png", True),
    ):
        p_out = paths.figures_dir / fn_k
        fig, ax = plt.subplots(figsize=(14.8, 8.2), facecolor="white", dpi=300)
        _plot_fig2_panel(ax, col_k, let_k, "", hatch_late=hatch_k)
        if hatch_k:
            ax.legend(
                handles=handles_2,
                loc="lower left",
                ncol=2,
                frameon=True,
                facecolor="white",
                framealpha=0.95,
                edgecolor="#cccccc",
                fontsize=9.0,
            )
        plt.tight_layout()
        fig.savefig(p_out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(p_out)

    fig2_comp = paths.figures_dir / "fig-2_composite.png"
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 17.5), facecolor="white", dpi=300)
    _plot_fig2_panel(
        axes[0], "t_2a", "a", "5-year CAP event coverage (Catalog A: without Groundsource, 2021–2025)", False
    )
    _plot_fig2_panel(axes[1], "t_2b", "b", "5-year CAP affected-population coverage (2021–2025)", False)
    _plot_fig2_panel(axes[2], "t_2c", "c", "Post-onboarding event coverage (from first CAP alert onward)", True)
    fig.legend(
        handles=handles_2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9.2,
    )
    plt.tight_layout(rect=[0.01, 0.055, 0.99, 0.99], h_pad=1.6)
    fig.savefig(fig2_comp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(fig2_comp)

    # --- Fig 4: Hybrid +pp Gains World Map ---
    bins_4 = [-0.01, 0.05, 10.0, 20.0, 35.0, 50.0, 60.0, 105.0]
    lbls_4 = ["No increase (0 pp)", "+0.1–10 pp", "+10–20 pp", "+20–35 pp", "+35–50 pp", "+50–60 pp", "> +60 pp"]
    cols_4 = ["#FFFFFF", "#E0F3F8", "#99D8C9", "#41B6C4", "#2C7FB8", "#225EA8", "#081D58"]
    cmap_4 = mcolors.ListedColormap(cols_4)
    norm_4 = mcolors.BoundaryNorm(bins_4, cmap_4.N)
    world["gain_a"] = world["iso3"].map(
        lambda x: c_map_dict.get(x, {}).get("pp_gain_wo_gs", np.nan) if x not in cohort_d else np.nan
    )
    world["gain_b"] = world["iso3"].map(
        lambda x: c_map_dict.get(x, {}).get("pp_gain_w_gs", np.nan) if x not in cohort_d else np.nan
    )
    handles_4 = [
        mpatches.Patch(facecolor=c, edgecolor="#404040", linewidth=0.5, label=lb) for c, lb in zip(cols_4, lbls_4)
    ]
    handles_4.append(
        mpatches.Patch(
            facecolor="#D9D9D9", edgecolor="#666666", hatch="///", linewidth=0.5, label="Excluded non-CAP countries"
        )
    )
    handles_4.append(
        mpatches.Patch(facecolor="#F0F0F0", edgecolor="#B0B0B0", linewidth=0.5, label="No recorded floods")
    )

    t1_a = t1_df[(t1_df["catalog"] == "Without_GS") & (t1_df["benchmark"] == "2021_2025")]
    t1_b = t1_df[(t1_df["catalog"] == "With_GS") & (t1_df["benchmark"] == "2021_2025")]
    n_ev_a = int(t1_a["n_events"].iloc[0]) if not t1_a.empty else 0
    n_ev_b = int(t1_b["n_events"].iloc[0]) if not t1_b.empty else 0

    fig4_p = paths.figures_dir / "fig-4.png"
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 11.3), facecolor="white", dpi=300)
    for ax_i, (col_g, let_g, ttl_g) in enumerate(
        (
            ("gain_a", "a", f"Catalog A: Without Groundsource ({n_ev_a:,} disasters)"),
            ("gain_b", "b", f"Catalog B: With Groundsource ({n_ev_b:,} flood events)"),
        )
    ):
        ax = axes[ax_i]
        world.plot(ax=ax, color="#F0F0F0", edgecolor="#B0B0B0", linewidth=0.35)
        cd_sub = world[world["iso3"].isin(cohort_d)]
        if not cd_sub.empty:
            cd_sub.plot(ax=ax, color="#D9D9D9", hatch="///", edgecolor="#666666", linewidth=0.38)
        valid_g = world[world[col_g].notna()].copy()
        if not valid_g.empty:
            valid_g.plot(column=col_g, ax=ax, cmap=cmap_4, norm=norm_4, edgecolor="#404040", linewidth=0.38)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.axis("off")
        ax.text(-176, 82, let_g, fontsize=17, fontweight="bold", ha="left", va="top")
        ax.set_title(ttl_g, fontsize=10.5, fontweight="bold", pad=6)
    fig.legend(
        handles=handles_4,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.5,
        title="Coverage gain from supplementing CAP with Google DHM (0–7 day, Q2yr) and FFM (τ = 0.50) (pp)",
        title_fontsize=9.2,
    )
    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.99], h_pad=1.6)
    fig.savefig(fig4_p, dpi=300, facecolor="white")
    plt.close(fig)
    written.append(fig4_p)

    # --- Extended Data Fig 1: N=400 validation + Admin-1 vs Admin-0 audit ---
    ed1_p = paths.figures_dir / "ed-fig-1.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.2), facecolor="white", dpi=300)

    def _ed4_row(prefix: str) -> pd.Series:
        matches = ed4_df[ed4_df["strategy"].astype(str).str.startswith(prefix)]
        if matches.empty:
            raise ValueError(f"ed_table4_validation_summary missing strategy starting with '{prefix}'")
        return matches.iloc[0]

    r_c0 = _ed4_row("Country-Only")
    r_ex = _ed4_row("Exclusion")
    r_in = _ed4_row("Inclusion")
    n_main = int(r_ex["sample_n"])

    c0_p = float(r_c0["precision_pct"])
    c0_lo, c0_hi = float(r_c0["precision_cp_ci_low_pct"]), float(r_c0["precision_cp_ci_high_pct"])
    ex_p = float(r_ex["precision_pct"])
    ex_lo, ex_hi = float(r_ex["precision_cp_ci_low_pct"]), float(r_ex["precision_cp_ci_high_pct"])
    ex_r = float(r_ex["recall_pct"]) if np.isfinite(r_ex["recall_pct"]) else 100.0
    ex_r_lo = float(r_ex["recall_cp_ci_low_pct"]) if np.isfinite(r_ex["recall_cp_ci_low_pct"]) else ex_r
    ex_r_hi = float(r_ex["recall_cp_ci_high_pct"]) if np.isfinite(r_ex["recall_cp_ci_high_pct"]) else ex_r
    in_p = float(r_in["precision_pct"])
    in_lo, in_hi = float(r_in["precision_cp_ci_low_pct"]), float(r_in["precision_cp_ci_high_pct"])
    in_r = float(r_in["recall_pct"]) if np.isfinite(r_in["recall_pct"]) else 100.0
    in_r_lo = float(r_in["recall_cp_ci_low_pct"]) if np.isfinite(r_in["recall_cp_ci_low_pct"]) else in_r
    in_r_hi = float(r_in["recall_cp_ci_high_pct"]) if np.isfinite(r_in["recall_cp_ci_high_pct"]) else in_r

    x_lo = max(0.0, min(83.0, min(ex_r_lo, in_r_lo) - 6.0))
    rm = np.linspace(max(0.02, x_lo / 100.0 - 0.01), 1.04, 250)
    pm = np.linspace(0.02, 1.02, 250)
    rg, pg = np.meshgrid(rm, pm)
    fg = 2 * pg * rg / (pg + rg)
    cs = ax1.contour(
        rg * 100,
        pg * 100,
        fg,
        levels=[0.20, 0.40, 0.60, 0.80, 0.90, 0.96],
        colors="#cccccc",
        linestyles=":",
        linewidths=0.9,
    )
    ax1.clabel(cs, inline=True, fontsize=7.8, fmt="F1=%.2f", colors="#777777")

    pts_val = [
        (
            f"Country-only pairing\n({c0_p:.1f}% [{c0_lo:.1f}–{c0_hi:.1f}%, 95% CI]; N = {int(r_c0['sample_n'])})",
            100.0,
            c0_p,
            None,
            [[max(0.0, c0_p - c0_lo)], [max(0.0, c0_hi - c0_p)]],
            "^",
            "#d62728",
            (-4.2, 4.0),
            "right",
            "center",
        ),
        (
            f"Subnational exclusion filter\n({ex_p:.1f}% [{ex_lo:.1f}–{ex_hi:.1f}%, 95% CI]; N = {n_main})",
            ex_r,
            ex_p,
            [[max(0.0, ex_r - ex_r_lo)], [max(0.0, ex_r_hi - ex_r)]],
            [[max(0.0, ex_p - ex_lo)], [max(0.0, ex_hi - ex_p)]],
            "o",
            "#1f77b4",
            (-4.2, 0.0),
            "right",
            "center",
        ),
        (
            f"Subnational inclusion filter\n({in_p:.1f}% [{in_lo:.1f}–{in_hi:.1f}%, 95% CI]; {in_r:.1f}% recall)",
            in_r,
            in_p,
            [[max(0.0, in_r - in_r_lo)], [max(0.0, in_r_hi - in_r)]],
            [[max(0.0, in_p - in_lo)], [max(0.0, in_hi - in_p)]],
            "s",
            "#e66101",
            (-2.0, -10.0),
            "right",
            "top",
        ),
    ]
    for lbl_v, rx, py, xerr, yerr, mk_v, col_v, off_xy, ha_v, va_v in pts_val:
        ax1.errorbar(
            rx,
            py,
            xerr=xerr,
            yerr=yerr,
            fmt=mk_v,
            color=col_v,
            ecolor=col_v,
            elinewidth=1.5,
            capsize=4,
            markersize=9.5,
            zorder=6,
        )
        ax1.annotate(
            lbl_v,
            xy=(rx, py),
            xytext=(rx + off_xy[0], py + off_xy[1]),
            fontsize=8.0,
            fontweight="bold",
            color=col_v,
            ha=ha_v,
            va=va_v,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col_v, alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=col_v, lw=1.0),
            zorder=8,
        )
    ax1.set_xlim(x_lo, 102.5)
    ax1.set_ylim(0, 103)
    ax1.set_xlabel("Conditional recall within retained pool (%)")
    ax1.set_ylabel("Classification / pairing precision (%)")
    ax1.text(-0.10, 1.03, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")
    val_title = (
        f"Manual validation audit (N = {n_main} event-alert pairs; 245 unique alerts)"
        if n_main == 400
        else f"Manual validation audit (N = {n_main} event-alert pairs)"
    )
    ax1.set_title(val_title, fontsize=10.0, pad=8)

    t_sp = df_aud[(df_aud["audit_type"] == "Spatial_Attribution") & (df_aud["lead_N"] == 7)].copy()
    bar_labels = ["Broad CAP\nCatalog A", "Flood-coded\nCatalog A", "Broad CAP\nCatalog B", "Flood-coded\nCatalog B"]
    pairs = [("Without_GS", "perm"), ("Without_GS", "rest"), ("With_GS", "perm"), ("With_GS", "rest")]
    gid1_rec, adm0_rec, gid1_prec, adm0_prec = [], [], [], []
    for cat_p, kind_p in pairs:
        r1 = t_sp[(t_sp["catalog"] == cat_p) & (t_sp["variant"] == f"{kind_p}_strict_GID1")]
        r0 = t_sp[(t_sp["catalog"] == cat_p) & (t_sp["variant"] == f"{kind_p}_legacy_Admin0")]
        gid1_rec.append(float(r1["recall_pct"].iloc[0]) if not r1.empty else 0.0)
        adm0_rec.append(float(r0["recall_pct"].iloc[0]) if not r0.empty else 0.0)
        p1 = float(r1["precision_pct"].iloc[0]) if (not r1.empty and np.isfinite(r1["precision_pct"].iloc[0])) else 0.0
        p0 = float(r0["precision_pct"].iloc[0]) if (not r0.empty and np.isfinite(r0["precision_pct"].iloc[0])) else 0.0
        gid1_prec.append(p1)
        adm0_prec.append(p0)

    x_pos = np.arange(len(bar_labels))
    w_b = 0.19
    ax2.bar(
        x_pos - 1.5 * w_b,
        gid1_rec,
        width=w_b,
        color="#1f77b4",
        edgecolor="#222222",
        label="State/province (Admin-1) recall (%)",
    )
    ax2.bar(
        x_pos - 0.5 * w_b,
        adm0_rec,
        width=w_b,
        color="#aec7e8",
        edgecolor="#222222",
        hatch="//",
        label="Country-level (Admin-0) recall (%)",
    )
    ax2.bar(
        x_pos + 0.5 * w_b,
        gid1_prec,
        width=w_b,
        color="#d95f02",
        edgecolor="#222222",
        label="State/province (Admin-1) precision (%)",
    )
    ax2.bar(
        x_pos + 1.5 * w_b,
        adm0_prec,
        width=w_b,
        color="#fdd0a2",
        edgecolor="#222222",
        hatch="//",
        label="Country-level (Admin-0) precision (%)",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels, fontsize=9.0)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percentage (%)")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.6, frameon=True, facecolor="white")
    ax2.text(-0.10, 1.03, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax2.set_title("Spatial attribution sensitivity (Admin-1 vs. Admin-0)", fontsize=10.0, pad=8)
    plt.tight_layout(w_pad=2.2)
    fig.savefig(ed1_p, dpi=300, facecolor="white")
    plt.close(fig)
    written.append(ed1_p)

    # --- Extended Data Fig 5: Catalog B 5-Year & Post-Onboarding Maps ---
    ed5_p = paths.figures_dir / "ed-fig-5.png"
    tier_map_5yr: dict[str, str] = {}
    tier_map_post: dict[str, str] = {}
    hatch_set: set[str] = set()
    for _, r in c_df.iterrows():
        iso = r["ISO3"]
        coh = str(r["Cohort_and_Category"])
        if coh.startswith("Cohort D"):
            tier_map_5yr[iso] = "Excluded non-CAP"
            tier_map_post[iso] = "Excluded non-CAP"
            continue
        nb = int(r["n_events_w_gs"])
        npb = int(r["post_onb_n_w_gs"])
        tier_map_5yr[iso] = "No floods / no reported pop." if nb == 0 else _tier_label(float(r["cap_pct_w_gs"]))
        tier_map_post[iso] = "No floods / no reported pop." if npb == 0 else _tier_label(float(r["post_onb_pct_w_gs"]))
        if "Category 1" in coh and int(r["onboard_year"]) >= 2022:
            hatch_set.add(iso)

    fig5, axes5 = plt.subplots(2, 1, figsize=(14.4, 11.8), dpi=300)
    for ax, pletter, tmap, do_hatch, title in (
        (
            axes5[0],
            "a",
            tier_map_5yr,
            False,
            f"Catalog B: 5-year CAP event coverage (With Groundsource, N = {n_ev_b:,} events)",
        ),
        (axes5[1], "b", tier_map_post, True, "Catalog B: Post-onboarding CAP coverage (from first CAP alert onward)"),
    ):
        wplot = world.copy()
        wplot["tier"] = wplot["iso3"].map(tmap).fillna("No floods / no reported pop.")
        for tname in tier_ord:
            sub_w = wplot[wplot["tier"] == tname]
            if sub_w.empty:
                continue
            hatch = "..." if tname == "Excluded non-CAP" else None
            sub_w.plot(ax=ax, color=tier_col[tname], edgecolor="#333333", linewidth=0.35, hatch=hatch, zorder=2)
        if do_hatch and hatch_set:
            sub_h = wplot[wplot["iso3"].isin(hatch_set)]
            if not sub_h.empty:
                sub_h.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.55, hatch="///", zorder=3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-56, 84)
        ax.axis("off")
        ax.set_title(title, fontsize=11.8, pad=6)
        ax.text(0.01, 0.96, pletter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")
    fig5.legend(
        handles=handles_2,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9.5,
    )
    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.99], h_pad=1.6)
    fig5.savefig(ed5_p, dpi=300)
    plt.close(fig5)
    written.append(ed5_p)

    return written


def generate_manuscript_tables_and_figures(
    paths: ProjectPaths,
    scoring_state: dict[str, Any] | None = None,
    strict_185: bool = True,
    render_figures: bool = True,
) -> dict[str, Any]:
    """Generate all manuscript & Extended Data CSV tables and publication figures."""
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
    else:
        gid1_list, iso3_list, gid_to_idx, _ = load_universe_sets(paths, strict_185=strict_185)
        data = load_benchmark_data(paths, gid1_list, iso3_list, gid_to_idx)
        df_pr = pd.read_csv(paths.results_dir / "pr_curves_summary.csv")
        df_aud = pd.read_csv(paths.results_dir / "sensitivity_audits.csv")
        df_leads = pd.read_csv(paths.results_dir / "lead_window_sweep.csv")
        df_single = pd.read_csv(paths.results_dir / "single_lead_decay.csv")
        df_ffm_slices = pd.read_csv(paths.results_dir / "ffm_gs_slices_pr.csv")

    n_g = len(gid1_list)

    # Load onboarding & hazard counts (from Zenodo 03_summary_results/ or work_dir)
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

    # 1. Table 1, ED Table 3, ED Table 5
    t1_df = build_table1_summary(df_pr)
    t1_df.to_csv(paths.results_dir / "table1_summary.csv", index=False)
    df_aud.to_csv(paths.results_dir / "ed_table3_sensitivity_matrix.csv", index=False)
    df_pr[df_pr["lead_N"] == 7].to_csv(paths.results_dir / "ed_table5_full_benchmark.csv", index=False)

    # 2. Table 2 & Stream-Gauge NSE summary (Tier 1 or Tier 2 sources)
    gm_path = _optional_resolve(
        paths, "dhm_gauged_metrics.parquet", ("02_admin1_exceedance_matrices", "03_summary_results")
    )
    nse_path = _optional_resolve(paths, "dhm_vs_glofas_geoglows_benchmark_nse.csv", ("03_summary_results",))
    ref_t2 = _optional_resolve(paths, "table2_gauge_vs_catalogs.csv", ("03_summary_results",))
    ref_nse = _optional_resolve(paths, "stream_gauge_nse_summary.csv", ("03_summary_results",))
    t2_df, nse_summary = None, None
    if gm_path is not None or ref_t2 is not None:
        t2_df, nse_summary = run_gauge_verification(
            pr_curves_csv=paths.results_dir / "pr_curves_summary.csv",
            results_dir=paths.results_dir,
            gauge_metrics_parquet=gm_path,
            nse_benchmark_csv=nse_path,
            reference_table2_csv=ref_t2,
            reference_nse_summary_csv=ref_nse,
        )

    # 3. De-identified N=400 validation summary (Extended Data Table 4 + full confusion/spatial tables)
    val_csv = _optional_resolve(paths, "validation_n400_deidentified.csv", ("04_validation_deidentified",))
    if val_csv is None:
        val_csv = _optional_resolve(paths, "validation_audit_deidentified_n400.csv", ("04_validation_deidentified",))
    ed4_df = run_validation_audit(
        deidentified_csv=val_csv,
        output_csv=paths.results_dir / "ed_table4_validation_summary.csv",
    )

    # 4. Country-level coverage, hybrid gains, and post-onboarding parity (Catalog A & Catalog B)
    w_cap7 = make_w_matrix(data, "CAP_Perm", 1, 7, n_g)
    w_dhm7 = make_w_matrix(data, "DHM_Oper", 2, 7, n_g)
    w_ffm7 = make_w_matrix(data, "FFM", 0.50, 7, n_g)
    w_mod7 = w_dhm7 | w_ffm7
    w_hyb7 = w_cap7 | w_mod7

    cat_a = data["cat_without_gs"]
    cat_b = data["cat_with_gs"]
    va_cap, da_cap = build_event_detection_flags(cat_a, w_cap7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, da_mod = build_event_detection_flags(cat_a, w_mod7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, da_hyb = build_event_detection_flags(cat_a, w_hyb7, gid_to_idx, "2021-01-01", "2025-12-31", 7)

    vb_cap, db_cap = build_event_detection_flags(cat_b, w_cap7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, db_mod = build_event_detection_flags(cat_b, w_mod7, gid_to_idx, "2021-01-01", "2025-12-31", 7)
    _, db_hyb = build_event_detection_flags(cat_b, w_hyb7, gid_to_idx, "2021-01-01", "2025-12-31", 7)

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

    # ED Table 1: Cohort D Excluded Jurisdictions
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

    # ED Table 6a: Annual CAP Onboarding Trajectory (Catalog A & Catalog B)
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

    # 5. Summary verification log
    summary_payload = {
        "n_eval_jurisdictions": len(eval_150),
        "n_cohort_d": len(cohort_d),
        "cat_a_eval_events": int(len(ev_a150)),
        "cat_b_eval_events": int(len(ev_b150)),
        "cat_a_eval_pop": float(ev_a150["waterfall_pop"].sum()),
        "cat_b_eval_pop": float(ev_b150["waterfall_pop"].sum()),
    }
    (paths.results_dir / "reproduction_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    # 6. Render all figures
    figures_written: list[Path] = []
    if render_figures:
        figures_written.extend(render_pr_and_lead_figures(df_pr, df_leads, df_single, df_ffm_slices, paths.figures_dir))
        figures_written.extend(
            render_maps_and_validation_figures(paths, supp, c_df, ed6b_df, haz, df_aud, t1_df, ed4_df)
        )

    print(
        f"Manuscript tables & {len(figures_written)} figures generated in {time.time() - t0:.1f}s.",
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
        "figures": figures_written,
    }


def render_figures_from_results(paths: ProjectPaths) -> list[Path]:
    """Re-render all figures directly from precomputed summary CSVs in ``paths.results_dir``."""
    df_pr = pd.read_csv(paths.results_dir / "pr_curves_summary.csv")
    df_leads = pd.read_csv(paths.results_dir / "lead_window_sweep.csv")
    df_single = pd.read_csv(paths.results_dir / "single_lead_decay.csv")
    df_ffm_slices = pd.read_csv(paths.results_dir / "ffm_gs_slices_pr.csv")
    df_aud = pd.read_csv(paths.results_dir / "sensitivity_audits.csv")
    t1_df = pd.read_csv(paths.results_dir / "table1_summary.csv")
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
            val_csv = _optional_resolve(
                paths, "validation_audit_deidentified_n400.csv", ("04_validation_deidentified",)
            )
        ed4_df = run_validation_audit(deidentified_csv=val_csv)
    written = render_pr_and_lead_figures(df_pr, df_leads, df_single, df_ffm_slices, paths.figures_dir)
    written.extend(render_maps_and_validation_figures(paths, supp, c_df, ed6b_df, haz, df_aud, t1_df, ed4_df))
    return written


def _optional_resolve(paths: ProjectPaths, filename: str, subdirs: tuple[str, ...]) -> Path | None:
    try:
        return paths.resolve_data_file(filename, *subdirs)
    except FileNotFoundError:
        return None
