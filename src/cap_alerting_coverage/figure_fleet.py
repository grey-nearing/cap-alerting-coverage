"""Publication-grade single-composite figure fleet (5 Main, 5 Extended Data, 1 Supplementary).

Every figure is exported as a single unified composite in vector PDF, vector SVG, and 300-DPI PNG:
  - Main: fig-1, fig-2, fig-3, fig-4, fig-5
  - Extended Data: ed-fig-1, ed-fig-2, ed-fig-3, ed-fig-4, ed-fig-5
  - Supplementary: supp-fig-1
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath

from cap_alerting_coverage.figure_style import (
    COVERAGE_CMAP,
    GAIN_CMAP,
    PALETTE,
    panel_label,
    save_publication_figure,
    setup_nature_style,
    style_robinson_ax,
)

SHORT_COUNTRY: dict[str, str] = {
    "Iran (Islamic Republic of)": "Iran",
    "Republic of the Congo": "Rep. of Congo",
    "North Macedonia": "N. Macedonia",
    "Solomon Islands": "Solomon Is.",
}


# ---------------------------------------------------------------------------
# Map & distribution figures: Fig. 1, Fig. 2, Fig. 4, Supplementary Fig. 1
# ---------------------------------------------------------------------------


def render_fig1(
    world_rob: gpd.GeoDataFrame,
    frame_gdf: gpd.GeoDataFrame,
    supp: pd.DataFrame,
    haz: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig. 1: Global CAP capability map (a) + hazard-type horizontal bar chart (b)."""
    setup_nature_style()
    supp_c = supp.copy()
    supp_c["ISO3"] = supp_c["ISO3"].astype(str).str.strip()
    cat1 = set(supp_c[supp_c["Cohort_and_Category"].str.startswith("Cohort A")]["ISO3"])
    cat2 = set(supp_c[supp_c["Cohort_and_Category"].str.startswith("Cohort B")]["ISO3"])
    cat3 = set(supp_c[supp_c["Cohort_and_Category"].str.startswith("Cohort C")]["ISO3"])
    excl = set(supp_c[supp_c["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"])
    eval150 = cat1 | cat2 | cat3

    lbl_c1 = f"Category 1: Active flood CAP alerts ({len(cat1)})"
    lbl_c2 = f"Category 2: Other CAP alerts only ({len(cat2)})"
    lbl_c3 = f"Category 3: No CAP alerts ({len(cat3)})"
    lbl_ex = f"Excluded non-CAP warning systems ({len(excl)})"
    lbl_no = "Outside study set"
    cmap_dict = {
        lbl_c1: PALETTE["cat1"],
        lbl_c2: PALETTE["cat2"],
        lbl_c3: PALETTE["cat3"],
        lbl_ex: PALETTE["excluded_fill"],
        lbl_no: PALETTE["no_data"],
    }

    def _cls(iso: str) -> str:
        if iso in excl:
            return lbl_ex
        if iso in cat1:
            return lbl_c1
        if iso in cat2:
            return lbl_c2
        if iso in cat3:
            return lbl_c3
        return lbl_no

    w = world_rob.copy()
    w["cls"] = w["iso3"].map(_cls)

    fig = plt.figure(figsize=(7.2, 6.25), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.55, 0.92], hspace=0.28)

    ax_a = fig.add_subplot(gs[0])
    style_robinson_ax(ax_a, frame_gdf)
    w.plot(ax=ax_a, color=w["cls"].map(cmap_dict), edgecolor=PALETTE["border_map"], linewidth=0.28, zorder=2)
    ex_sub = w[w["iso3"].isin(excl)]
    if not ex_sub.empty:
        ex_sub.plot(
            ax=ax_a,
            facecolor="none",
            edgecolor=PALETTE["excluded_edge"],
            linewidth=0.35,
            hatch="////",
            zorder=3,
        )
    handles = [
        mpatches.Patch(facecolor=cmap_dict[lbl_c1], edgecolor=PALETTE["border_map"], lw=0.45, label=lbl_c1),
        mpatches.Patch(facecolor=cmap_dict[lbl_c2], edgecolor=PALETTE["border_map"], lw=0.45, label=lbl_c2),
        mpatches.Patch(facecolor=cmap_dict[lbl_c3], edgecolor=PALETTE["border_map"], lw=0.45, label=lbl_c3),
        mpatches.Patch(
            facecolor=cmap_dict[lbl_ex],
            edgecolor=PALETTE["excluded_edge"],
            lw=0.45,
            hatch="////",
            label=lbl_ex,
        ),
        mpatches.Patch(facecolor=cmap_dict[lbl_no], edgecolor="#B0B7C3", lw=0.45, label=lbl_no),
    ]
    ax_a.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.4,
    )
    panel_label(ax_a, "a", x=-0.01, y=0.98)

    ax_b = fig.add_subplot(gs[1])
    haz150 = haz[haz["country_iso3"].isin(eval150)]
    n_eval = max(1, len(eval150))
    bars_data = sorted(
        [
            ("Storms & heavy rain", int((haz150["n_storms"] > 0).sum()), False),
            ("Floods (Category 1)", len(cat1), True),
            ("Extreme cold & snow", int((haz150["n_cold"] > 0).sum()), False),
            ("Extreme heat", int((haz150["n_heat"] > 0).sum()), False),
            ("Wildfires", int((haz150["n_wildfire"] > 0).sum()), False),
            ("Earthquakes & tsunami", int((haz150["n_earthquake"] > 0).sum()), False),
            ("Volcanic activity", int((haz150["n_volcano"] > 0).sum()), False),
        ],
        key=lambda x: x[1],
    )
    labels = [b[0] for b in bars_data]
    counts = [b[1] for b in bars_data]
    colors = [PALETTE["cat1"] if b[2] else "#476C8A" for b in bars_data]
    y_pos = np.arange(len(labels))
    ax_b.barh(y_pos, counts, height=0.62, color=colors, edgecolor="none", zorder=3)
    xmax = max(132, int(max(counts) * 1.28) + 5) if max(counts) > 10 else max(10, int(max(counts) * 1.45) + 2)
    pad_txt = xmax * 0.014
    for yi, cnt, (_, _, is_fl) in zip(y_pos, counts, bars_data):
        pct = 100.0 * cnt / n_eval
        ax_b.text(
            cnt + pad_txt,
            yi,
            f"{cnt} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=6.5,
            fontweight="bold" if is_fl else "normal",
            color=PALETTE["cat1"] if is_fl else PALETTE["ink"],
        )
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(labels, fontsize=6.9)
    for tick, (_, _, is_fl) in zip(ax_b.get_yticklabels(), bars_data):
        if is_fl:
            tick.set_fontweight("bold")
            tick.set_color(PALETTE["cat1"])
    ax_b.set_xlim(0, xmax)
    if xmax >= 100:
        ax_b.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax_b.set_xlabel(f"Countries issuing CAP alerts in 2021–2025 (of {len(eval150)} in CAP analysis)", fontsize=7.1)
    ax_b.grid(axis="x", alpha=0.6, zorder=0)
    ax_b.spines["left"].set_linewidth(0.65)
    panel_label(ax_b, "b", x=-0.30, y=1.06)

    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.07, hspace=0.34)
    ax_a.set_position([0.02, 0.43, 0.96, 0.54])
    ax_b.set_position([0.23, 0.08, 0.72, 0.27])
    return save_publication_figure(fig, figures_dir, "fig-1")


def _plot_onboarding_dumbbell(
    ax: plt.Axes,
    ed6b: pd.DataFrame,
    cov5_col: str,
    post_col: str,
    n_post_col: str,
    diff_col: str,
    ev5_col: str,
    cov5_cnt_col: str,
    post_cnt_col: str,
    title: str,
    panel_let: str,
) -> None:
    """Horizontal Cleveland dumbbell chart for Category 1 late onboarders (2022-2025)."""
    n_total_late = len(ed6b)
    if n_total_late == 0:
        ax.text(0.5, 0.5, "No 2022–2025 Category 1 late onboarders in subset", ha="center", va="center", fontsize=7.0)
        ax.set_title(title, fontsize=7.3, fontweight="bold", pad=5)
        ax.axis("off")
        panel_label(ax, panel_let, x=-0.355, y=1.03)
        return

    df_zero = ed6b[ed6b[n_post_col] <= 0].copy()
    df = ed6b[ed6b[n_post_col] > 0].copy()
    tot_5 = float(ed6b[ev5_col].sum())
    cov_5 = float(ed6b[cov5_cnt_col].sum())
    tot_p = float(ed6b[n_post_col].sum())
    cov_p = float(ed6b[post_cnt_col].sum())
    subtotal_5yr = (100.0 * cov_5 / tot_5) if tot_5 > 0 else 0.0
    subtotal_post = (100.0 * cov_p / tot_p) if tot_p > 0 else 0.0

    df["short"] = df["Country"].map(lambda c: SHORT_COUNTRY.get(str(c), str(c)))
    df["label"] = df.apply(lambda r: f"{r['short']} ({int(r['Onboard_Year'])}, n={int(r[n_post_col])})", axis=1)
    df = df.sort_values([post_col, diff_col], ascending=[True, True]).reset_index(drop=True)

    y_rows = np.arange(len(df), dtype=float)
    y_sub = -1.35

    ax.axhspan(y_sub - 0.55, y_sub + 0.55, color="#EAF3F1", zorder=0)
    ax.axhline(-0.65, color="#CBD5E1", lw=0.65, ls="-", zorder=1)

    for y, (_, r) in zip(y_rows, df.iterrows()):
        v0 = float(r[cov5_col]) if np.isfinite(r[cov5_col]) else 0.0
        v1 = float(r[post_col]) if np.isfinite(r[post_col]) else 0.0
        is_aus = str(r["ISO3"]) == "AUS"
        seg_col = "#94A3B8" if abs(v1 - v0) < 0.05 else "#2E8B7A"
        ax.plot([v0, v1], [y, y], color=seg_col, lw=1.35, zorder=2)
        ax.scatter([v0], [y], s=20, facecolor="white", edgecolor="#475569", lw=0.9, zorder=4)
        ax.scatter([v1], [y], s=26, facecolor="#0E4D64", edgecolor="white", lw=0.45, zorder=5)
        gain = v1 - v0
        tag = f"+{gain:.0f} pp" if gain >= 0.5 else "0 pp"
        if is_aus:
            tag = f"{tag} (non-CAP prior to Sep 2025)"
        ax.text(
            max(v0, v1) + 1.8,
            y,
            tag,
            va="center",
            ha="left",
            fontsize=5.4,
            color="#475569" if not is_aus else "#9A3412",
            style="italic" if is_aus else "normal",
        )

    ax.plot([subtotal_5yr, subtotal_post], [y_sub, y_sub], color=PALETTE["cat1"], lw=2.0, zorder=3)
    ax.scatter([subtotal_5yr], [y_sub], s=28, facecolor="white", edgecolor=PALETTE["ink"], lw=1.0, zorder=5)
    ax.scatter([subtotal_post], [y_sub], s=34, facecolor=PALETTE["cat1"], edgecolor="white", lw=0.5, zorder=6)
    ax.text(
        subtotal_post + 1.8,
        y_sub,
        f"{subtotal_5yr:.1f}% → {subtotal_post:.1f}% (+{subtotal_post - subtotal_5yr:.1f} pp)",
        va="center",
        ha="left",
        fontsize=5.8,
        fontweight="bold",
        color=PALETTE["cat1"],
    )

    yticks = [y_sub] + list(y_rows)
    ylabels = [f"All {n_total_late} late onboarders"] + df["label"].tolist()
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5.75)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(PALETTE["cat1"])

    ax.set_xlim(-2, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(y_sub - 0.85, max(0.5, len(df) - 0.35))
    ax.set_xlabel("CAP flood event coverage (%)", fontsize=7.0)
    ax.set_title(title, fontsize=7.3, fontweight="bold", pad=5)
    ax.grid(axis="x", alpha=0.55, zorder=0)

    h5 = plt.Line2D([], [], marker="o", ms=4.5, mfc="white", mec="#475569", mew=0.9, ls="none", label="Full 5-year (2021–2025)")
    hp = plt.Line2D([], [], marker="o", ms=5.0, mfc="#0E4D64", mec="white", mew=0.45, ls="none", label="Post-onboarding onward (n events)")
    ax.legend(handles=[h5, hp], loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=5.7, handletextpad=0.35)
    if not df_zero.empty:
        notes = [f"{SHORT_COUNTRY.get(str(r['Country']), str(r['Country']))} ({int(r['Onboard_Year'])})" for _, r in df_zero.iterrows()]
        ax.text(
            116,
            min(1.5, max(0.2, len(df) * 0.15)),
            f"{', '.join(notes)}:\n0 Catalog A events",
            ha="right",
            va="bottom",
            fontsize=5.3,
            style="italic",
            color="#64748B",
        )
    panel_label(ax, panel_let, x=-0.355, y=1.03)


def render_fig2(
    world_rob: gpd.GeoDataFrame,
    frame_gdf: gpd.GeoDataFrame,
    supp: pd.DataFrame,
    c_df: pd.DataFrame,
    ed6b: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig. 2: Continuous CAP coverage maps (a: events, b: people) + onboarding dumbbell chart (c)."""
    setup_nature_style()
    excl = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"].astype(str).str.strip())
    cmap_dict = c_df.set_index("ISO3").to_dict("index")
    c_eval = c_df[~c_df["ISO3"].isin(excl)]
    n_ev_a = int(c_eval["n_events_wo_gs"].sum())
    pop_m = float(c_eval["pop_tot_wo_gs"].sum()) / 1e6

    w = world_rob.copy()
    w["ev_pct"] = w["iso3"].map(
        lambda x: float(cmap_dict[x]["cap_pct_wo_gs"])
        if (x in cmap_dict and x not in excl and cmap_dict[x]["n_events_wo_gs"] > 0 and np.isfinite(cmap_dict[x]["cap_pct_wo_gs"]))
        else np.nan
    )
    w["pop_pct"] = w["iso3"].map(
        lambda x: float(cmap_dict[x]["pop_pct_wo_gs"])
        if (x in cmap_dict and x not in excl and cmap_dict[x]["pop_tot_wo_gs"] > 0 and np.isfinite(cmap_dict[x]["pop_pct_wo_gs"]))
        else np.nan
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)

    fig = plt.figure(figsize=(7.2, 9.75), dpi=300, facecolor="white")
    ax_a = fig.add_axes([0.02, 0.715, 0.96, 0.262])
    ax_b = fig.add_axes([0.02, 0.432, 0.96, 0.262])
    cax = fig.add_axes([0.15, 0.398, 0.44, 0.0125])
    ax_c = fig.add_axes([0.27, 0.048, 0.70, 0.278])

    pop_lbl = f"{pop_m:.2f} million people" if pop_m >= 0.1 else f"{int(c_eval['pop_tot_wo_gs'].sum()):,} people"
    for ax, col, let, ttl in (
        (ax_a, "ev_pct", "a", f"Five-year CAP event coverage (Catalog A; n = {n_ev_a:,} events)"),
        (ax_b, "pop_pct", "b", f"Five-year CAP affected-population coverage ({pop_lbl})"),
    ):
        style_robinson_ax(ax, frame_gdf)
        w.plot(ax=ax, color=PALETTE["no_data"], edgecolor="#B5BDC7", linewidth=0.25, zorder=1)
        ex_sub = w[w["iso3"].isin(excl)]
        if not ex_sub.empty:
            ex_sub.plot(
                ax=ax,
                color=PALETTE["excluded_fill"],
                edgecolor=PALETTE["excluded_edge"],
                linewidth=0.30,
                hatch="////",
                zorder=2,
            )
        valid = w[w[col].notna()]
        if not valid.empty:
            valid.plot(
                column=col,
                ax=ax,
                cmap=COVERAGE_CMAP,
                norm=norm,
                edgecolor=PALETTE["border_map"],
                linewidth=0.28,
                zorder=3,
            )
        ax.set_title(ttl, fontsize=7.4, fontweight="bold", pad=2)
        panel_label(ax, let, x=0.00, y=0.95)

    sm = plt.cm.ScalarMappable(cmap=COVERAGE_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cb.ax.tick_params(labelsize=6.3, length=2.2, pad=1.5)
    cb.set_label("CAP flood warning coverage (%)", fontsize=6.5, labelpad=1.5)
    cb.outline.set_linewidth(0.5)

    h_map = [
        mpatches.Patch(
            facecolor=PALETTE["excluded_fill"],
            edgecolor=PALETTE["excluded_edge"],
            hatch="////",
            lw=0.45,
            label=f"Excluded non-CAP (n = {len(excl)})",
        ),
        mpatches.Patch(facecolor=PALETTE["no_data"], edgecolor="#B5BDC7", lw=0.45, label="No recorded floods / pop."),
    ]
    fig.legend(
        handles=h_map,
        loc="center left",
        bbox_to_anchor=(0.61, 0.404),
        ncol=1,
        frameon=False,
        fontsize=6.0,
        handlelength=1.4,
        handleheight=0.85,
        labelspacing=0.35,
    )

    _plot_onboarding_dumbbell(
        ax_c,
        ed6b,
        cov5_col="Cov_5yr_Pct",
        post_col="Post_Onboard_Pct",
        n_post_col="Post_Onboard_Events",
        diff_col="Diff_pp",
        ev5_col="Events_5yr_WoGS",
        cov5_cnt_col="CAP_Cov_5yr_WoGS",
        post_cnt_col="Post_Onboard_CAP_Cov",
        title="CAP coverage before vs. after onboarding in Category 1 late onboarders (Catalog A)",
        panel_let="c",
    )
    return save_publication_figure(fig, figures_dir, "fig-2")


def _plot_gain_distribution(
    ax: plt.Axes,
    c_df: pd.DataFrame,
    gain_col: str,
    n_col: str,
    let: str,
    title: str,
    show_xlabel: bool,
) -> None:
    """Distributional panel for Fig. 4b & 4d: country-level +pp gains by CAP Category."""
    rng = np.random.default_rng(42)
    cats = (
        ("Cohort C", "Category 3\n(No CAP)", PALETTE["cat3"]),
        ("Cohort B", "Category 2\n(Other CAP)", PALETTE["cat2"]),
        ("Cohort A", "Category 1\n(Flood CAP)", PALETTE["cat1"]),
    )
    y_positions = [0, 1, 2]
    ylabels = []
    for yi, (prefix, lbl, col) in zip(y_positions, cats):
        sub = c_df[
            c_df["Cohort_and_Category"].str.startswith(prefix)
            & (c_df[n_col] > 0)
            & np.isfinite(c_df[gain_col])
        ]
        vals = sub[gain_col].to_numpy(dtype=float)
        n_c = len(vals)
        ylabels.append(f"{lbl}\n(n = {n_c})")
        if n_c == 0:
            continue
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        jit = rng.uniform(-0.17, 0.17, size=n_c)
        ax.scatter(vals, yi + jit, s=11, color=col, alpha=0.45, edgecolor="none", zorder=2)
        ax.plot([q25, q75], [yi, yi], color=PALETTE["ink"], lw=2.4, solid_capstyle="round", zorder=4)
        ax.scatter([med], [yi], s=34, marker="D", facecolor="white", edgecolor=PALETTE["ink"], lw=1.1, zorder=5)
        ax.text(
            med,
            yi + 0.25,
            f"med +{med:.0f} pp",
            fontsize=5.6,
            fontweight="bold",
            color=PALETTE["ink"],
            va="bottom",
            ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.4),
            zorder=6,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=6.2)
    ax.set_xlim(-3, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.58, 2.65)
    ax.grid(axis="x", alpha=0.55, zorder=0)
    ax.set_title(title, fontsize=7.0, fontweight="bold", pad=4)
    if show_xlabel:
        ax.set_xlabel("Coverage gain (+pp)", fontsize=6.9)
    panel_label(ax, let, x=-0.28, y=1.03)


def render_fig4(
    world_rob: gpd.GeoDataFrame,
    frame_gdf: gpd.GeoDataFrame,
    supp: pd.DataFrame,
    c_df: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig. 4: 2x3 grid — global coverage-gain maps (a, c) + Category 1/2/3 gain distributions (b, d)."""
    setup_nature_style()
    excl = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"].astype(str).str.strip())
    cmap_dict = c_df.set_index("ISO3").to_dict("index")
    c_eval = c_df[~c_df["ISO3"].isin(excl)]
    n_ev_a = int(c_eval["n_events_wo_gs"].sum())
    n_ev_b = int(c_eval["n_events_w_gs"].sum())

    w = world_rob.copy()
    w["gain_a"] = w["iso3"].map(
        lambda x: float(cmap_dict[x]["pp_gain_wo_gs"])
        if (x in cmap_dict and x not in excl and cmap_dict[x]["n_events_wo_gs"] > 0 and np.isfinite(cmap_dict[x]["pp_gain_wo_gs"]))
        else np.nan
    )
    w["gain_b"] = w["iso3"].map(
        lambda x: float(cmap_dict[x]["pp_gain_w_gs"])
        if (x in cmap_dict and x not in excl and cmap_dict[x]["n_events_w_gs"] > 0 and np.isfinite(cmap_dict[x]["pp_gain_w_gs"]))
        else np.nan
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)

    fig = plt.figure(figsize=(7.2, 5.55), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.1, 0.98], hspace=0.28, wspace=0.28)

    rows = (
        (0, "gain_a", "pp_gain_wo_gs", "n_events_wo_gs", "a", "b", f"Catalog A: Coverage gain (n = {n_ev_a:,} disasters)", "Country gains (Catalog A)"),
        (1, "gain_b", "pp_gain_w_gs", "n_events_w_gs", "c", "d", f"Catalog B: Coverage gain (n = {n_ev_b:,} flood events)", "Country gains (Catalog B)"),
    )
    for r_i, map_col, gain_col, n_col, let_m, let_d, ttl_m, ttl_d in rows:
        ax_m = fig.add_subplot(gs[r_i, :2])
        style_robinson_ax(ax_m, frame_gdf)
        w.plot(ax=ax_m, color=PALETTE["no_data"], edgecolor="#B5BDC7", linewidth=0.24, zorder=1)
        ex_sub = w[w["iso3"].isin(excl)]
        if not ex_sub.empty:
            ex_sub.plot(
                ax=ax_m,
                color=PALETTE["excluded_fill"],
                edgecolor=PALETTE["excluded_edge"],
                linewidth=0.28,
                hatch="////",
                zorder=2,
            )
        valid = w[w[map_col].notna()]
        if not valid.empty:
            valid.plot(
                column=map_col,
                ax=ax_m,
                cmap=GAIN_CMAP,
                norm=norm,
                edgecolor=PALETTE["border_map"],
                linewidth=0.26,
                zorder=3,
            )
        ax_m.set_title(ttl_m, fontsize=7.2, fontweight="bold", pad=3)
        panel_label(ax_m, let_m, x=-0.01, y=0.96)

        ax_d = fig.add_subplot(gs[r_i, 2])
        _plot_gain_distribution(ax_d, c_df, gain_col, n_col, let_d, ttl_d, show_xlabel=(r_i == 1))

    cax = fig.add_axes([0.07, 0.045, 0.38, 0.018])
    sm = plt.cm.ScalarMappable(cmap=GAIN_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.set_ticklabels(["0", "+20", "+40", "+60", "+80", "+100 pp"])
    cb.ax.tick_params(labelsize=6.2, length=2.2, pad=1.5)
    cb.set_label("Absolute coverage gain: CAP + DHM (0–7 d, Q2yr) + FFM (τ = 0.50)", fontsize=6.2, labelpad=1.5)
    cb.outline.set_linewidth(0.5)

    h_ex = [
        mpatches.Patch(
            facecolor=PALETTE["excluded_fill"],
            edgecolor=PALETTE["excluded_edge"],
            hatch="////",
            lw=0.45,
            label=f"Excluded non-CAP (n = {len(excl)})",
        ),
        mpatches.Patch(facecolor=PALETTE["no_data"], edgecolor="#B5BDC7", lw=0.45, label="No recorded floods"),
    ]
    fig.legend(
        handles=h_ex,
        loc="center left",
        bbox_to_anchor=(0.48, 0.052),
        ncol=1,
        frameon=False,
        fontsize=5.9,
        handlelength=1.4,
        handleheight=0.85,
        labelspacing=0.3,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.955, bottom=0.145, hspace=0.30, wspace=0.34)
    return save_publication_figure(fig, figures_dir, "fig-4")


def render_supp_fig1(
    world_rob: gpd.GeoDataFrame,
    frame_gdf: gpd.GeoDataFrame,
    supp: pd.DataFrame,
    c_df: pd.DataFrame,
    ed6b: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Supplementary Fig. 1: Catalog B CAP coverage map (a) + 25-country onboarding chart (b)."""
    setup_nature_style()
    excl = set(supp[supp["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"].astype(str).str.strip())
    cmap_dict = c_df.set_index("ISO3").to_dict("index")
    n_ev_b = int(c_df[~c_df["ISO3"].isin(excl)]["n_events_w_gs"].sum())

    w = world_rob.copy()
    w["ev_b"] = w["iso3"].map(
        lambda x: float(cmap_dict[x]["cap_pct_w_gs"])
        if (x in cmap_dict and x not in excl and cmap_dict[x]["n_events_w_gs"] > 0 and np.isfinite(cmap_dict[x]["cap_pct_w_gs"]))
        else np.nan
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)

    fig = plt.figure(figsize=(7.2, 7.25), dpi=300, facecolor="white")
    ax_a = fig.add_axes([0.02, 0.59, 0.96, 0.378])
    cax = fig.add_axes([0.15, 0.545, 0.44, 0.016])
    ax_b = fig.add_axes([0.27, 0.062, 0.70, 0.382])

    style_robinson_ax(ax_a, frame_gdf)
    w.plot(ax=ax_a, color=PALETTE["no_data"], edgecolor="#B5BDC7", linewidth=0.25, zorder=1)
    ex_sub = w[w["iso3"].isin(excl)]
    if not ex_sub.empty:
        ex_sub.plot(
            ax=ax_a,
            color=PALETTE["excluded_fill"],
            edgecolor=PALETTE["excluded_edge"],
            linewidth=0.30,
            hatch="////",
            zorder=2,
        )
    valid = w[w["ev_b"].notna()]
    if not valid.empty:
        valid.plot(
            column="ev_b",
            ax=ax_a,
            cmap=COVERAGE_CMAP,
            norm=norm,
            edgecolor=PALETTE["border_map"],
            linewidth=0.28,
            zorder=3,
        )
    ax_a.set_title(
        f"Five-year CAP event coverage (Catalog B, with local news; n = {n_ev_b:,} events)",
        fontsize=7.4,
        fontweight="bold",
        pad=2,
    )
    panel_label(ax_a, "a", x=0.00, y=0.95)

    sm = plt.cm.ScalarMappable(cmap=COVERAGE_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cb.ax.tick_params(labelsize=6.3, length=2.2, pad=1.5)
    cb.set_label("CAP flood warning coverage (%)", fontsize=6.5, labelpad=1.5)
    cb.outline.set_linewidth(0.5)

    h_map = [
        mpatches.Patch(
            facecolor=PALETTE["excluded_fill"],
            edgecolor=PALETTE["excluded_edge"],
            hatch="////",
            lw=0.45,
            label=f"Excluded non-CAP (n = {len(excl)})",
        ),
        mpatches.Patch(facecolor=PALETTE["no_data"], edgecolor="#B5BDC7", lw=0.45, label="No recorded floods"),
    ]
    fig.legend(
        handles=h_map,
        loc="center left",
        bbox_to_anchor=(0.61, 0.553),
        ncol=1,
        frameon=False,
        fontsize=6.0,
        handlelength=1.4,
        handleheight=0.85,
        labelspacing=0.35,
    )

    _plot_onboarding_dumbbell(
        ax_b,
        ed6b,
        cov5_col="Cov_5yr_Pct_WGS",
        post_col="Post_Onboard_Pct_WGS",
        n_post_col="Post_Onboard_Events_WGS",
        diff_col="Diff_pp_WGS",
        ev5_col="Events_5yr_WGS",
        cov5_cnt_col="CAP_Cov_5yr_WGS",
        post_cnt_col="Post_Onboard_CAP_Cov_WGS",
        title="CAP coverage before vs. after onboarding in Category 1 late onboarders (Catalog B)",
        panel_let="b",
    )
    return save_publication_figure(fig, figures_dir, "supp-fig-1")


# ---------------------------------------------------------------------------
# PR & Lead-Time figures: Fig. 3, ED Fig. 2, ED Fig. 3, ED Fig. 4
# ---------------------------------------------------------------------------


def _draw_timeline_panel(ax: plt.Axes) -> None:
    """Fig. 3a: Editorial scoring timeline centered on flood onset."""
    ax.set_xlim(-8.8, 6.2)
    ax.set_ylim(-0.42, 4.55)
    ax.axvspan(-7.0, 0.0, color="#E8F1F8", alpha=0.75, zorder=0)
    ax.axvspan(0.0, 3.0, color="#FBF0E4", alpha=0.80, zorder=0)
    for xv, col, ls, lw in (
        (-7.0, "#1B5E8A", ":", 1.0),
        (0.0, "#9E2A2B", "--", 1.25),
        (3.0, "#B45309", ":", 1.0),
    ):
        ax.axvline(xv, color=col, linestyle=ls, linewidth=lw, zorder=1)

    ax.add_patch(
        FancyArrowPatch(
            (-7.0, 4.18),
            (3.0, 4.18),
            arrowstyle="<->,head_length=3.2,head_width=1.8",
            color=PALETTE["ink"],
            lw=0.95,
            mutation_scale=1.0,
        )
    )
    ax.text(
        -2.0,
        4.26,
        "Matching window (7 days before onset through flood end)",
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.add_patch(
        FancyArrowPatch(
            (-7.0, 3.76),
            (0.0, 3.76),
            arrowstyle="<->,head_length=2.8,head_width=1.6",
            color="#1B5E8A",
            lw=0.85,
            mutation_scale=1.0,
        )
    )
    ax.text(-3.5, 3.84, "Pre-onset (7 days)", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#1B5E8A")
    ax.text(1.5, 3.84, "Active flood", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#9A3412")

    y1 = 3.02
    ax.axhline(y1, color="#E2E8F0", lw=0.6, zorder=0)
    ax.scatter([-7.95], [y1], s=34, marker="^", color="#9CA3AF", edgecolor="#4B5563", lw=0.6, zorder=4)
    ax.text(-7.95, y1 - 0.25, "Outside window\n(unmatched)", ha="center", va="top", fontsize=5.8, color="#4B5563")
    ax.add_patch(
        FancyBboxPatch(
            (-4.55, y1 - 0.16),
            3.60,
            0.32,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            fc="#FCEAEA",
            ec=PALETTE["cap_perm"],
            lw=0.9,
            zorder=2,
        )
    )
    ax.scatter([-4.2, -3.2, -1.3], [y1] * 3, s=38, marker="^", color=PALETTE["cap_perm"], edgecolor="white", lw=0.45, zorder=5)
    ax.text(-2.75, y1 + 0.23, "Alerts ≤ 2 alert-free days apart → 1 alert episode", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#8A1414")

    y2 = 2.06
    ax.axhline(y2, color="#E2E8F0", lw=0.6, zorder=0)
    xs_sim = np.linspace(-7.4, 4.6, 180)
    ys_sim = y2 - 0.13 + 0.38 * np.exp(-0.5 * ((xs_sim - 1.1) / 1.45) ** 2)
    q2_lvl = y2 + 0.10
    ax.plot(xs_sim, ys_sim, color=PALETTE["dhm_0d"], lw=1.35, zorder=3)
    ax.plot([-7.4, 4.6], [q2_lvl, q2_lvl], color=PALETTE["dhm_0d"], ls="--", lw=0.8, alpha=0.85, zorder=2)
    ax.fill_between(xs_sim, q2_lvl, ys_sim, where=(ys_sim >= q2_lvl), color="#A7E2D2", alpha=0.65, zorder=2)
    ax.text(3.25, q2_lvl + 0.05, "2-yr flood threshold (Q2yr)", fontsize=6.1, fontweight="bold", color="#0F6B53", va="bottom", ha="left")
    ax.text(-4.8, y2 + 0.15, "Simulated river peak", fontsize=6.1, color="#0F6B53", ha="left", va="bottom")

    y3 = 1.10
    ax.axhline(y3, color="#E2E8F0", lw=0.6, zorder=0)
    ax.scatter([-5.0], [y3 + 0.07], s=30, marker="o", color="white", edgecolor=PALETTE["dhm_oper"], lw=1.1, zorder=5)
    ax.scatter([0.0], [y3 + 0.07], s=40, marker="o", color=PALETTE["dhm_oper"], edgecolor="white", lw=0.6, zorder=5)
    ax.add_patch(
        FancyArrowPatch(
            (-4.82, y3 + 0.07),
            (-0.18, y3 + 0.07),
            arrowstyle="-|>,head_length=3.6,head_width=2.0",
            color=PALETTE["dhm_oper"],
            lw=1.3,
            mutation_scale=1.0,
            zorder=4,
        )
    )
    ax.text(-2.5, y3 + 0.19, "Issued 5 d ahead → scored on predicted flood date", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=PALETTE["dhm_oper"])
    ax.scatter([3.0], [y3 - 0.12], s=24, marker="o", color="#9CA3AF", edgecolor="#4B5563", lw=0.55, zorder=4)
    ax.scatter([5.55], [y3 - 0.12], s=28, marker="x", color="#6B7280", lw=1.1, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            (3.15, y3 - 0.12),
            (5.40, y3 - 0.12),
            arrowstyle="-|>,head_length=3.2,head_width=1.7",
            color="#6B7280",
            lw=0.95,
            ls="--",
            mutation_scale=1.0,
            zorder=4,
        )
    )
    ax.text(4.35, y3 - 0.23, "Predicted after flood end\n(excluded)", ha="center", va="top", fontsize=5.7, color="#4B5563")

    y4 = 0.16
    ax.axhline(y4, color="#E2E8F0", lw=0.6, zorder=0)
    xs_ffm = np.linspace(-7.4, 4.6, 180)
    ys_ffm = y4 - 0.12 + 0.34 * np.exp(-0.5 * ((xs_ffm + 0.65) / 0.52) ** 2)
    tau_lvl = y4 + 0.08
    ax.plot(xs_ffm, ys_ffm, color=PALETTE["dhm_ffm"], lw=1.35, zorder=3)
    ax.plot([-7.4, 4.6], [tau_lvl, tau_lvl], color=PALETTE["dhm_ffm"], ls="--", lw=0.8, alpha=0.85, zorder=2)
    ax.fill_between(xs_ffm, tau_lvl, ys_ffm, where=(ys_ffm >= tau_lvl), color="#D8C2EB", alpha=0.65, zorder=2)
    ax.text(-0.65, y4 + 0.25, "Flash-flood probability ≥ 0.50 (0–24 h)", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color="#522B78")

    ax.set_yticks([y1, y2, y3, y4])
    ax.set_yticklabels(
        ["Official CAP alerts", "0-day river models", "0–7 day river forecasts", "0–24 h flash-flood model"],
        fontsize=6.9,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5])
    ax.set_xticklabels(["−7 d", "−5 d", "−3 d", "−1 d", "Onset", "+1 d", "End", "End+2 d"], fontsize=6.6)
    ax.set_xlabel("Days relative to recorded state- or province-level flood onset and end", fontsize=7.1)
    ax.grid(False)
    panel_label(ax, "a", x=-0.19, y=1.03)
    ax.set_title("Alert matching window and forecast-date alignment", fontsize=7.8, fontweight="bold", pad=6)


def _plot_fig3_pr_panel(
    ax: plt.Axes,
    df_pr_sub: pd.DataFrame,
    df_fine_sub: pd.DataFrame,
    title: str,
    letter: str,
    is_overlap: bool,
    is_cat_a: bool,
    show_ylabel: bool,
    show_xlabel: bool,
) -> dict[str, plt.Artist]:
    handles: dict[str, plt.Artist] = {}
    if not df_fine_sub.empty:
        pareto = df_fine_sub[
            (df_fine_sub["system"] == "DHM_Oper_U_FFM_Pareto_fine")
            & (df_fine_sub["tau"] <= 0.95)
            & (df_fine_sub["tau"] >= 0.10)
        ].sort_values("recall_pct")
        if not pareto.empty:
            (ln_p,) = ax.plot(
                pareto["recall_pct"],
                pareto["precision_pct"],
                color="#9D7CC1",
                linestyle=":",
                linewidth=1.25,
                alpha=0.90,
                zorder=3,
                label="Google DHM + FFM (2D Pareto ceiling)",
            )
            handles["Google DHM + FFM (2D Pareto ceiling)"] = ln_p

        q2_ffm = df_fine_sub[
            (df_fine_sub["system"] == "DHM_Oper_Q2_U_FFM_fine")
            & (df_fine_sub["tau"] <= 0.99)
            & (df_fine_sub["tau"] >= 0.10)
        ].sort_values("recall_pct")
        if not q2_ffm.empty:
            (ln_c,) = ax.plot(
                q2_ffm["recall_pct"],
                q2_ffm["precision_pct"],
                color=PALETTE["dhm_ffm"],
                linestyle="-",
                linewidth=1.9,
                zorder=6,
                label="Google DHM (Q2yr) + FFM (0–24 h)",
            )
            handles["Google DHM (Q2yr) + FFM (0–24 h)"] = ln_c
            pt50 = q2_ffm[q2_ffm["tau"].round(2) == 0.50]
            if not pt50.empty and np.isfinite(pt50["precision_pct"].iloc[0]):
                rx, py = float(pt50["recall_pct"].iloc[0]), float(pt50["precision_pct"].iloc[0])
                ax.scatter([rx], [py], s=44, marker="o", color=PALETTE["dhm_ffm"], edgecolor="white", linewidth=1.0, zorder=9)
                ax.scatter([rx], [py], s=44, marker="o", facecolor="none", edgecolor=PALETTE["ink"], linewidth=0.75, zorder=10)
                tx, ty = (rx + 2.0, 2.7 if not is_overlap else 2.8) if is_cat_a else (rx + 3.0, 12.5)
                ax.annotate(
                    f"DHM+FFM\n({rx:.1f}%, {py:.1f}%)",
                    xy=(rx, py),
                    xytext=(min(tx, 84.0), ty),
                    fontsize=5.7,
                    fontweight="bold",
                    color="#4C2273",
                    ha="center",
                    va="top",
                    arrowprops=dict(arrowstyle="-", color="#6A3D9A", lw=0.65),
                    zorder=11,
                )

    if is_overlap:
        river_specs = (
            ("DHM_0d", "Google DHM (0-day)", PALETTE["dhm_0d"], "--", "D", True, 1.4, 4),
            ("GeoGLOWS_Oper_HRES", "GeoGLOWS v2 control HRES (0–7 d)", PALETTE["geoglows_hres"], "-.", "^", True, 1.45, 5),
            ("GeoGLOWS_Oper_EnsMedian", "GeoGLOWS v2 ens. median (0–7 d)", PALETTE["geoglows_ens"], "-", "s", False, 1.75, 6),
            ("DHM_Oper", "Google DHM (0–7 day)", PALETTE["dhm_oper"], "-", "o", False, 2.0, 7),
        )
    else:
        river_specs = (
            ("GloFAS_v5_0d", "Copernicus GloFAS v5 (0-day)", PALETTE["glofas_0d"], "-.", "s", True, 1.4, 4),
            ("GeoGLOWS_Retro_0d", "GeoGLOWS v2 (0-day)", PALETTE["geoglows_0d"], ":", "^", True, 1.45, 4),
            ("DHM_0d", "Google DHM (0-day)", PALETTE["dhm_0d"], "--", "D", True, 1.5, 5),
            ("DHM_Oper", "Google DHM (0–7 day)", PALETTE["dhm_oper"], "-", "o", False, 2.0, 7),
        )

    for sys_k, lbl, col, ls, mk, hollow, lw, zo in river_specs:
        sub = df_pr_sub[df_pr_sub["system"] == sys_k].sort_values("recall_pct")
        if sub.empty:
            continue
        (ln,) = ax.plot(sub["recall_pct"], sub["precision_pct"], color=col, linestyle=ls, linewidth=lw, zorder=zo, label=lbl)
        handles[lbl] = ln
        other_pts = sub[sub["thr_label"].isin(["RP=25" if sys_k.startswith("GeoGLOWS_Oper_") else "RP=20", "RP=5"])]
        if not other_pts.empty:
            ax.scatter(
                other_pts["recall_pct"],
                other_pts["precision_pct"],
                s=10,
                marker="o",
                color="white" if hollow else col,
                edgecolor=col,
                linewidth=0.65,
                zorder=zo + 0.5,
            )
        pt2 = sub[sub["thr_label"] == "RP=2"]
        if not pt2.empty and np.isfinite(pt2["precision_pct"].iloc[0]):
            ax.scatter(
                [float(pt2["recall_pct"].iloc[0])],
                [float(pt2["precision_pct"].iloc[0])],
                s=36,
                marker=mk,
                facecolor="white" if hollow else col,
                edgecolor=col if hollow else PALETTE["ink"],
                linewidth=1.15 if hollow else 0.7,
                zorder=zo + 2,
            )

    for cap_k, lbl, col, mk in (
        ("CAP_Perm", "Official CAP (flood & storm)", PALETTE["cap_perm"], "^"),
        ("CAP_Rest", "Official CAP (flood-coded only)", PALETTE["cap_rest"], "v"),
    ):
        csub = df_pr_sub[df_pr_sub["system"] == cap_k].sort_values("recall_pct")
        if csub.empty:
            continue
        (ln,) = ax.plot(
            csub["recall_pct"],
            csub["precision_pct"],
            color=col,
            linestyle="-",
            linewidth=1.7,
            marker=mk,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=8,
            label=lbl,
        )
        handles[lbl] = ln
        all_pt = csub[csub["thr_label"] == "All"]
        if not all_pt.empty and np.isfinite(all_pt["precision_pct"].iloc[0]):
            rx, py = float(all_pt["recall_pct"].iloc[0]), float(all_pt["precision_pct"].iloc[0])
            ax.scatter([rx], [py], s=42, marker=mk, color=col, edgecolor=PALETTE["ink"], linewidth=0.75, zorder=10)
            if cap_k == "CAP_Perm":
                tx, ty, ha = (
                    ((38.5, 1.8, "center") if not is_overlap else (43.5, 1.9, "center"))
                    if is_cat_a
                    else ((45.0, 10.5, "center") if not is_overlap else (48.0, 10.5, "center"))
                )
                ax.annotate(
                    f"CAP ({rx:.1f}%, {py:.1f}%)",
                    xy=(rx, py),
                    xytext=(tx, ty),
                    fontsize=5.6,
                    fontweight="bold",
                    color=col,
                    ha=ha,
                    va="top",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.65),
                    zorder=11,
                )
            else:
                tx, ty = (rx + 5.5, py - 1.2) if is_cat_a else (rx + 4.5, py - 11.5)
                ax.annotate(
                    f"Flood-coded\n({rx:.1f}%)",
                    xy=(rx, py),
                    xytext=(min(tx, 85.0), max(0.9 if is_cat_a else 5.5, ty)),
                    fontsize=5.4,
                    color=col,
                    ha="left",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.55),
                    zorder=11,
                )

    ax.set_xlim(0, 100)
    if is_cat_a:
        ymax = 20 if not is_overlap else 22
        ax.set_ylim(0, ymax)
        ax.set_yticks(range(0, ymax + 1, 5))
    else:
        ax.set_ylim(0, 68)
        ax.set_yticks([0, 15, 30, 45, 60])
    ax.grid(True, alpha=0.55)
    ax.set_title(title, fontsize=7.4, fontweight="bold", pad=5)
    panel_label(ax, letter, x=-0.14 if show_ylabel else -0.09, y=1.03)
    if show_xlabel:
        ax.set_xlabel("Flood event coverage / recall (%)", fontsize=7.2)
    if show_ylabel:
        ax.set_ylabel("Alert precision (%)", fontsize=7.2)
    return handles


def render_fig3(df_pr: pd.DataFrame, df_fine: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Fig. 3: Scoring timeline (a) + decluttered 5-yr & 2024-25 operational PR panels (b-e)."""
    setup_nature_style()
    fig = plt.figure(figsize=(7.2, 7.65), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.98, 1.15, 1.15], hspace=0.42, wspace=0.18)

    ax_a = fig.add_subplot(gs[0, :])
    _draw_timeline_panel(ax_a)

    def _n_ev(cat_k: str, bm_k: str) -> int:
        s = df_pr[(df_pr["catalog"] == cat_k) & (df_pr["benchmark"] == bm_k) & (df_pr["lead_N"] == 7)]
        return int(s["n_events"].iloc[0]) if not s.empty else 0

    panels = (
        (1, 0, "Without_GS", "2021_2025", "b", f"Catalog A: Disaster registries (2021–2025; n = {_n_ev('Without_GS', '2021_2025'):,})", False, True),
        (1, 1, "With_GS", "2021_2025", "c", f"Catalog B: With local news (2021–2025; n = {_n_ev('With_GS', '2021_2025'):,})", False, False),
        (2, 0, "Without_GS", "2024_2025", "d", f"Catalog A: Overlap (Apr 2024–Dec 2025; n = {_n_ev('Without_GS', '2024_2025'):,})", True, True),
        (2, 1, "With_GS", "2024_2025", "e", f"Catalog B: Overlap (Apr 2024–Dec 2025; n = {_n_ev('With_GS', '2024_2025'):,})", True, False),
    )
    all_h: dict[str, plt.Artist] = {}
    for r_i, c_i, cat_k, bm_k, let, ttl, is_ov, is_a in panels:
        ax = fig.add_subplot(gs[r_i, c_i])
        pr_sub = df_pr[(df_pr["catalog"] == cat_k) & (df_pr["benchmark"] == bm_k) & (df_pr["lead_N"] == 7)]
        fn_sub = df_fine[(df_fine["catalog"] == cat_k) & (df_fine["benchmark"] == bm_k)] if not df_fine.empty else pd.DataFrame()
        all_h.update(
            _plot_fig3_pr_panel(
                ax,
                pr_sub,
                fn_sub,
                ttl,
                let,
                is_overlap=is_ov,
                is_cat_a=is_a,
                show_ylabel=(c_i == 0),
                show_xlabel=(r_i == 2),
            )
        )

    leg_order = (
        "Official CAP (flood & storm)",
        "Official CAP (flood-coded only)",
        "Google DHM (0–7 day)",
        "Google DHM (0-day)",
        "Google DHM (Q2yr) + FFM (0–24 h)",
        "Google DHM + FFM (2D Pareto ceiling)",
        "Copernicus GloFAS v5 (0-day)",
        "GeoGLOWS v2 (0-day)",
        "GeoGLOWS v2 ens. median (0–7 d)",
        "GeoGLOWS v2 control HRES (0–7 d)",
    )
    leg_h = [all_h[k] for k in leg_order if k in all_h]
    leg_l = [k for k in leg_order if k in all_h]
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.53, 0.004),
            frameon=True,
            facecolor="#FAFAFB",
            edgecolor="#D1D5DB",
            columnspacing=1.0,
            handlelength=2.0,
            fontsize=6.15,
        )
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.145, top=0.965, hspace=0.42, wspace=0.17)
    return save_publication_figure(fig, figures_dir, "fig-3")


def render_ed_fig2(df_pr: pd.DataFrame, df_fine: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Extended Data Fig. 2: CAP combined with individual global models (64-level CAP + fine FFM)."""
    setup_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.25), dpi=300, facecolor="white")

    def _n_ev(cat_k: str, bm_k: str) -> int:
        s = df_pr[(df_pr["catalog"] == cat_k) & (df_pr["benchmark"] == bm_k) & (df_pr["lead_N"] == 7)]
        return int(s["n_events"].iloc[0]) if not s.empty else 0

    panels = (
        (0, 0, "Without_GS", "2021_2025", "a", f"Catalog A (2021–2025; n = {_n_ev('Without_GS', '2021_2025'):,})", False, True),
        (0, 1, "With_GS", "2021_2025", "b", f"Catalog B (2021–2025; n = {_n_ev('With_GS', '2021_2025'):,})", False, False),
        (1, 0, "Without_GS", "2024_2025", "c", f"Catalog A (Apr 2024–Dec 2025; n = {_n_ev('Without_GS', '2024_2025'):,})", True, True),
        (1, 1, "With_GS", "2024_2025", "d", f"Catalog B (Apr 2024–Dec 2025; n = {_n_ev('With_GS', '2024_2025'):,})", True, False),
    )
    all_h: dict[str, plt.Artist] = {}
    for r_i, c_i, cat_k, bm_k, let, ttl, is_ov, is_a in panels:
        ax = axes[r_i, c_i]
        pr_sub = df_pr[(df_pr["catalog"] == cat_k) & (df_pr["benchmark"] == bm_k) & (df_pr["lead_N"] == 7)]
        fn_sub = df_fine[(df_fine["catalog"] == cat_k) & (df_fine["benchmark"] == bm_k)] if not df_fine.empty else pd.DataFrame()

        for cap_k, lbl, col, ls in (
            ("CAP_Perm_Cont", "Official CAP (64-level, flood & storm)", PALETTE["cap_perm"], "-"),
            ("CAP_Rest_Cont", "Official CAP (64-level, flood-coded)", PALETTE["cap_rest"], "--"),
        ):
            cs = pr_sub[pr_sub["system"] == cap_k].sort_values("recall_pct")
            if not cs.empty:
                (ln,) = ax.plot(cs["recall_pct"], cs["precision_pct"], color=col, ls=ls, lw=1.45, zorder=5, label=lbl)
                all_h[lbl] = ln

        comb_specs = [
            ("CAP_Perm_U_GloFAS_v5_0d", "CAP + GloFAS v5 (0-day)", PALETTE["glofas_0d"], "-.", "s", True, 1.35, 4),
            ("CAP_Perm_U_GeoGLOWS_Retro_0d", "CAP + GeoGLOWS v2 (0-day)", PALETTE["geoglows_0d"], ":", "^", True, 1.4, 4),
            ("CAP_Perm_U_DHM_0d", "CAP + Google DHM (0-day)", PALETTE["dhm_0d"], "--", "D", True, 1.45, 5),
            ("CAP_Perm_U_DHM_Oper", "CAP + Google DHM (0–7 day)", PALETTE["dhm_oper"], "-", "o", False, 1.8, 6),
        ]
        if is_ov:
            comb_specs.extend(
                [
                    ("CAP_Perm_U_GeoGLOWS_Oper_EnsMedian", "CAP + GeoGLOWS v2 ens. med. (0–7 d)", PALETTE["geoglows_ens"], "-", "s", False, 1.6, 6),
                    ("CAP_Perm_U_GeoGLOWS_Oper_HRES", "CAP + GeoGLOWS v2 HRES (0–7 d)", PALETTE["geoglows_hres"], "-.", "^", True, 1.45, 5),
                ]
            )
        for sys_k, lbl, col, ls, mk, hollow, lw, zo in comb_specs:
            sub = pr_sub[pr_sub["system"] == sys_k].sort_values("recall_pct")
            if sub.empty:
                continue
            (ln,) = ax.plot(sub["recall_pct"], sub["precision_pct"], color=col, ls=ls, lw=lw, zorder=zo, label=lbl)
            all_h[lbl] = ln
            pt2 = sub[sub["thr_label"] == "RP=2"]
            if not pt2.empty and np.isfinite(pt2["precision_pct"].iloc[0]):
                ax.scatter(
                    [float(pt2["recall_pct"].iloc[0])],
                    [float(pt2["precision_pct"].iloc[0])],
                    s=30,
                    marker=mk,
                    facecolor="white" if hollow else col,
                    edgecolor=col if hollow else PALETTE["ink"],
                    linewidth=1.0 if hollow else 0.65,
                    zorder=zo + 2,
                )

        if not fn_sub.empty:
            for fn_k, lbl, col, ls, lw, zo in (
                ("CAP_Perm_U_FFM_fine", "CAP + Google FFM (0–24 h)", PALETTE["ffm"], "--", 1.5, 6),
                ("CAP_Perm_U_DHM_Oper_Q2_U_FFM_fine", "CAP + DHM (Q2yr) + FFM", PALETTE["dhm_ffm"], "-", 2.0, 8),
            ):
                g = fn_sub[(fn_sub["system"] == fn_k) & (fn_sub["tau"] <= 0.95) & (fn_sub["tau"] >= 0.10)].sort_values("recall_pct")
                if g.empty:
                    continue
                (ln,) = ax.plot(g["recall_pct"], g["precision_pct"], color=col, ls=ls, lw=lw, zorder=zo, label=lbl)
                all_h[lbl] = ln
                pt50 = g[g["tau"].round(2) == 0.50]
                if not pt50.empty and np.isfinite(pt50["precision_pct"].iloc[0]):
                    rx, py = float(pt50["recall_pct"].iloc[0]), float(pt50["precision_pct"].iloc[0])
                    ax.scatter(
                        [rx],
                        [py],
                        s=42 if zo == 8 else 28,
                        marker="*" if zo == 8 else "o",
                        color=col,
                        edgecolor=PALETTE["ink"],
                        lw=0.65,
                        zorder=zo + 2,
                    )
                    if zo == 8:
                        ax.annotate(
                            f"({rx:.1f}%, {py:.1f}%)",
                            xy=(rx, py),
                            xytext=(rx - 3.0, py + (2.4 if is_a else 7.5)),
                            fontsize=5.8,
                            fontweight="bold",
                            color="#4C2273",
                            ha="center",
                            va="bottom",
                            arrowprops=dict(arrowstyle="-", color="#6A3D9A", lw=0.6),
                            zorder=11,
                        )

        ax.set_xlim(0, 100)
        if is_a:
            ax.set_ylim(0, 16)
            ax.set_yticks([0, 4, 8, 12, 16])
        else:
            ax.set_ylim(0, 56)
            ax.set_yticks([0, 15, 30, 45, 55])
        ax.grid(True, alpha=0.55)
        ax.set_title(ttl, fontsize=7.4, fontweight="bold", pad=5)
        panel_label(ax, let, x=-0.13 if c_i == 0 else -0.09, y=1.03)
        if r_i == 1:
            ax.set_xlabel("Flood event coverage / recall (%)", fontsize=7.2)
        if c_i == 0:
            ax.set_ylabel("Alert precision (%)", fontsize=7.2)

    leg_order = (
        "Official CAP (64-level, flood & storm)",
        "Official CAP (64-level, flood-coded)",
        "CAP + GloFAS v5 (0-day)",
        "CAP + GeoGLOWS v2 (0-day)",
        "CAP + Google DHM (0-day)",
        "CAP + Google DHM (0–7 day)",
        "CAP + GeoGLOWS v2 ens. med. (0–7 d)",
        "CAP + GeoGLOWS v2 HRES (0–7 d)",
        "CAP + Google FFM (0–24 h)",
        "CAP + DHM (Q2yr) + FFM",
    )
    leg_h = [all_h[k] for k in leg_order if k in all_h]
    leg_l = [k for k in leg_order if k in all_h]
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.53, 0.005),
            frameon=True,
            facecolor="#FAFAFB",
            edgecolor="#D1D5DB",
            columnspacing=0.85,
            handlelength=1.8,
            fontsize=5.9,
        )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.95, hspace=0.33, wspace=0.17)
    return save_publication_figure(fig, figures_dir, "ed-fig-2")


def render_ed_fig3(df_leads: pd.DataFrame, df_single: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Extended Data Fig. 3: Pre-flood window sweep (0-15 d) and single-lead-day skill decay (0-14 d)."""
    setup_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.95), dpi=300, facecolor="white")
    bm_lead = "2024_2025" if "2024_2025" in set(df_leads["benchmark"]) else "2021_2025"
    sub_l_a = df_leads[(df_leads["benchmark"] == bm_lead) & (df_leads["catalog"] == "Without_GS")]

    lead_specs = (
        ("CAP_Perm", "Official CAP (flood & storm)", PALETTE["cap_perm"], "-", "^", 1.7),
        ("GloFAS_v5_0d", "GloFAS v5 (0-day)", PALETTE["glofas_0d"], "-.", "s", 1.35),
        ("DHM_0d", "Google DHM (0-day)", PALETTE["dhm_0d"], "--", "D", 1.45),
        ("DHM_Oper", "Google DHM (0–7 day)", PALETTE["dhm_oper"], "-", "o", 1.85),
        ("GeoGLOWS_Oper_EnsMedian", "GeoGLOWS v2 ens. med. (0–7 d)", PALETTE["geoglows_ens"], "-", "s", 1.6),
        ("GeoGLOWS_Oper_HRES", "GeoGLOWS v2 HRES (0–7 d)", PALETTE["geoglows_hres"], "-.", "^", 1.45),
        ("DHM_Oper_U_FFM050", "Google DHM + FFM (Q2yr, τ = 0.50)", PALETTE["dhm_ffm"], "-", "*", 1.9),
    )

    h_top: dict[str, plt.Artist] = {}
    for col_i, (metric, ylabel, let, ttl, ymax) in enumerate(
        [
            ("recall_pct", "Event coverage / recall (%)", "a", "Catalog A coverage vs. pre-flood window (2024–2025)", 90),
            ("precision_pct", "Alert precision (%)", "b", "Catalog A precision vs. pre-flood window (2024–2025)", 18),
        ]
    ):
        ax = axes[0, col_i]
        ax.axvline(7, color="#94A3B8", ls=":", lw=0.95, zorder=1)
        ax.text(7.25, ymax * 0.96, "Baseline\n(7 d)", fontsize=5.8, color="#475569", va="top", ha="left")
        for sys_k, lbl, col, ls, mk, lw in lead_specs:
            g = sub_l_a[sub_l_a["system"] == sys_k].sort_values("lead_N")
            if g.empty:
                continue
            (ln,) = ax.plot(
                g["lead_N"],
                g[metric],
                color=col,
                ls=ls,
                lw=lw,
                marker=mk,
                markersize=3.5,
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=lbl,
                zorder=5,
            )
            h_top[lbl] = ln
        ax.set_xlim(-0.4, 15.4)
        ax.set_xticks([0, 3, 7, 10, 15])
        ax.set_xlabel("Cumulative pre-flood matching window (days)", fontsize=7.0)
        ax.set_ylabel(ylabel, fontsize=7.0)
        ax.set_ylim(0, ymax)
        if col_i == 1:
            ax.set_yticks([0, 4, 8, 12, 16])
        ax.set_title(ttl, fontsize=7.1, fontweight="bold", pad=5)
        ax.grid(True, alpha=0.55)
        panel_label(ax, let, x=-0.13, y=1.03)

    single_specs = (
        ("DHM_10D", "Google DHM (0–10 d)", PALETTE["dhm_oper"], "-", "o"),
        ("GeoGLOWS_EnsMedian", "GeoGLOWS v2 ens. med. (0–14 d)", PALETTE["geoglows_ens"], "-", "s"),
        ("GeoGLOWS_HRES", "GeoGLOWS v2 HRES (0–9 d)", PALETTE["geoglows_hres"], "-.", "^"),
    )
    for col_i, (cat_k, let, ttl, ymax) in enumerate(
        [
            ("Without_GS", "c", "Single-lead-day skill decay: Catalog A (Q2yr)", 50),
            ("With_GS", "d", "Single-lead-day skill decay: Catalog B (Q2yr)", 48),
        ]
    ):
        ax = axes[1, col_i]
        sub_s = df_single[
            (df_single["catalog"] == cat_k) & (df_single["benchmark"] == bm_lead) & (df_single["rp_tier"] == 2)
        ]
        for sys_k, lbl, col, ls, mk in single_specs:
            g = sub_s[sub_s["system"] == sys_k].sort_values("lead_day")
            if g.empty:
                continue
            ax.plot(
                g["lead_day"],
                g["recall_pct"],
                color=col,
                ls=ls,
                lw=1.8,
                marker=mk,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=f"{lbl} — recall",
                zorder=5,
            )
            ax.plot(
                g["lead_day"],
                g["precision_pct"],
                color=col,
                ls=":",
                lw=1.25,
                marker=mk,
                markersize=2.8,
                markerfacecolor="white",
                markeredgecolor=col,
                markeredgewidth=0.7,
                alpha=0.88,
                label=f"{lbl} — precision",
                zorder=4,
            )
        ax.set_xlim(-0.4, 14.4)
        ax.set_xticks(range(0, 15, 2))
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Forecast lead time (days before predicted flood date)", fontsize=7.0)
        ax.set_ylabel("Recall (solid) / Precision (dotted) (%)", fontsize=6.9)
        ax.set_title(ttl, fontsize=7.1, fontweight="bold", pad=5)
        ax.grid(True, alpha=0.55)
        panel_label(ax, let, x=-0.13, y=1.03)
        if col_i == 0:
            ax.text(
                13.8,
                ymax * 0.93,
                "Solid: event coverage\nDotted: alert precision",
                fontsize=5.8,
                color="#475569",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D1D5DB", lw=0.5),
            )

    leg_order = (
        "Official CAP (flood & storm)",
        "GloFAS v5 (0-day)",
        "Google DHM (0-day)",
        "Google DHM (0–7 day)",
        "GeoGLOWS v2 ens. med. (0–7 d)",
        "GeoGLOWS v2 HRES (0–7 d)",
        "Google DHM + FFM (Q2yr, τ = 0.50)",
    )
    leg_h = [h_top[k] for k in leg_order if k in h_top]
    leg_l = [k for k in leg_order if k in h_top]
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.53, 0.005),
            frameon=True,
            facecolor="#FAFAFB",
            edgecolor="#D1D5DB",
            columnspacing=0.9,
            handlelength=1.9,
            fontsize=6.0,
        )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.165, top=0.945, hspace=0.38, wspace=0.22)
    return save_publication_figure(fig, figures_dir, "ed-fig-3")


def render_ed_fig4(
    df_ffm_slices: pd.DataFrame,
    df_fine: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Extended Data Fig. 4: Standalone & combined FFM across Catalog A, Catalog B, and news-only (2x3)."""
    setup_nature_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.15), dpi=300, facecolor="white")

    col_defs = (
        ("Without_GS", "Catalog A (no news)", 22),
        ("With_GS", "Catalog B (all events)", 68),
        ("GS_Media_Only", "News-only events", 62),
    )
    row_defs = (
        ("2021_2025", "2021–2025", ["a", "b", "c"]),
        ("2024_2025", "Apr 2024–Dec 2025", ["d", "e", "f"]),
    )
    all_h: dict[str, plt.Artist] = {}
    for r_i, (bm_k, bm_lbl, lets) in enumerate(row_defs):
        for c_i, (cat_k, cat_lbl, ymax) in enumerate(col_defs):
            ax = axes[r_i, c_i]
            fn_sub = df_fine[(df_fine["catalog"] == cat_k) & (df_fine["benchmark"] == bm_k)] if not df_fine.empty else pd.DataFrame()
            sl_sub = df_ffm_slices[(df_ffm_slices["catalog"] == cat_k) & (df_ffm_slices["benchmark"] == bm_k)]
            n_ev = int(fn_sub["n_events"].iloc[0]) if not fn_sub.empty else (int(sl_sub["n_events"].iloc[0]) if not sl_sub.empty else 0)

            cap = sl_sub[sl_sub["system"] == "CAP_Perm"].sort_values("recall_pct")
            if not cap.empty:
                (ln,) = ax.plot(
                    cap["recall_pct"],
                    cap["precision_pct"],
                    color=PALETTE["cap_perm"],
                    ls="-",
                    lw=1.45,
                    marker="^",
                    markersize=3.4,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    label="Official CAP (flood & storm)",
                    zorder=5,
                )
                all_h["Official CAP (flood & storm)"] = ln

            dhm = sl_sub[sl_sub["system"] == "DHM_Oper"].sort_values("recall_pct")
            if not dhm.empty:
                (ln,) = ax.plot(
                    dhm["recall_pct"],
                    dhm["precision_pct"],
                    color=PALETTE["dhm_oper"],
                    ls="-",
                    lw=1.65,
                    label="Google DHM (0–7 day)",
                    zorder=6,
                )
                all_h["Google DHM (0–7 day)"] = ln
                pt2 = dhm[dhm["thr_label"] == "RP=2"]
                if not pt2.empty and np.isfinite(pt2["precision_pct"].iloc[0]):
                    ax.scatter(
                        [float(pt2["recall_pct"].iloc[0])],
                        [float(pt2["precision_pct"].iloc[0])],
                        s=26,
                        marker="o",
                        color=PALETTE["dhm_oper"],
                        edgecolor=PALETTE["ink"],
                        lw=0.6,
                        zorder=8,
                    )

            if not fn_sub.empty:
                for fk, lbl, col, ls, lw, zo in (
                    ("FFM_fine", "Google FFM standalone (0–24 h)", PALETTE["ffm"], "--", 1.55, 6),
                    ("CAP_Perm_U_FFM_fine", "CAP + Google FFM (0–24 h)", "#C026D3", ":", 1.45, 5),
                    ("DHM_Oper_Q2_U_FFM_fine", "Google DHM (Q2yr) + FFM", PALETTE["dhm_ffm"], "-", 1.9, 7),
                ):
                    g = fn_sub[(fn_sub["system"] == fk) & (fn_sub["tau"] <= 0.95) & (fn_sub["tau"] >= 0.10)].sort_values("recall_pct")
                    if g.empty:
                        continue
                    (ln,) = ax.plot(g["recall_pct"], g["precision_pct"], color=col, ls=ls, lw=lw, label=lbl, zorder=zo)
                    all_h[lbl] = ln
                    for tv, mk, sz in ((0.70, "P", 26), (0.50, "s", 26)):
                        pt = g[g["tau"].round(2) == tv]
                        if not pt.empty and np.isfinite(pt["precision_pct"].iloc[0]):
                            ax.scatter(
                                [float(pt["recall_pct"].iloc[0])],
                                [float(pt["precision_pct"].iloc[0])],
                                s=sz,
                                marker=mk,
                                color=col,
                                edgecolor="white" if mk == "s" else PALETTE["ink"],
                                lw=0.55,
                                zorder=zo + 2,
                            )

            ax.set_xlim(0, 100)
            ax.set_ylim(0, ymax)
            ax.grid(True, alpha=0.55)
            ax.set_title(f"{cat_lbl}\n{bm_lbl} (n = {n_ev:,})", fontsize=6.6, fontweight="bold", pad=4)
            panel_label(ax, lets[c_i], x=-0.16 if c_i == 0 else -0.11, y=1.03, fontsize=9.8)
            if r_i == 1:
                ax.set_xlabel("Event coverage (%)", fontsize=6.8)
            if c_i == 0:
                ax.set_ylabel("Alert precision (%)", fontsize=6.8)

    leg_order = (
        "Official CAP (flood & storm)",
        "Google DHM (0–7 day)",
        "Google FFM standalone (0–24 h)",
        "CAP + Google FFM (0–24 h)",
        "Google DHM (Q2yr) + FFM",
    )
    leg_h = [all_h[k] for k in leg_order if k in all_h]
    leg_l = [k for k in leg_order if k in all_h]
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.52, 0.005),
            frameon=True,
            facecolor="#FAFAFB",
            edgecolor="#D1D5DB",
            columnspacing=0.9,
            handlelength=1.9,
            fontsize=5.95,
        )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.925, hspace=0.40, wspace=0.22)
    return save_publication_figure(fig, figures_dir, "ed-fig-4")


# ---------------------------------------------------------------------------
# Conceptual diagram: Fig. 5 (flat Nature architectural vector PADM)
# ---------------------------------------------------------------------------


def _draw_node(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    kind: str = "factor",
    zorder: int = 4,
) -> dict[str, tuple[float, float]]:
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    if kind == "spine_blue":
        fc, ec, lw, tc, fw, fs = "#EAF2F8", "#1B4965", 1.15, "#0F2942", "bold", 6.0
    elif kind == "spine_teal":
        fc, ec, lw, tc, fw, fs = "#E7F3F0", "#1B6B5A", 1.15, "#0E3D33", "bold", 6.0
    elif kind == "spine_red":
        fc, ec, lw, tc, fw, fs = "#FAECEC", "#942E2E", 1.15, "#5A1818", "bold", 6.0
    elif kind == "mediator_auth":
        fc, ec, lw, tc, fw, fs = "#FFFFFF", "#5A758A", 0.78, PALETTE["ink"], "normal", 5.55
    elif kind == "mediator_pub":
        fc, ec, lw, tc, fw, fs = "#FEFCF7", "#A38246", 0.82, PALETTE["ink"], "normal", 5.65
    else:
        fc, ec, lw, tc, fw, fs = "#F8F9FB", "#94A3B8", 0.65, "#2D3748", "normal", 5.55

    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=zorder,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight=fw, color=tc, linespacing=1.20, zorder=zorder + 2)
    return {
        "N": (cx, y0 + h),
        "S": (cx, y0),
        "E": (x0 + w, cy),
        "W": (x0, cy),
        "C": (cx, cy),
    }


def _draw_bezier(
    ax: plt.Axes,
    p0: tuple[float, float],
    c1: tuple[float, float],
    c2: tuple[float, float],
    p3: tuple[float, float],
    polarity: str | None = "+",
    pol_pos: tuple[float, float] | None = None,
    color: str = "#475569",
    lw: float = 0.88,
    zorder: int = 3,
    halo_bg: str | None = None,
) -> None:
    path = MplPath([p0, c1, c2, p3], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>,head_length=3.6,head_width=2.0",
        color=color,
        linewidth=lw,
        mutation_scale=1.0,
        shrinkA=0.5,
        shrinkB=1.2,
        zorder=zorder,
    )
    if halo_bg:
        arrow.set_path_effects([pe.withStroke(linewidth=3.0, foreground=halo_bg), pe.Normal()])
    ax.add_patch(arrow)
    if polarity and pol_pos is not None:
        pcol = "#942E2E" if polarity == "−" else "#334155"
        ax.text(
            pol_pos[0],
            pol_pos[1],
            polarity,
            ha="center",
            va="center",
            fontsize=6.5 if polarity == "+" else 7.4,
            fontweight="bold",
            color=pcol,
            zorder=zorder + 3,
        )


def _draw_straight(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    polarity: str | None = "+",
    pol_offset: tuple[float, float] = (0.0, 0.14),
    color: str = "#475569",
    lw: float = 0.92,
    zorder: int = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            posA=p0,
            posB=p1,
            arrowstyle="-|>,head_length=3.8,head_width=2.1",
            color=color,
            linewidth=lw,
            mutation_scale=1.0,
            shrinkA=0.8,
            shrinkB=1.2,
            zorder=zorder,
        )
    )
    if polarity:
        mx = 0.52 * p1[0] + 0.48 * p0[0] + pol_offset[0]
        my = 0.52 * p1[1] + 0.48 * p0[1] + pol_offset[1]
        pcol = "#942E2E" if polarity == "−" else "#334155"
        ax.text(
            mx,
            my,
            polarity,
            ha="center",
            va="center",
            fontsize=6.5 if polarity == "+" else 7.6,
            fontweight="bold",
            color=pcol,
            zorder=zorder + 3,
        )


def render_fig5(figures_dir: str | Path) -> list[Path]:
    """Fig. 5: Editorial Nature conceptual diagram adapted from Lindell & Perry (2012)."""
    setup_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.05), dpi=300, facecolor="white")
    ax.set_xlim(0.0, 16.4)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")

    for x0, x1, label, bg in (
        (0.18, 6.45, "1   FORECAST & ALERT ISSUANCE", "#FAF7F2"),
        (6.68, 9.18, "2   DISSEMINATION", "#F8FAFC"),
        (9.41, 14.22, "3   PROTECTIVE ACTION DECISION", "#FDFCF9"),
        (14.45, 16.22, "4   OUTCOME", "#FAF7F7"),
    ):
        ax.add_patch(Rectangle((x0, 0.72), x1 - x0, 7.65, facecolor=bg, edgecolor="none", zorder=0))
        ax.plot([x0, x1], [8.37, 8.37], color="#CBD5E1", linewidth=0.75, zorder=1)
        ax.text((x0 + x1) / 2.0, 8.54, label, ha="center", va="bottom", fontsize=6.0, fontweight="bold", color="#475569")

    ax.add_patch(
        FancyBboxPatch(
            (0.24, 0.88),
            6.15,
            7.22,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor="none",
            edgecolor="#9E3A2B",
            linewidth=0.90,
            linestyle=(0, (4.0, 2.4)),
            zorder=2,
        )
    )
    ax.text(0.46, 7.88, "Scope of this study", ha="left", va="top", fontsize=6.2, fontweight="bold", color="#962E22", zorder=5)

    n_acc = _draw_node(ax, 2.85, 6.82, 2.25, 0.80, "Accuracy of\nflood forecast", "factor")
    n_sev = _draw_node(ax, 1.48, 5.42, 2.08, 0.80, "Severity of\nforecasted flood", "factor")
    n_fc = _draw_node(ax, 1.12, 4.02, 1.38, 0.72, "Flood\nforecast", "factor")
    n_awill = _draw_node(ax, 3.45, 4.02, 2.66, 0.84, "Authorities are willing\nto issue alert", "mediator_auth")
    n_acap = _draw_node(ax, 3.45, 2.75, 2.66, 0.84, "Authorities have capacity\nto issue alert", "mediator_auth")
    n_aknow = _draw_node(ax, 3.45, 1.50, 2.66, 0.84, "Authorities know how\nto issue alert", "mediator_auth")
    n_alert = _draw_node(ax, 5.68, 3.55, 1.16, 0.84, "Flood\nalert", "spine_blue")

    n_reach = _draw_node(ax, 7.93, 5.62, 2.38, 0.84, "Reach of alert\ndissemination channels", "factor")
    n_recv = _draw_node(ax, 7.93, 3.55, 1.98, 0.84, "Flood alerts\nreceived", "spine_blue")
    n_util = _draw_node(ax, 7.93, 2.18, 2.04, 0.78, "Utility of flood\nalert content", "factor")
    n_time = _draw_node(ax, 7.93, 1.16, 2.04, 0.72, "Timeliness of\nflood alert", "factor")

    n_pwill = _draw_node(ax, 10.88, 5.95, 2.46, 0.84, "People are willing\nto take action", "mediator_pub")
    n_pknow = _draw_node(ax, 10.88, 3.55, 2.46, 0.84, "People know how\nto take action", "mediator_pub")
    n_pcap = _draw_node(ax, 10.88, 1.48, 2.46, 0.84, "People have capacity\nto take action", "mediator_pub")
    n_act = _draw_node(ax, 13.30, 3.55, 1.62, 0.90, "Protective\nearly actions", "spine_teal")

    n_imp = _draw_node(ax, 15.34, 3.55, 1.46, 0.84, "Flood\nimpacts", "spine_red")
    n_exp = _draw_node(ax, 14.90, 6.72, 1.92, 0.78, "Experience\nof flooding", "factor")

    c_pos, c_spine, c_neg, c_fb = "#475569", "#1B4965", "#942E2E", "#64748B"

    _draw_bezier(ax, (n_acc["S"][0] + 0.30, n_acc["S"][1]), (3.40, 5.75), (3.60, 5.05), (n_awill["N"][0] + 0.18, n_awill["N"][1]), "+", (3.82, 4.72), c_pos)
    _draw_bezier(ax, (n_sev["S"][0] + 0.38, n_sev["S"][1]), (2.15, 4.82), (2.68, 4.72), (n_awill["N"][0] - 0.55, n_awill["N"][1]), "+", (2.60, 4.76), c_pos)
    _draw_straight(ax, n_fc["E"], n_awill["W"], None, color=c_pos)

    _draw_bezier(ax, n_awill["E"], (4.88, 4.02), (4.88, 3.74), (n_alert["W"][0], n_alert["W"][1] + 0.18), "+", (4.88, 4.10), c_pos)
    _draw_bezier(ax, n_acap["E"], (4.92, 2.75), (4.92, 3.30), (n_alert["W"][0], n_alert["W"][1] - 0.18), "+", (4.86, 2.86), c_pos)
    _draw_bezier(ax, n_aknow["E"], (5.45, 1.50), (5.65, 2.25), n_alert["S"], "+", (5.84, 2.82), c_pos)

    _draw_straight(ax, n_alert["E"], n_recv["W"], "+", (0.0, 0.18), c_spine, lw=1.12)
    _draw_straight(ax, n_reach["S"], n_recv["N"], "+", (0.17, 0.0), c_pos)

    _draw_bezier(ax, (n_recv["E"][0], n_recv["E"][1] + 0.22), (9.28, 4.42), (9.28, 5.95), n_pwill["W"], "+", (9.48, 6.12), c_pos)
    _draw_straight(ax, n_recv["E"], n_pknow["W"], "+", (0.0, 0.18), c_spine, lw=1.05)
    _draw_bezier(ax, n_util["E"], (9.30, 2.18), (9.30, 3.25), (n_pknow["W"][0], n_pknow["W"][1] - 0.20), "+", (9.38, 2.95), c_pos)
    _draw_bezier(ax, n_time["E"], (9.28, 1.16), (9.32, 1.48), n_pcap["W"], "+", (9.42, 1.24), c_pos)

    _draw_bezier(ax, n_pwill["E"], (13.00, 5.95), (13.30, 4.95), n_act["N"], "+", (13.52, 4.30), c_pos)
    _draw_straight(ax, n_pknow["E"], n_act["W"], "+", (0.0, 0.18), c_spine, lw=1.05)
    _draw_bezier(ax, n_pcap["E"], (13.00, 1.48), (13.30, 2.25), n_act["S"], "+", (13.52, 2.78), c_pos)
    _draw_straight(ax, n_act["E"], n_imp["W"], "−", (0.0, 0.20), c_neg, lw=1.22)

    _draw_bezier(ax, (n_acc["N"][0] + 0.35, n_acc["N"][1]), (4.80, 8.10), (9.15, 8.10), (n_pwill["N"][0] - 0.35, n_pwill["N"][1]), "+", (10.22, 6.62), c_fb)
    ax.text(
        7.93,
        7.82,
        "Trust & warning fatigue",
        ha="center",
        va="bottom",
        fontsize=5.6,
        style="italic",
        color="#475569",
        bbox=dict(boxstyle="square,pad=0.08", fc="#F8FAFC", ec="none"),
        zorder=5,
    )
    _draw_bezier(ax, (n_imp["N"][0] + 0.18, n_imp["N"][1]), (15.95, 4.95), (15.95, 5.85), (n_exp["S"][0] + 0.42, n_exp["S"][1]), "+", (15.95, 5.40), c_fb)
    _draw_bezier(ax, n_exp["W"], (13.15, 6.72), (12.15, 6.72), (n_pwill["N"][0] + 0.38, n_pwill["N"][1]), "+", (11.58, 6.62), c_fb)
    _draw_bezier(
        ax,
        (n_exp["S"][0] - 0.55, n_exp["S"][1]),
        (13.85, 5.20),
        (11.95, 4.75),
        (n_pknow["N"][0] + 0.35, n_pknow["N"][1]),
        "+",
        (11.50, 4.28),
        c_fb,
        zorder=5,
        halo_bg="#FDFCF9",
    )

    ly = 0.32
    ax.add_patch(FancyArrowPatch((0.25, ly), (0.95, ly), arrowstyle="-|>,head_length=3.6,head_width=2.0", color=c_pos, linewidth=0.95, mutation_scale=1.0))
    ax.text(1.06, ly, "+  Reinforcing (same direction)", ha="left", va="center", fontsize=5.8, color="#334155")
    ax.add_patch(FancyArrowPatch((4.55, ly), (5.25, ly), arrowstyle="-|>,head_length=3.6,head_width=2.0", color=c_neg, linewidth=1.15, mutation_scale=1.0))
    ax.text(5.36, ly, "−  Reducing (opposite direction)", ha="left", va="center", fontsize=5.8, color="#7F1D1D")
    ax.text(16.20, ly, "Adapted from Lindell & Perry (2012)", ha="right", va="center", fontsize=5.6, style="italic", color="#64748B")
    return save_publication_figure(fig, figures_dir, "fig-5")


# ---------------------------------------------------------------------------
# Validation & Stream-Gauge figures: ED Fig. 1, NEW ED Fig. 5
# ---------------------------------------------------------------------------


def render_ed_fig1(
    df_aud: pd.DataFrame,
    ed4_df: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Extended Data Fig. 1: Manual CAP audit driven dynamically by ed4_df (a) + spatial sensitivity (b)."""
    setup_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=300, facecolor="white")

    def _row(prefix: str) -> pd.Series:
        m = ed4_df[ed4_df["strategy"].astype(str).str.startswith(prefix)]
        if m.empty:
            raise ValueError(f"ed_table4_validation_summary missing strategy '{prefix}'")
        return m.iloc[0]

    r_c0, r_ex, r_in = _row("Country-Only"), _row("Exclusion"), _row("Inclusion")
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

    x_lo = max(60.0, min(83.0, min(ex_r_lo, in_r_lo) - 5.0))
    rm = np.linspace(max(0.02, (x_lo - 1.0) / 100.0), 1.03, 220)
    pm = np.linspace(0.02, 1.02, 220)
    rg, pg = np.meshgrid(rm, pm)
    fg = 2 * pg * rg / (pg + rg)
    levels = [0.20, 0.40, 0.60, 0.80, 0.90]
    cs = ax1.contour(rg * 100, pg * 100, fg, levels=levels, colors="#CBD5E1", linestyles=":", linewidths=0.75, zorder=1)
    lx = x_lo + 3.2
    manual_locs = []
    for lv in levels:
        r_f = lx / 100.0
        denom = 2.0 * r_f - lv
        if denom > 0.01:
            p_f = (lv * r_f / denom) * 100.0
            if 3.0 < p_f < 98.0:
                manual_locs.append((lx, p_f))
    if manual_locs:
        ax1.clabel(cs, inline=True, fontsize=5.7, fmt="F1 = %.2f", colors="#64748B", manual=manual_locs)

    pts = (
        (
            f"Country-only pairing\n({c0_p:.1f}% [{c0_lo:.1f}–{c0_hi:.1f}%] prec.)",
            100.0,
            c0_p,
            None,
            [[max(0.0, c0_p - c0_lo)], [max(0.0, c0_hi - c0_p)]],
            "^",
            PALETTE["cap_perm"],
            (min(95.4, x_lo + (100.0 - x_lo) * 0.72), max(16.0, c0_p + 10.0)),
            "right",
            "center",
        ),
        (
            f"Subnational exclusion filter\n({ex_p:.1f}% [{ex_lo:.1f}–{ex_hi:.1f}%] prec.; {ex_r:.1f}% recall)",
            ex_r,
            ex_p,
            [[max(0.0, ex_r - ex_r_lo)], [max(0.0, ex_r_hi - ex_r)]],
            [[max(0.0, ex_p - ex_lo)], [max(0.0, ex_hi - ex_p)]],
            "o",
            PALETTE["dhm_oper"],
            (min(94.8, ex_r - 4.8), min(70.0, ex_p + 4.5)),
            "right",
            "center",
        ),
        (
            f"Subnational inclusion filter\n({in_p:.1f}% [{in_lo:.1f}–{in_hi:.1f}%] prec.; {in_r:.1f}% recall)",
            in_r,
            in_p,
            [[max(0.0, in_r - in_r_lo)], [max(0.0, in_r_hi - in_r)]],
            [[max(0.0, in_p - in_lo)], [max(0.0, in_hi - in_p)]],
            "s",
            PALETTE["cat1"],
            (min(94.4, in_r - 2.6), max(72.0, in_p - 7.5)),
            "right",
            "top",
        ),
    )
    for lbl, rx, py, xerr, yerr, mk, col, txy, ha, va in pts:
        ax1.errorbar(
            rx,
            py,
            xerr=xerr,
            yerr=yerr,
            fmt=mk,
            color=col,
            ecolor=col,
            elinewidth=1.15,
            capsize=3.0,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=6,
        )
        ax1.annotate(
            lbl,
            xy=(rx, py),
            xytext=txy,
            fontsize=5.6,
            fontweight="bold",
            color=col,
            ha=ha,
            va=va,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.75),
            zorder=8,
        )

    ax1.set_xlim(x_lo, 101.8)
    ax1.set_ylim(0, 104)
    ax1.set_xlabel("Conditional recall within retained pool (%)", fontsize=7.0)
    ax1.set_ylabel("Classification / pairing precision (%)", fontsize=7.0)
    ax1.set_title(f"CAP keyword & location audit (n = {n_main} sample)", fontsize=7.3, fontweight="bold", pad=5)
    ax1.grid(True, alpha=0.45)
    panel_label(ax1, "a", x=-0.16, y=1.04)

    t_sp = df_aud[(df_aud["audit_type"] == "Spatial_Attribution") & (df_aud["lead_N"] == 7)]
    bar_labels = ["Broad CAP\nCatalog A", "Flood-coded\nCatalog A", "Broad CAP\nCatalog B", "Flood-coded\nCatalog B"]
    pairs = (("Without_GS", "perm"), ("Without_GS", "rest"), ("With_GS", "perm"), ("With_GS", "rest"))
    g1_rec, a0_rec, g1_prec, a0_prec = [], [], [], []
    for cat_p, kind_p in pairs:
        r1 = t_sp[(t_sp["catalog"] == cat_p) & (t_sp["variant"] == f"{kind_p}_strict_GID1")]
        r0 = t_sp[(t_sp["catalog"] == cat_p) & (t_sp["variant"] == f"{kind_p}_legacy_Admin0")]
        g1_rec.append(float(r1["recall_pct"].iloc[0]) if not r1.empty and np.isfinite(r1["recall_pct"].iloc[0]) else 0.0)
        a0_rec.append(float(r0["recall_pct"].iloc[0]) if not r0.empty and np.isfinite(r0["recall_pct"].iloc[0]) else 0.0)
        g1_prec.append(float(r1["precision_pct"].iloc[0]) if not r1.empty and np.isfinite(r1["precision_pct"].iloc[0]) else 0.0)
        a0_prec.append(float(r0["precision_pct"].iloc[0]) if not r0.empty and np.isfinite(r0["precision_pct"].iloc[0]) else 0.0)

    x = np.arange(len(bar_labels))
    w = 0.19
    ax2.bar(x - 1.5 * w, g1_rec, width=w, color=PALETTE["dhm_oper"], edgecolor="white", lw=0.4, label="State/province recall", zorder=3)
    ax2.bar(x - 0.5 * w, a0_rec, width=w, color="#9CC5E0", edgecolor=PALETTE["dhm_oper"], lw=0.5, hatch="///", label="Country-level recall", zorder=3)
    ax2.bar(x + 0.5 * w, g1_prec, width=w, color=PALETTE["geoglows_ens"], edgecolor="white", lw=0.4, label="State/province precision", zorder=3)
    ax2.bar(x + 1.5 * w, a0_prec, width=w, color="#F4C29F", edgecolor=PALETTE["geoglows_ens"], lw=0.5, hatch="///", label="Country-level precision", zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, fontsize=6.4)
    ymax_b = max(78.0, max(g1_rec + a0_rec + g1_prec + a0_prec, default=65.0) * 1.22)
    ax2.set_ylim(0, min(115.0, ymax_b))
    ax2.set_ylabel("Percentage (%)", fontsize=7.0)
    ax2.set_title("Spatial-unit sensitivity (Admin-1 vs. Admin-0)", fontsize=7.3, fontweight="bold", pad=5)
    ax2.grid(axis="y", alpha=0.55, zorder=0)
    ax2.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=5.6, columnspacing=0.7, handlelength=1.4)
    panel_label(ax2, "b", x=-0.15, y=1.04)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.89, wspace=0.25)
    return save_publication_figure(fig, figures_dir, "ed-fig-1")


def render_ed_fig5(
    t2_df: pd.DataFrame,
    ecdf_df: pd.DataFrame | None,
    iqr_df: pd.DataFrame | None,
    figures_dir: str | Path,
) -> list[Path]:
    """Extended Data Fig. 5: Global stream-gauge NSE ECDFs (a) & DHM precision vs. catalogs (b)."""
    setup_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.45), dpi=300, facecolor="white")

    # Panel a: NSE ECDFs from clean-room anonymized quantile summary
    ax1.axvline(0.0, color="#94A3B8", ls="--", lw=0.85, zorder=1)
    ax1.axhline(0.5, color="#CBD5E1", ls=":", lw=0.80, zorder=1)
    ax1.text(-0.04, 0.56, "NSE = 0\n(mean-flow baseline)", fontsize=5.5, color="#64748B", va="center", ha="right")

    style_map = {
        "dhm_shared_9756": (PALETTE["dhm_oper"], "-", 1.95, 6),
        "dhm_all_18047": (PALETTE["dhm_0d"], ":", 1.45, 5),
        "glofas_v5_shared_9756": (PALETTE["glofas_0d"], "--", 1.55, 4),
        "geoglows_v2_shared_9756": (PALETTE["geoglows_ens"], "-.", 1.55, 4),
    }
    if ecdf_df is not None and not ecdf_df.empty:
        for key in ("dhm_shared_9756", "dhm_all_18047", "glofas_v5_shared_9756", "geoglows_v2_shared_9756"):
            sub = ecdf_df[ecdf_df["series_key"] == key].sort_values("ecdf_prob")
            if sub.empty:
                continue
            col, ls, lw, zo = style_map[key]
            lbl_base = str(sub["series_label"].iloc[0])
            med = float(sub["median_nse"].iloc[0])
            pos_pct = float(sub["pct_positive_nse"].iloc[0])
            ax1.plot(
                sub["nse_value"],
                sub["ecdf_prob"],
                color=col,
                ls=ls,
                lw=lw,
                zorder=zo,
                label=f"{lbl_base}\n  med {med:.2f} ({pos_pct:.1f}% > 0)",
            )
            ax1.scatter([med], [0.5], s=26, color=col, edgecolor="white", lw=0.55, zorder=zo + 2)
        ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=5.35, labelspacing=0.45, handlelength=1.7)

    ax1.set_xlim(-1.0, 1.0)
    ax1.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.set_ylim(0.0, 1.02)
    ax1.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("Daily streamflow Nash–Sutcliffe Efficiency (NSE)", fontsize=7.0)
    ax1.set_ylabel("Cumulative share of stream gauges", fontsize=7.0)
    ax1.set_title("Global hydrograph skill across shared gauges", fontsize=7.3, fontweight="bold", pad=5)
    ax1.grid(True, alpha=0.45)
    panel_label(ax1, "a", x=-0.16, y=1.04)

    # Panel b: High-flow precision — physical gauges (median/IQR + mean) vs Catalog B & Catalog A
    rps = [rp for rp in (20, 10, 5, 2) if rp in set(t2_df["rp_yr"])]
    x = np.arange(len(rps), dtype=float)
    iqr_map = iqr_df.set_index("rp_yr").to_dict("index") if (iqr_df is not None and not iqr_df.empty) else {}
    t2_map = t2_df.set_index("rp_yr").to_dict("index")

    g_mean, g_med, g_q25, g_q75, g_n, cat_a_p, cat_b_p = [], [], [], [], [], [], []
    for rp in rps:
        rt = t2_map[rp]
        ri = iqr_map.get(rp, {})
        mn = float(ri.get("gauge_precision_mean_pct", rt["gauge_precision_mean_pct"]))
        md = float(ri.get("gauge_precision_median_pct", rt["gauge_precision_median_pct"]))
        q25 = float(ri.get("gauge_precision_q25_pct", max(0.0, md - 15.0)))
        q75 = float(ri.get("gauge_precision_q75_pct", min(100.0, md + 15.0)))
        g_mean.append(mn)
        g_med.append(md)
        g_q25.append(q25)
        g_q75.append(q75)
        g_n.append(int(rt["gauge_n"]))
        cat_a_p.append(float(rt["dhm_oper_without_gs_prec_pct"]) if np.isfinite(rt["dhm_oper_without_gs_prec_pct"]) else 0.0)
        cat_b_p.append(float(rt["dhm_oper_with_gs_prec_pct"]) if np.isfinite(rt["dhm_oper_with_gs_prec_pct"]) else 0.0)

    if len(rps) >= 2:
        gap_lo = round(min(m - a for m, a in zip(g_mean, cat_a_p)))
        gap_hi = round(max(m - a for m, a in zip(g_mean, cat_a_p)))
        ax2.fill_between(x, cat_a_p, g_mean, color="#F3F4F6", alpha=0.85, zorder=1)
        ax2.text(
            0.5 * (x[0] + x[-1]),
            21.0,
            f"Disaster-catalog\nreporting gap\n({gap_lo}–{gap_hi} pp)",
            ha="center",
            va="center",
            fontsize=5.6,
            style="italic",
            color="#4B5563",
            zorder=2,
        )

    for xi, q25, q75 in zip(x, g_q25, g_q75):
        ax2.plot([xi - 0.08, xi - 0.08], [q25, q75], color="#93C5DF", lw=3.6, solid_capstyle="round", zorder=3)
    ax2.plot(x - 0.08, g_med, color=PALETTE["dhm_oper"], ls="-", lw=1.6, marker="o", ms=5.0, mec="white", mew=0.5, label="Gauges (median & IQR)", zorder=5)
    ax2.plot(x - 0.08, g_mean, color=PALETTE["dhm_oper"], ls="--", lw=1.15, marker="D", ms=4.2, mfc="white", mec=PALETTE["dhm_oper"], mew=1.0, label="Gauges (mean)", zorder=6)
    ax2.plot(x + 0.08, cat_b_p, color=PALETTE["cat2"], ls="-", lw=1.75, marker="s", ms=4.8, mec="white", mew=0.5, label="Catalog B (with news)", zorder=6)
    ax2.plot(x + 0.08, cat_a_p, color=PALETTE["cap_perm"], ls="-", lw=1.75, marker="^", ms=5.2, mec="white", mew=0.5, label="Catalog A (disasters)", zorder=6)

    if len(rps) >= 2:
        halo = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.45)
        ax2.text(x[0] - 0.08, g_mean[0] + 4.8, f"mean {g_mean[0]:.1f}%", fontsize=5.2, color=PALETTE["dhm_oper"], ha="center", bbox=halo, zorder=7)
        ax2.text(x[-1] - 0.08, g_med[-1] + 4.8, f"med {g_med[-1]:.1f}%", fontsize=5.2, fontweight="bold", color=PALETTE["dhm_oper"], ha="center", bbox=halo, zorder=7)
        ax2.text(x[0] + 0.14, max(2.0, cat_b_p[0] - 5.5), f"{cat_b_p[0]:.1f}%", fontsize=5.3, fontweight="bold", color="#9A6B0C", ha="center")
        ax2.text(x[-1] + 0.14, max(2.0, cat_b_p[-1] - 5.5), f"{cat_b_p[-1]:.1f}%", fontsize=5.3, fontweight="bold", color="#9A6B0C", ha="center")
        ax2.text(x[0] + 0.14, max(1.5, cat_a_p[0] - 5.2), f"{cat_a_p[0]:.1f}%", fontsize=5.3, fontweight="bold", color=PALETTE["cap_perm"], ha="center")
        ax2.text(x[-1] + 0.14, max(1.5, cat_a_p[-1] - 5.2), f"{cat_a_p[-1]:.1f}%", fontsize=5.3, fontweight="bold", color=PALETTE["cap_perm"], ha="center")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Q{rp}yr\n(n = {n:,})" for rp, n in zip(rps, g_n)], fontsize=6.1)
    ax2.set_xlim(-0.48, max(0.5, len(rps) - 0.52))
    ax2.set_ylim(0, 94)
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.set_xlabel("Flood return-period threshold", fontsize=7.0)
    ax2.set_ylabel("Google DHM alert precision (%)", fontsize=7.0)
    ax2.set_title("Gauge-verified vs. catalog-scored precision", fontsize=7.3, fontweight="bold", pad=5)
    ax2.grid(axis="y", alpha=0.5)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="#E2E8F0",
        fontsize=5.2,
        columnspacing=0.9,
        handlelength=1.45,
        handletextpad=0.4,
    )
    panel_label(ax2, "b", x=-0.15, y=1.04)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.89, wspace=0.25)
    return save_publication_figure(fig, figures_dir, "ed-fig-5")
