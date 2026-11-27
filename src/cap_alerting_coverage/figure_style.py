"""Shared Nature publication styling, Robinson cartography, and PDF/SVG/PNG export helpers."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from cap_alerting_coverage.config import ISO_UNIFY

# Cohesive, colorblind-safe Nature editorial palette
PALETTE: dict[str, str] = {
    "ink": "#1B2430",
    "ink_muted": "#4B5563",
    "ink_light": "#6B7280",
    "spine": "#374151",
    "grid": "#E5E7EB",
    "bg_subtle": "#F8FAFC",
    "no_data": "#EEF0F3",
    "excluded_fill": "#D4D8DF",
    "excluded_edge": "#6B7280",
    "border_map": "#525B66",
    # CAP capability categories (Fig. 1)
    "cat1": "#1B7A68",
    "cat2": "#D99B26",
    "cat3": "#C44E36",
    # Models & alerting systems (Figs. 3, ED 2-5)
    "cap_perm": "#B91C1C",
    "cap_rest": "#6B1111",
    "dhm_oper": "#006BA4",
    "dhm_0d": "#009E73",
    "dhm_ffm": "#6A3D9A",
    "ffm": "#8C56A8",
    "geoglows_ens": "#D55E00",
    "geoglows_hres": "#E69F00",
    "geoglows_0d": "#C4622D",
    "glofas_0d": "#7A5C43",
}

COVERAGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nature_cov",
    ["#F9F4E8", "#D7E8D4", "#8EC4BA", "#3B8EA5", "#1C5D85", "#0E3456"],
    N=256,
)

GAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nature_gain",
    ["#F8FAFC", "#D9EDF4", "#8ECAE0", "#3D92C2", "#1A5F96", "#0B2E59"],
    N=256,
)


def setup_nature_style() -> None:
    """Configure matplotlib for Nature house style with editable vector text."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "axes.edgecolor": PALETTE["spine"],
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.45,
        }
    )


def save_publication_figure(fig: plt.Figure, out_dir: str | Path, stem: str) -> list[Path]:
    """Save a single composite figure as vector PDF, vector SVG, and 300-DPI PNG."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        p = dest / f"{stem}.{ext}"
        kw: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.05, "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(p, **kw)
        written.append(p)
    plt.close(fig)
    return written


def panel_label(ax: plt.Axes, letter: str, x: float = -0.06, y: float = 1.04, fontsize: float = 10.5) -> None:
    """Place a crisp bold lowercase Nature panel label."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color=PALETTE["ink"],
        va="bottom",
        ha="left",
    )


def load_world_robinson(geojson_path: str | Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load Natural Earth countries (excluding Antarctica) projected to Robinson + globe frame."""
    world = gpd.read_file(geojson_path)
    iso_col = "ISO_A3_EH" if "ISO_A3_EH" in world.columns else "ISO_A3"
    world["iso3"] = world[iso_col].astype(str).str.strip().replace(ISO_UNIFY)
    if "ADM0_A3" in world.columns:
        bad = world["iso3"].isin(["-99", "", "nan"])
        world.loc[bad, "iso3"] = world.loc[bad, "ADM0_A3"].astype(str).str.strip().replace(ISO_UNIFY)
    if "NAME" in world.columns:
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
                if "NAME" in guf_row.index:
                    guf_row["NAME"] = "French Guiana"
                if "ISO_A3" in guf_row.index:
                    guf_row["ISO_A3"] = "GUF"
                if "ADM0_A3" in guf_row.index:
                    guf_row["ADM0_A3"] = "GUF"
                guf_row["iso3"] = "GUF"
                guf_row["geometry"] = MultiPolygon(guf_parts) if len(guf_parts) > 1 else guf_parts[0]
                world = pd.concat([world, gpd.GeoDataFrame([guf_row], crs=world.crs)], ignore_index=True)

    world = world[world["iso3"] != "ATA"].copy()
    world_rob = world.to_crs("+proj=robin")

    lons_top = np.linspace(-180, 180, 181)
    lats_right = np.linspace(84, -58, 91)
    lons_bot = np.linspace(180, -180, 181)
    lats_left = np.linspace(-58, 84, 91)
    ring = (
        [(x, 84.0) for x in lons_top]
        + [(180.0, y) for y in lats_right]
        + [(x, -58.0) for x in lons_bot]
        + [(-180.0, y) for y in lats_left]
    )
    frame_gdf = gpd.GeoDataFrame(geometry=[Polygon(ring)], crs="EPSG:4326").to_crs("+proj=robin")
    return world_rob, frame_gdf


def style_robinson_ax(ax: plt.Axes, frame_gdf: gpd.GeoDataFrame) -> None:
    """Draw a clean subtle ocean tint and hairline Robinson globe outline."""
    frame_gdf.plot(ax=ax, facecolor="#F7FAFC", edgecolor="#CBD5E1", linewidth=0.55, zorder=0)
    b = frame_gdf.total_bounds
    pad_x = (b[2] - b[0]) * 0.012
    pad_y = (b[3] - b[1]) * 0.018
    ax.set_xlim(b[0] - pad_x, b[2] + pad_x)
    ax.set_ylim(b[1] - pad_y, b[3] + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
