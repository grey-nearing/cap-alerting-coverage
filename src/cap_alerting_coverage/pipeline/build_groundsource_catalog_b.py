"""Step 09 (Tier 2): Spatial-join Groundsource v2 (Zenodo 24553440) to GADM Admin-1 & build Catalog B."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from shapely import make_valid
from shapely.strtree import STRtree

from cap_alerting_coverage.clustering import PopRule, merge_strictly_nested_catalog_b
from cap_alerting_coverage.config import ProjectPaths


def spatial_map_points_to_layer(
    pts,
    target_gdf: gpd.GeoDataFrame,
    max_nearest_deg: float = 0.15,
) -> dict[int, int]:
    """Map representative points to GADM Admin-1 polygons (intersects + optional <=0.15° coastal nearest)."""
    tree = STRtree(target_gdf.geometry.values)
    mapped: dict[int, int] = {}
    pt_idx, tree_idx = tree.query(pts, predicate="intersects")
    for pt_i, g_i in zip(pt_idx, tree_idx):
        pi = int(pt_i)
        if pi not in mapped:
            mapped[pi] = int(g_i)

    if max_nearest_deg > 0:
        miss = [k for k in range(len(pts)) if k not in mapped]
        if miss:
            near_pairs, _ = tree.query_nearest(pts[miss], max_distance=max_nearest_deg, return_distance=True)
            if near_pairs.shape[1] > 0:
                seen_m: set[int] = set()
                for m_pos, g_pos in zip(near_pairs[0], near_pairs[1]):
                    m_int = int(m_pos)
                    if m_int not in seen_m:
                        seen_m.add(m_int)
                        mapped[miss[m_int]] = int(g_pos)
    return mapped


def map_groundsource_to_adm1_daily(
    groundsource_parquet: Path,
    adm1_gdf: gpd.GeoDataFrame,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Read Groundsource v2 GeoParquet (2021-2025) and reduce to de-identified GID_1 daily footprints."""
    tbl = pq.read_table(groundsource_parquet, columns=["flood_start_date", "flood_end_date", "geometry"])
    tbl = tbl.filter(
        pc.and_(
            pc.greater_equal(tbl["flood_start_date"], start_date),
            pc.less_equal(tbl["flood_start_date"], end_date),
        )
    )
    s_dts = pd.to_datetime(tbl["flood_start_date"].to_pandas(), errors="coerce").dt.floor("D")
    e_dts = pd.to_datetime(tbl["flood_end_date"].to_pandas(), errors="coerce").dt.floor("D").fillna(s_dts)
    e_dts = e_dts.where(e_dts >= s_dts, s_dts)

    geoms = shapely.from_wkb(tbl["geometry"].to_pandas().values)
    invalid_mask = ~shapely.is_valid(geoms)
    if invalid_mask.any():
        geoms[invalid_mask] = make_valid(geoms[invalid_mask])
    pts = shapely.point_on_surface(geoms)

    mapped = spatial_map_points_to_layer(pts, adm1_gdf, max_nearest_deg=0.15)
    adm1_gids = adm1_gdf["gid_1"].values
    adm1_isos = adm1_gdf["country_iso3"].values
    sel = sorted(mapped.keys())
    df_mapped = pd.DataFrame(
        {
            "gid_1": [adm1_gids[mapped[k]] for k in sel],
            "country_iso3": [adm1_isos[mapped[k]] for k in sel],
            "start_date": s_dts.iloc[sel].values,
            "end_date": e_dts.iloc[sel].values,
        }
    )
    df_daily = df_mapped.groupby(["gid_1", "country_iso3", "start_date"], as_index=False)["end_date"].max()
    df_daily["has_idmc"] = False
    df_daily["has_montandon"] = False
    df_daily["has_groundsource"] = True
    df_daily["displacements"] = 0.0
    df_daily["num_affected"] = 0.0
    df_daily["num_killed"] = 0.0
    return df_daily


def build_catalog_b_from_groundsource(
    paths: ProjectPaths,
    groundsource_parquet: Path,
    pop_rule: PopRule = "priority_fallback",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild strictly nested Catalog B (150-country and 185-jurisdiction variants) from Groundsource v2."""
    sup1 = pd.read_csv(paths.supplementary_data_1_csv, keep_default_na=False)
    sup1_all_185 = set(sup1["ISO3"].str.strip())
    cohort_d_35 = set(sup1[sup1["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"].str.strip())
    eval_150 = sup1_all_185 - cohort_d_35

    adm1 = gpd.read_parquet(paths.work_dir / "gadm41_adm1.parquet")
    adm1["geometry"] = [make_valid(g) for g in adm1.geometry]
    cat_a_all = pd.read_parquet(paths.work_dir / "catalog_without_gs_all_185_2021_2025.parquet")

    df_gs_daily = map_groundsource_to_adm1_daily(groundsource_parquet, adm1)
    df_gs_daily.to_parquet(paths.work_dir / "groundsource_gid1_daily_2021_2025.parquet", index=False)

    cat_b_all, _ = merge_strictly_nested_catalog_b(
        cat_a_all, df_gs_daily, cohort_d_35, eval_150, sup1_all_185, pop_rule=pop_rule
    )
    cat_b_eval = cat_b_all[cat_b_all["in_eval_150"]].reset_index(drop=True)
    cat_b_eval.to_parquet(paths.work_dir / "catalog_with_gs_2021_2025.parquet", index=False)
    cat_b_all.to_parquet(paths.work_dir / "catalog_with_gs_all_185_2021_2025.parquet", index=False)
    return cat_b_eval, cat_b_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strictly nested Catalog B from Groundsource v2 (Tier 2).")
    parser.add_argument(
        "--groundsource-parquet", required=True, help="Path to groundsource.parquet (doi:10.5281/zenodo.24553440)."
    )
    parser.add_argument("--work-dir", default=None, help="Working directory containing GADM and Catalog A.")
    parser.add_argument("--pop-rule", choices=["priority_fallback", "max"], default="priority_fallback")
    args = parser.parse_args()

    paths = ProjectPaths.from_env(work_dir=args.work_dir)
    cat_b_eval, cat_b_all = build_catalog_b_from_groundsource(
        paths, Path(args.groundsource_parquet), pop_rule=args.pop_rule
    )
    print(f"Saved Catalog B: eval_150={len(cat_b_eval):,}, all={len(cat_b_all):,}")


if __name__ == "__main__":
    main()
