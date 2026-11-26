"""Step 03c (Tier 2): Portable Distributed Hydrological Model (DHM) fsspec/Zarr Admin-1 extractor."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import make_valid
from shapely.strtree import STRtree

from cap_alerting_coverage.config import D_TOTAL, RETURN_PERIODS, ProjectPaths

T_START_2020_12_15 = 1810
T_END_2025_12_31 = 3653
EXPECTED_CHUNK_BYTES = 2 * 3653 * 11 * 4 * 4
HALF_CHUNK_BYTES = 1 * 3653 * 11 * 4 * 4


def _require_tier2_mod(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Optional Tier-2 dependency '{name}' is required. Install with: pip install -e '.[tier2]'"
        ) from exc


def _read_uri_bytes(base_uri: str, *rel_parts: str) -> bytes:
    fsspec = _require_tier2_mod("fsspec")
    uri = base_uri.rstrip("/") + "/" + "/".join(rel_parts)
    with fsspec.open(uri, "rb") as fh:
        return fh.read()


def build_dhm_reach_tasks(
    zarr_10d_uri: str,
    hydro_atlas_parquet: Path,
    return_periods_parquet: Path,
    zarr_gauge_ids_npy: Path,
    subbasins_geoparquet: Path,
    adm1_parquet: Path,
    out_dir: Path,
    min_mean_flow_m3s: float = 10.0,
) -> Path:
    """Map HydroBASINS L12 reaches (>=10 m^3/s) strictly to GADM Admin-1 and persist chunk task index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_time = _read_uri_bytes(zarr_10d_uri, "time", "c", "0")
    zstandard = _require_tier2_mod("zstandard")
    t_arr = np.frombuffer(zstandard.ZstdDecompressor().decompress(comp_time), dtype="<i8")
    dts = pd.to_datetime("2016-01-01") + pd.to_timedelta(t_arr, unit="D")
    t_mask = (dts >= "2020-12-15") & (dts <= "2025-12-31")
    t_idx = np.where(t_mask)[0]
    issue_dates = np.asarray([str(d.date()) for d in dts[t_idx]], dtype=str)

    z_gids = np.load(zarr_gauge_ids_npy, allow_pickle=False)
    gid_to_zidx = {str(g): int(i) for i, g in enumerate(z_gids)}

    ha = pd.read_parquet(hydro_atlas_parquet, columns=["HYBAS_ID", "UP_AREA", "dis_m3_pyr"])
    ha = ha[ha["dis_m3_pyr"] >= min_mean_flow_m3s].copy()
    ha["HYBAS_ID"] = ha["HYBAS_ID"].astype(str)
    ha["z_idx"] = ha["HYBAS_ID"].map(gid_to_zidx)
    ha = ha[ha["z_idx"].notna()].copy()
    ha["z_idx"] = ha["z_idx"].astype(np.int32)

    rp_df = pd.read_parquet(return_periods_parquet)[list(RETURN_PERIODS)]
    rp_aligned = rp_df.reindex(ha["HYBAS_ID"].values).to_numpy(dtype=np.float64)
    rp_clean = np.where(np.isfinite(rp_aligned) & (rp_aligned > 0), rp_aligned, np.inf).astype(np.float32)
    ha = ha[np.any(np.isfinite(rp_clean), axis=1)].copy()

    active_set = set(ha["HYBAS_ID"].values)
    sub_gdf = gpd.read_parquet(subbasins_geoparquet, columns=["HYBAS_ID", "geometry"])
    sub_ids = "subhybas_" + sub_gdf["HYBAS_ID"].astype(str)
    sub_active = sub_gdf[sub_ids.isin(active_set).values].copy()
    sub_active["gauge_id"] = sub_ids[sub_ids.isin(active_set).values].values
    sub_active = sub_active.drop_duplicates(subset=["gauge_id"])
    reps = sub_active.geometry.representative_point()
    sub_active["rep_lon"] = reps.x.values
    sub_active["rep_lat"] = reps.y.values

    ha = ha.merge(sub_active[["gauge_id", "rep_lon", "rep_lat"]], left_on="HYBAS_ID", right_on="gauge_id", how="inner")
    rp_mat = np.where(
        np.isfinite(rp_df.reindex(ha["HYBAS_ID"].values).to_numpy(dtype=np.float64))
        & (rp_df.reindex(ha["HYBAS_ID"].values).to_numpy(dtype=np.float64) > 0),
        rp_df.reindex(ha["HYBAS_ID"].values).to_numpy(dtype=np.float64),
        np.inf,
    ).astype(np.float32)

    adm1 = gpd.read_parquet(adm1_parquet)
    adm1_geoms = np.array([make_valid(g) for g in adm1.geometry.values], dtype=object)
    tree = STRtree(adm1_geoms)
    pts = shapely.points(ha["rep_lon"].values, ha["rep_lat"].values)
    pi, gi = tree.query(pts, predicate="intersects")
    adm1_idx = np.full(len(ha), -1, dtype=np.int32)
    _, first_u = np.unique(pi, return_index=True)
    adm1_idx[pi[first_u]] = gi[first_u].astype(np.int32)

    miss_mask = adm1_idx < 0
    if np.any(miss_mask):
        miss_indices = np.where(miss_mask)[0]
        pt_hits = tree.nearest(pts[miss_indices])
        for k, r_i in enumerate(miss_indices):
            ai = int(pt_hits[k])
            if adm1_geoms[ai].distance(pts[r_i]) <= 0.15:
                adm1_idx[r_i] = ai

    keep_geo = adm1_idx >= 0
    ha = ha[keep_geo].copy().reset_index(drop=True)
    rp_mat = rp_mat[keep_geo]
    adm1_idx = adm1_idx[keep_geo]

    ha["adm1_idx"] = adm1_idx
    ha["gid_1"] = adm1["gid_1"].values[adm1_idx]
    ha["country_iso3"] = adm1["country_iso3"].values[adm1_idx]
    ha["c_idx"] = (ha["z_idx"].values // 2).astype(np.int32)
    ha["c_rem"] = (ha["z_idx"].values % 2).astype(np.uint8)
    ord_i = np.lexsort((ha["c_rem"].values, ha["c_idx"].values))
    ha = ha.iloc[ord_i].reset_index(drop=True)
    rp_mat = rp_mat[ord_i]

    tasks_npz = out_dir / "dhm_chunk_tasks.npz"
    np.savez_compressed(
        tasks_npz,
        c_idx=ha["c_idx"].values.astype(np.int32),
        c_rem=ha["c_rem"].values.astype(np.uint8),
        adm1_idx=ha["adm1_idx"].values.astype(np.int32),
        rp_mat=rp_mat,
        gid_1=np.asarray(adm1["gid_1"].values, dtype=str),
        country_iso3=np.asarray(adm1["country_iso3"].values, dtype=str),
        issue_dates=issue_dates,
    )
    return tasks_npz


def stream_dhm_10d_to_adm1(
    zarr_10d_uri: str,
    tasks_npz: Path,
    out_dir: Path,
    max_workers: int = 32,
) -> Path:
    """Stream DHM 10-day 4-member ensemble Zarr via fsspec into Admin-1 exceedance tiers."""
    t0 = time.time()
    tk = np.load(tasks_npz, allow_pickle=False)
    c_idx, c_rem, adm1_idx, rp_mat = tk["c_idx"], tk["c_rem"], tk["adm1_idx"], tk["rp_mat"]
    gid_1, country_iso3, issue_dates = tk["gid_1"], tk["country_iso3"], tk["issue_dates"]

    uniq_c, c_st, c_cnt = np.unique(c_idx, return_index=True, return_counts=True)
    tasks = [
        (int(uc), c_rem[st : st + cnt], adm1_idx[st : st + cnt], rp_mat[st : st + cnt])
        for uc, st, cnt in zip(uniq_c, c_st, c_cnt)
    ]

    def _score_chunk(task):
        uc, rems, a_idxs, rps = task
        try:
            comp = _read_uri_bytes(zarr_10d_uri, "streamflow", "c", str(uc), "0", "0", "0")
        except Exception:
            return None
        zstandard = _require_tier2_mod("zstandard")
        raw = zstandard.ZstdDecompressor().decompress(comp, max_output_size=EXPECTED_CHUNK_BYTES)
        if len(raw) == EXPECTED_CHUNK_BYTES:
            arr = np.frombuffer(raw, dtype="<f4").reshape(2, 3653, 11, 4)
        elif len(raw) == HALF_CHUNK_BYTES:
            arr = np.frombuffer(raw, dtype="<f4").reshape(1, 3653, 11, 4)
        else:
            return None
        sub = arr[rems, T_START_2020_12_15:T_END_2025_12_31, :, :]
        if not np.isnan(sub).any():
            s = np.sort(sub, axis=-1)
            med = 0.5 * (s[..., 1] + s[..., 2])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(sub, axis=-1)
        tiers = np.sum(med[..., None] >= rps[:, None, None, :], axis=-1).astype(np.uint8)
        return a_idxs, tiers

    dhm_tiers = np.zeros((len(gid_1), D_TOTAL, 11), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_score_chunk, tasks, chunksize=16):
            if res is None:
                continue
            a_idxs, tiers = res
            for k, ai in enumerate(a_idxs):
                dhm_tiers[ai] = np.maximum(dhm_tiers[ai], tiers[k])

    out_npz = out_dir / "dhm_10d_gid1_multilead_2021_2025.npz"
    np.savez_compressed(
        out_npz,
        tiers=dhm_tiers,
        gid_1=np.asarray(gid_1, dtype=str),
        country_iso3=np.asarray(country_iso3, dtype=str),
        issue_dates=np.asarray(issue_dates, dtype=str),
    )
    print(f"Saved DHM 10D Admin-1 cube to {out_npz} in {time.time() - t0:.1f}s.", flush=True)
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Google DHM 10-day Admin-1 exceedance cubes via fsspec (Tier 2)."
    )
    parser.add_argument(
        "--zarr-10d-uri", default=os.environ.get("CAP_GAP_DHM_ZARR_10D_URI", ""), help="fsspec URI to DHM 10D Zarr."
    )
    parser.add_argument(
        "--tasks-npz", default=None, help="Existing dhm_chunk_tasks.npz (skip reach mapping if provided)."
    )
    parser.add_argument("--hydro-atlas-parquet", default=os.environ.get("CAP_GAP_HYDRO_ATLAS_PARQUET", ""))
    parser.add_argument("--return-periods-parquet", default=os.environ.get("CAP_GAP_DHM_RP_PARQUET", ""))
    parser.add_argument("--zarr-gauge-ids-npy", default=os.environ.get("CAP_GAP_DHM_GAUGE_IDS_NPY", ""))
    parser.add_argument("--subbasins-geoparquet", default=os.environ.get("CAP_GAP_HYDROBASINS_GPQ", ""))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    if not args.zarr_10d_uri:
        raise RuntimeError("Set --zarr-10d-uri or CAP_GAP_DHM_ZARR_10D_URI.")
    paths = ProjectPaths.from_env(work_dir=args.work_dir)
    if args.tasks_npz:
        tasks_p = Path(args.tasks_npz)
    else:
        tasks_p = build_dhm_reach_tasks(
            args.zarr_10d_uri,
            Path(args.hydro_atlas_parquet),
            Path(args.return_periods_parquet),
            Path(args.zarr_gauge_ids_npy),
            Path(args.subbasins_geoparquet),
            paths.work_dir / "gadm41_adm1.parquet",
            paths.work_dir,
        )
    stream_dhm_10d_to_adm1(args.zarr_10d_uri, tasks_p, paths.work_dir, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
