"""Step 03d (Tier 2): Portable Copernicus GloFAS v5 fsspec/Zarr extractor with Bulletin 17C LP3."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import make_valid
from shapely.strtree import STRtree

from cap_alerting_coverage.config import D_TOTAL, RETURN_PERIODS, ProjectPaths
from cap_alerting_coverage.stats_utils import fit_lp3_return_periods

EXPECTED_CHUNK_BYTES = 16617 * 10 * 10 * 4


def _read_uri_bytes(base_uri: str, *rel_parts: str) -> bytes:
    try:
        import fsspec
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Optional Tier-2 dependency 'fsspec' is required. Install with: pip install -e '.[tier2]'"
        ) from exc
    uri = base_uri.rstrip("/") + "/" + "/".join(rel_parts)
    with fsspec.open(uri, "rb") as fh:
        return fh.read()


def score_glofas_pixel_timeseries(
    discharge_matrix: np.ndarray,
    yr_indices: list[np.ndarray],
    idx_target: np.ndarray,
    min_mean_flow_m3s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter pixels to 1981-2025 mean_Q >= 10 m^3/s, fit LP3 return periods, and return tiers.

    Args:
        discharge_matrix: Array ``(n_hist_days, n_pixels)`` of daily streamflow (m^3/s).
        yr_indices: List of 45 annual day-index arrays (1981..2025).
        idx_target: Target day-index array covering 2020-12-15..2025-12-31.
        min_mean_flow_m3s: Minimum historical mean annual flow threshold.

    Returns:
        ``(keep_mask, tiers)`` where ``keep_mask`` is boolean ``(n_pixels,)`` and ``tiers`` is
        ``uint8`` of shape ``(len(idx_target), n_kept_pixels)`` with values in ``0..8``.
    """
    arr_hist = discharge_matrix[: int(idx_target[-1]) + 1, :]
    mean_q = np.nanmean(arr_hist, axis=0)
    keep = np.isfinite(mean_q) & (mean_q >= min_mean_flow_m3s)
    if not np.any(keep):
        return keep, np.zeros((len(idx_target), 0), dtype=np.uint8)

    arr_k = discharge_matrix[:, keep]
    ann_max = np.empty((len(yr_indices), arr_k.shape[1]), dtype=np.float64)
    for i, yidx in enumerate(yr_indices):
        sub_y = arr_k[yidx, :]
        has_fin = np.isfinite(sub_y).any(axis=0)
        ann_max[i, :] = np.where(has_fin, np.nanmax(np.where(np.isfinite(sub_y), sub_y, -np.inf), axis=0), np.nan)

    q_rp = fit_lp3_return_periods(ann_max, return_periods=RETURN_PERIODS)
    valid_rp = np.where(np.isfinite(q_rp) & (q_rp > 0), q_rp, np.inf)
    sim_t = arr_k[idx_target, :]
    tiers_rk = np.sum(sim_t[:, :, None] >= valid_rp[None, :, :], axis=-1).astype(np.uint8)
    return keep, tiers_rk


def extract_glofas_v5_to_adm1(
    zarr_uri: str,
    adm1_parquet: Path,
    out_dir: Path,
    max_workers: int = 32,
) -> Path:
    """Stream GloFAS v5 Zarr chunks via fsspec, fit 1981-2025 LP3 thresholds, and aggregate to GID_1."""
    try:
        import zstandard
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Optional Tier-2 dependency 'zstandard' is required. Install with: pip install -e '.[tier2]'"
        ) from exc

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    lats = np.frombuffer(dctx.decompress(_read_uri_bytes(zarr_uri, "latitude", "c", "0")), dtype="<f8")
    lons = np.frombuffer(dctx.decompress(_read_uri_bytes(zarr_uri, "longitude", "c", "0")), dtype="<f8")
    vts = np.frombuffer(dctx.decompress(_read_uri_bytes(zarr_uri, "valid_time", "c", "0")), dtype="<i8")

    gf_dates = pd.to_datetime("1981-01-02") + pd.to_timedelta(vts, unit="D")
    yrs = gf_dates.year.values
    yr_indices = [np.where(yrs == y)[0] for y in range(1981, 2026)]
    idx_target = np.where((gf_dates >= "2020-12-15") & (gf_dates <= "2025-12-31"))[0]
    target_dates = gf_dates[idx_target]

    adm1 = gpd.read_parquet(adm1_parquet)
    adm1_geoms = np.array([make_valid(g) for g in adm1.geometry.values], dtype=object)
    adm1_gids = adm1["gid_1"].values
    adm1_isos = adm1["country_iso3"].values
    n_adm1 = len(adm1_gids)

    map_cache = out_dir / "glofas_pixel_gid1_map.npz"
    if map_cache.exists():
        mc = np.load(map_cache, allow_pickle=False)
        all_lat_c, all_lon_c, all_pix_off, all_adm1 = mc["lat_c"], mc["lon_c"], mc["pix_off"], mc["adm1_idx"]
    else:
        tree = STRtree(adm1_geoms)
        bounds = np.array([g.bounds for g in adm1_geoms])
        lat_c_list, lon_c_list, off_list, adm_list = [], [], [], []
        for lc in range(len(lats) // 10):
            r0, r1 = lc * 10, (lc + 1) * 10
            if not np.any((bounds[:, 3] >= float(lats[r1 - 1]) - 0.05) & (bounds[:, 1] <= float(lats[r0]) + 0.05)):
                continue
            grid_lon, grid_lat = np.meshgrid(lons, lats[r0:r1])
            pts = shapely.points(grid_lon.ravel(), grid_lat.ravel())
            pi, gi = tree.query(pts, predicate="intersects")
            if len(pi) == 0:
                continue
            _, first_u = np.unique(pi, return_index=True)
            pi_u, gi_u = pi[first_u], gi[first_u]
            local_r, local_c = pi_u // len(lons), pi_u % len(lons)
            lat_c_list.append(np.full(len(pi_u), lc, dtype=np.int16))
            lon_c_list.append((local_c // 10).astype(np.int16))
            off_list.append((local_r * 10 + (local_c % 10)).astype(np.uint8))
            adm_list.append(gi_u.astype(np.int32))
        all_lat_c = np.concatenate(lat_c_list)
        all_lon_c = np.concatenate(lon_c_list)
        all_pix_off = np.concatenate(off_list)
        all_adm1 = np.concatenate(adm_list)
        np.savez_compressed(map_cache, lat_c=all_lat_c, lon_c=all_lon_c, pix_off=all_pix_off, adm1_idx=all_adm1)

    ckey = all_lat_c.astype(np.int32) * 1000 + all_lon_c.astype(np.int32)
    ord_k = np.argsort(ckey, kind="mergesort")
    all_pix_off, all_adm1, ckey = all_pix_off[ord_k], all_adm1[ord_k], ckey[ord_k]
    uniq_ck, ck_st, ck_cnt = np.unique(ckey, return_index=True, return_counts=True)
    tasks = [
        (int(ck // 1000), int(ck % 1000), all_pix_off[st : st + cnt], all_adm1[st : st + cnt])
        for ck, st, cnt in zip(uniq_ck, ck_st, ck_cnt)
    ]

    def _score_chunk(task):
        lat_c, lon_c, pix_offsets, pix_adm1 = task
        try:
            comp = _read_uri_bytes(zarr_uri, "avg_dis", "c", "0", str(lat_c), str(lon_c))
        except Exception:
            return None
        if len(comp) < 800:
            return None
        raw = zstandard.ZstdDecompressor().decompress(comp, max_output_size=EXPECTED_CHUNK_BYTES)
        if len(raw) != EXPECTED_CHUNK_BYTES:
            return None
        arr = np.frombuffer(raw, dtype="<f4").reshape(16617, 100)[:, pix_offsets]
        keep, tiers_rk = score_glofas_pixel_timeseries(arr, yr_indices, idx_target)
        if tiers_rk.shape[1] == 0:
            return None
        adm1_k = pix_adm1[keep]
        u_gids, inv = np.unique(adm1_k, return_inverse=True)
        block_max = np.zeros((len(u_gids), D_TOTAL), dtype=np.uint8)
        for u in range(len(u_gids)):
            block_max[u] = tiers_rk[:, inv == u].max(axis=1)
        return u_gids, block_max

    glofas_tiers = np.zeros((n_adm1, D_TOTAL), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_score_chunk, tasks, chunksize=8):
            if res is None:
                continue
            u_gids, block_max = res
            glofas_tiers[u_gids] = np.maximum(glofas_tiers[u_gids], block_max)

    out_npz = out_dir / "glofas_v5_gid1_daily_2021_2025.npz"
    np.savez_compressed(
        out_npz,
        tiers=glofas_tiers,
        gid_1=np.asarray(adm1_gids, dtype=str),
        country_iso3=np.asarray(adm1_isos, dtype=str),
        valid_dates=np.asarray(target_dates.strftime("%Y-%m-%d").values, dtype=str),
    )
    print(f"Saved GloFAS v5 Admin-1 cube to {out_npz} in {time.time() - t0:.1f}s.", flush=True)
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Copernicus GloFAS v5 Admin-1 tiers via fsspec (Tier 2).")
    parser.add_argument(
        "--zarr-uri", default=os.environ.get("CAP_GAP_GLOFAS_V5_ZARR_URI", ""), help="fsspec URI to GloFAS v5 Zarr."
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()
    if not args.zarr_uri:
        raise RuntimeError("Set --zarr-uri or CAP_GAP_GLOFAS_V5_ZARR_URI.")
    paths = ProjectPaths.from_env(work_dir=args.work_dir)
    extract_glofas_v5_to_adm1(
        args.zarr_uri, paths.work_dir / "gadm41_adm1.parquet", paths.work_dir, max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
