"""Step 03a (Tier 2): Portable Flash Flood Model (0-24h) Zarr -> GADM Admin-1 probability extractor."""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import Point
from shapely.strtree import STRtree

from cap_alerting_coverage.config import ProjectPaths
from cap_alerting_coverage.pipeline.blosc_decode import decode_blosc


def decode_vlen_utf8(raw_bytes: bytes) -> list[str]:
    """Decode Zarr vlen-utf8 string array payload."""
    n_items = struct.unpack_from("<I", raw_bytes, 0)[0]
    pos = 4
    out: list[str] = []
    for _ in range(n_items):
        length = struct.unpack_from("<I", raw_bytes, pos)[0]
        pos += 4
        out.append(raw_bytes[pos : pos + length].decode("utf-8"))
        pos += length
    return out


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


def extract_ffm_to_adm1(
    zarr_uri: str,
    sites_parquet: Path,
    adm1_parquet: Path,
    out_dir: Path,
    min_prob_to_store: float = 0.05,
) -> pd.DataFrame:
    """Extract FFM 0-24h daily maximum flash-flood probabilities per GADM Admin-1 unit."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(_read_uri_bytes(zarr_uri, ".zmetadata").decode("utf-8"))["metadata"]
    f_meta = meta["forecast/.zarray"]
    n_pix, n_time = f_meta["shape"]
    ch_r, ch_c = f_meta["chunks"]

    t_sec = np.frombuffer(decode_blosc(_read_uri_bytes(zarr_uri, "forecast_time", "0")), dtype="<f8")
    all_dates = pd.to_datetime(t_sec, unit="s", utc=True).tz_localize(None).floor("D")
    t_mask = (all_dates >= "2020-12-15") & (all_dates <= "2025-12-31")
    t_idxs = np.where(t_mask)[0]
    keep_dates = all_dates[t_idxs]
    col_range = np.arange(int(t_idxs[0]) // ch_c, int(t_idxs[-1]) // ch_c + 1)

    area_ids = decode_vlen_utf8(decode_blosc(_read_uri_bytes(zarr_uri, "area_id", "0")))
    area_to_zidx = {aid: i for i, aid in enumerate(area_ids)}

    adm1 = gpd.read_parquet(adm1_parquet)
    adm1_geoms = np.array([make_valid(g) for g in adm1.geometry.values], dtype=object)
    adm1_gids = adm1["gid_1"].values
    adm1_isos = adm1["country_iso3"].values
    n_gids = len(adm1_gids)
    tree = STRtree(adm1_geoms)

    sites = pd.read_parquet(sites_parquet, columns=["pixel_name", "min_lat", "min_lon"])
    sites = sites[sites["pixel_name"].isin(area_to_zidx)].copy().reset_index(drop=True)
    sites["z_idx"] = sites["pixel_name"].map(area_to_zidx).astype(np.int32)
    sites["cen_lon"] = sites["min_lon"].astype(np.float64) + 0.1
    sites["cen_lat"] = sites["min_lat"].astype(np.float64) + 0.1
    pts = [Point(xy) for xy in zip(sites["cen_lon"], sites["cen_lat"])]

    pt_idx, tree_idx = tree.query(pts, predicate="intersects")
    mapped_gid_idx = np.full(len(sites), -1, dtype=np.int32)
    seen: set[int] = set()
    for pi, gi in zip(pt_idx, tree_idx):
        pii = int(pi)
        if pii not in seen:
            seen.add(pii)
            mapped_gid_idx[pii] = int(gi)

    miss = np.where(mapped_gid_idx < 0)[0]
    if len(miss) > 0:
        near_pairs, _ = tree.query_nearest([pts[i] for i in miss], max_distance=0.15, return_distance=True)
        if near_pairs.shape[1] > 0:
            seen_m: set[int] = set()
            for mp, gp in zip(near_pairs[0], near_pairs[1]):
                mpi = int(mp)
                if mpi not in seen_m:
                    seen_m.add(mpi)
                    mapped_gid_idx[miss[mpi]] = int(gp)

    sites["gid_idx"] = mapped_gid_idx
    sites_valid = sites[sites["gid_idx"] >= 0].copy()
    sites_valid["gid_1"] = adm1_gids[sites_valid["gid_idx"].values]
    sites_valid["country_iso3"] = adm1_isos[sites_valid["gid_idx"].values]
    sites_valid[["pixel_name", "z_idx", "cen_lon", "cen_lat", "gid_1", "country_iso3"]].to_parquet(
        out_dir / "ffm_active_pixels_gid1.parquet", index=False
    )

    zidx_to_gid_idx = np.full(n_pix, -1, dtype=np.int32)
    zidx_to_gid_idx[sites_valid["z_idx"].values] = sites_valid["gid_idx"].values

    day_lookup = {int(t): pos for pos, t in enumerate(t_idxs)}
    gid_daily_max = np.full((n_gids, len(t_idxs)), -1.0, dtype=np.float32)

    n_rc = (n_pix + ch_r - 1) // ch_r
    for rc in range(n_rc):
        r0 = rc * ch_r
        r1 = min(n_pix, r0 + ch_r)
        pix_gid = zidx_to_gid_idx[r0:r1]
        active_local = np.where(pix_gid >= 0)[0]
        if len(active_local) == 0:
            continue
        uniq_gids, inv_gids = np.unique(pix_gid[active_local], return_inverse=True)
        for cc in col_range:
            c0 = int(cc) * ch_c
            c1 = min(n_time, c0 + ch_c)
            local_cols, keep_pos = [], []
            for lc in range(c1 - c0):
                gc = c0 + lc
                if gc in day_lookup:
                    local_cols.append(lc)
                    keep_pos.append(day_lookup[gc])
            if not local_cols:
                continue
            raw = decode_blosc(_read_uri_bytes(zarr_uri, "forecast", f"{rc}.{cc}"))
            arr = np.frombuffer(raw, dtype="<f2").reshape(ch_r, ch_c)[: r1 - r0, :]
            sub = arr[active_local][:, local_cols].astype(np.float32)
            sub = np.where(np.isnan(sub), -1.0, sub)
            block_max = np.full((len(uniq_gids), len(local_cols)), -1.0, dtype=np.float32)
            np.maximum.at(block_max, inv_gids, sub)
            kp_arr = np.array(keep_pos, dtype=np.int32)
            gid_daily_max[np.ix_(uniq_gids, kp_arr)] = np.maximum(gid_daily_max[np.ix_(uniq_gids, kp_arr)], block_max)

    gi, di = np.where(gid_daily_max >= min_prob_to_store)
    df_out = pd.DataFrame(
        {
            "gid_1": adm1_gids[gi],
            "country_iso3": adm1_isos[gi],
            "valid_date": keep_dates.values[di],
            "max_prob": np.round(gid_daily_max[gi, di], 4),
        }
    )
    df_out.to_parquet(out_dir / "ffm_gid1_daily_2021_2025.parquet", index=False)
    print(f"Extracted {len(df_out):,} FFM Admin-1 days in {time.time() - t0:.1f}s.", flush=True)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FFM 0-24h daily Admin-1 exceedance probabilities (Tier 2).")
    parser.add_argument(
        "--zarr-uri", default=os.environ.get("CAP_GAP_FFM_ZARR_URI", ""), help="Local or fsspec URI to FFM global.zarr."
    )
    parser.add_argument(
        "--sites-parquet",
        default=os.environ.get("CAP_GAP_FFM_SITES_PARQUET", ""),
        help="Path to FFM pixel sites Parquet.",
    )
    parser.add_argument("--work-dir", default=None, help="Working directory containing gadm41_adm1.parquet.")
    args = parser.parse_args()
    if not args.zarr_uri or not args.sites_parquet:
        raise RuntimeError(
            "Provide --zarr-uri and --sites-parquet (or CAP_GAP_FFM_ZARR_URI / CAP_GAP_FFM_SITES_PARQUET)."
        )
    paths = ProjectPaths.from_env(work_dir=args.work_dir)
    extract_ffm_to_adm1(args.zarr_uri, Path(args.sites_parquet), paths.work_dir / "gadm41_adm1.parquet", paths.work_dir)


if __name__ == "__main__":
    main()
