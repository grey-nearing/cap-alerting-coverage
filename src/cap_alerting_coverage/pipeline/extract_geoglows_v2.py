"""Step 03b (Tier 2): Portable GeoGLOWS v2 retrospective (S3 Zarr) & operational multi-lead extractor."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from cap_alerting_coverage.config import D_TOTAL, ProjectPaths
from cap_alerting_coverage.pipeline.blosc_decode import decode_blosc

DEFAULT_GEOGLOWS_S3_RETRO_URL = "https://geoglows-v2.s3-us-west-2.amazonaws.com/retrospective/daily.zarr"
PRODUCTS: tuple[str, ...] = ("EnsMedian", "EnsMean", "EnsP75", "HRES")
N_OPER_DAYS = 640

_tls = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_tls, "sess", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128)
        sess.mount("https://", adapter)
        _tls.sess = sess
    return sess


def fetch_s3_with_retry(url: str, max_retries: int = 5) -> bytes:
    """Fetch a public GeoGLOWS S3 Zarr chunk with exponential backoff."""
    sess = _get_session()
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = sess.get(url, timeout=45)
            if resp.status_code == 200:
                return resp.content
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Failed fetching {url} after {max_retries} attempts: {last_err}")


def build_operational_multilead(
    shards_dir: Path,
    adm1_gids: np.ndarray,
    adm1_isos: np.ndarray,
    meta_to_adm1: np.ndarray,
    out_dir: Path,
    n_shards: int = 8,
) -> Path:
    """Assemble GeoGLOWS v2 operational multi-lead NPZ & Parquet (Apr 2024 - Dec 2025)."""
    n_adm1 = len(adm1_gids)
    dfs = []
    for i in range(n_shards):
        shard_p = shards_dir / f"geoglows_v2_forecast_202404_202512-{i:05d}-of-{n_shards:05d}.csv"
        dfs.append(
            pd.read_csv(
                shard_p,
                header=None,
                names=["day_idx", "gid_idx", "tiers"],
                dtype={"day_idx": np.int32, "gid_idx": np.int32, "tiers": str},
            )
        )
    df = pd.concat(dfs, ignore_index=True)
    df["adm1_idx"] = meta_to_adm1[df["gid_idx"].values]
    df = df[df["adm1_idx"] >= 0].reset_index(drop=True)

    joined = "".join(df["tiers"].tolist()).encode("ascii")
    raw_u8 = np.frombuffer(joined, dtype=np.uint8).reshape(len(df), 4, 15, 2) - 48
    parsed = (raw_u8[:, :, :, 0] * 10 + raw_u8[:, :, :, 1]).astype(np.uint8)

    oper_tiers = np.zeros((n_adm1, N_OPER_DAYS, 4, 15), dtype=np.uint8)
    np.maximum.at(oper_tiers, (df["adm1_idx"].values, df["day_idx"].values), parsed)

    issue_dates = pd.date_range("2024-04-01", periods=N_OPER_DAYS, freq="D")
    oper_npz = out_dir / "geoglows_v2_oper_gid1_multilead_2024_2025.npz"
    np.savez_compressed(
        oper_npz,
        tiers=oper_tiers,
        gid_1=np.asarray(adm1_gids, dtype=str),
        country_iso3=np.asarray(adm1_isos, dtype=str),
        issue_dates=np.asarray(issue_dates.strftime("%Y-%m-%d").values, dtype=str),
        products=np.asarray(PRODUCTS, dtype=str),
    )
    return oper_npz


def extract_retrospective_from_s3(
    meta_npz: Path,
    adm1_gids: np.ndarray,
    adm1_isos: np.ndarray,
    meta_to_adm1: np.ndarray,
    out_dir: Path,
    s3_base: str = DEFAULT_GEOGLOWS_S3_RETRO_URL,
    max_workers: int = 64,
    wave_size: int = 3000,
) -> Path:
    """Stream GeoGLOWS v2 retrospective daily discharge from public AWS S3 into Admin-1 tiers."""
    meta_z = np.load(meta_npz, allow_pickle=False)
    cids = meta_z["chunk_ids"]
    offs = meta_z["chunk_offsets"]
    cols = meta_z["cols"]
    gids = meta_z["gids"]
    rp8 = meta_z["rp8"]

    global_idx_list, reach_adm1_list, rp8_list = [], [], []
    for ci, cid in enumerate(cids):
        sl = slice(offs[ci], offs[ci + 1])
        a_idxs = meta_to_adm1[gids[sl]]
        keep = a_idxs >= 0
        if not np.any(keep):
            continue
        global_idx_list.append(int(cid) * 686 + cols[sl][keep].astype(np.int64))
        reach_adm1_list.append(a_idxs[keep])
        rp8_list.append(rp8[sl][keep])

    all_gidx = np.concatenate(global_idx_list)
    all_adm1 = np.concatenate(reach_adm1_list)
    all_rp8 = np.concatenate(rp8_list, axis=0)
    order = np.argsort(all_gidx, kind="mergesort")
    all_gidx, all_adm1, all_rp8 = all_gidx[order], all_adm1[order], all_rp8[order]

    s3_chunks = all_gidx // 50
    col50 = (all_gidx % 50).astype(np.int32)
    uniq_c50, start_pos, counts = np.unique(s3_chunks, return_index=True, return_counts=True)
    tasks = [
        (int(c), col50[st : st + cnt], all_adm1[st : st + cnt], all_rp8[st : st + cnt])
        for c, st, cnt in zip(uniq_c50, start_pos, counts)
    ]

    def _process_chunk(task):
        c50, cols50, reach_adm1, reach_rp8 = task
        b0 = fetch_s3_with_retry(f"{s3_base}/Q/0.{c50}")
        b1 = fetch_s3_with_retry(f"{s3_base}/Q/1.{c50}")
        arr0 = np.frombuffer(decode_blosc(b0), dtype="<f4").reshape(31047, 50)[29569:, cols50]
        arr1 = np.frombuffer(decode_blosc(b1), dtype="<f4").reshape(31047, 50)[:365, cols50]
        q = np.concatenate([arr0, arr1], axis=0)
        valid_rp8 = np.where(np.isfinite(reach_rp8) & (reach_rp8 > 0), reach_rp8, np.inf)
        tiers_rk = np.sum(q[:, :, None] >= valid_rp8[None, :, :], axis=-1).astype(np.uint8)
        u_gids, inv = np.unique(reach_adm1, return_inverse=True)
        block_max = np.zeros((len(u_gids), D_TOTAL), dtype=np.uint8)
        for k in range(len(u_gids)):
            block_max[k] = tiers_rk[:, inv == k].max(axis=1)
        return u_gids, block_max

    retro_tiers = np.zeros((len(adm1_gids), D_TOTAL), dtype=np.uint8)
    n_waves = (len(tasks) + wave_size - 1) // wave_size
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for w in range(n_waves):
            w_tasks = tasks[w * wave_size : (w + 1) * wave_size]
            futs = [ex.submit(_process_chunk, t) for t in w_tasks]
            for fut in as_completed(futs):
                u_gids, block_max = fut.result()
                retro_tiers[u_gids] = np.maximum(retro_tiers[u_gids], block_max)

    retro_dates = pd.date_range("2020-12-15", periods=D_TOTAL, freq="D")
    retro_npz = out_dir / "geoglows_v2_retro_gid1_daily_2021_2025.npz"
    np.savez_compressed(
        retro_npz,
        tiers=retro_tiers,
        gid_1=np.asarray(adm1_gids, dtype=str),
        country_iso3=np.asarray(adm1_isos, dtype=str),
        valid_dates=np.asarray(retro_dates.strftime("%Y-%m-%d").values, dtype=str),
    )
    return retro_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GeoGLOWS v2 retrospective & operational Admin-1 cubes (Tier 2)."
    )
    parser.add_argument(
        "--meta-npz", default=os.environ.get("CAP_GAP_GEOGLOWS_META_NPZ", ""), help="Reach-to-GID_1 metadata NPZ."
    )
    parser.add_argument(
        "--oper-shards-dir",
        default=os.environ.get("CAP_GAP_GEOGLOWS_SHARDS_DIR", ""),
        help="Directory containing operational forecast shards.",
    )
    parser.add_argument("--work-dir", default=None, help="Working directory with gadm41_adm1.parquet.")
    parser.add_argument("--skip-retro", action="store_true", help="Skip S3 retrospective streaming.")
    args = parser.parse_args()

    paths = ProjectPaths.from_env(work_dir=args.work_dir)
    adm1 = pd.read_parquet(paths.work_dir / "gadm41_adm1.parquet", columns=["gid_1", "country_iso3"])
    adm1_gids = adm1["gid_1"].values
    adm1_isos = adm1["country_iso3"].values
    gid_to_idx = {str(g): i for i, g in enumerate(adm1_gids)}

    meta_p = Path(args.meta_npz) if args.meta_npz else (paths.work_dir / "geoglows_forecast_chunk_meta.npz")
    meta_z = np.load(meta_p, allow_pickle=False)
    meta_to_adm1 = np.array([gid_to_idx.get(str(g), -1) for g in meta_z["unique_gids"]], dtype=np.int32)

    if args.oper_shards_dir:
        build_operational_multilead(Path(args.oper_shards_dir), adm1_gids, adm1_isos, meta_to_adm1, paths.work_dir)
    if not args.skip_retro:
        extract_retrospective_from_s3(meta_p, adm1_gids, adm1_isos, meta_to_adm1, paths.work_dir)


if __name__ == "__main__":
    main()
