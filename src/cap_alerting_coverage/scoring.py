"""Unified GADM Admin-1 flood alerting evaluation and scoring engine (Tier 1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cap_alerting_coverage.config import (
    BENCHMARKS,
    CAP_CONTINUOUS_THRESHOLDS,
    CAP_SEVERITIES,
    D_TOTAL,
    DATES,
    FFM_THRESHOLDS,
    GEOGLOWS_OPER_RP_TIERS,
    LEAD_WINDOWS_DAYS,
    LEGACY_EXCLUDED_19,
    RETURN_PERIODS,
    ProjectPaths,
    date_to_idx,
)
from cap_alerting_coverage.schemas import (
    validate_cap_gid0_matrix,
    validate_cap_gid1_matrix,
    validate_event_catalog,
    validate_ffm_gid1_matrix,
    validate_tier_cube,
)


def _find_first_existing(paths: ProjectPaths, candidates: list[tuple[str, tuple[str, ...]]]) -> Path:
    last_err: Exception | None = None
    for fname, subdirs in candidates:
        try:
            return paths.resolve_data_file(fname, *subdirs)
        except FileNotFoundError as exc:
            last_err = exc
    raise FileNotFoundError(str(last_err))


def load_universe_sets(
    paths: ProjectPaths,
    strict_185: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, set[str]]]:
    """Load canonical GADM Admin-1 index and jurisdiction cohorts from Supplementary Data 1."""
    try:
        adm1_path = _find_first_existing(
            paths,
            [
                ("gadm41_adm1_index.parquet", ("01_event_catalogs",)),
                ("gadm41_adm1.parquet", ("01_event_catalogs",)),
                ("gadm41_adm1_index.csv", ("01_event_catalogs",)),
            ],
        )
        if adm1_path.suffix == ".csv":
            adm1 = pd.read_csv(adm1_path, usecols=["gid_1", "country_iso3"])
        else:
            adm1 = pd.read_parquet(adm1_path, columns=["gid_1", "country_iso3"])
        gid1_list = adm1["gid_1"].astype(str).to_numpy()
        iso3_list = adm1["country_iso3"].astype(str).to_numpy()
    except FileNotFoundError:
        # Fallback: extract identical (gid_1, country_iso3) index directly from DHM NPZ header
        dhm_path = paths.resolve_data_file("dhm_10d_gid1_multilead_2021_2025.npz", "02_admin1_exceedance_matrices")
        with np.load(dhm_path, allow_pickle=False) as dhm_npz:
            gid1_list = dhm_npz["gid_1"].astype(str)
            iso3_list = dhm_npz["country_iso3"].astype(str)

    gid_to_idx = {str(g): i for i, g in enumerate(gid1_list)}

    sup1 = pd.read_csv(paths.supplementary_data_1_csv, keep_default_na=False)
    sup1["ISO3"] = sup1["ISO3"].str.strip()
    sup1_isos = set(sup1["ISO3"].tolist())
    cat1_isos = set(sup1[sup1["Cohort_and_Category"].str.startswith("Cohort A")]["ISO3"].tolist())
    cat2_isos = set(sup1[sup1["Cohort_and_Category"].str.startswith("Cohort B")]["ISO3"].tolist())
    cat3_isos = set(sup1[sup1["Cohort_and_Category"].str.startswith("Cohort C")]["ISO3"].tolist())
    cohort_d_35 = set(sup1[sup1["Cohort_and_Category"].str.contains("Cohort D", na=False)]["ISO3"].tolist())
    if strict_185:
        if len(sup1_isos) != 185 or len(cohort_d_35) != 35:
            raise AssertionError(
                f"Expected 185 audited jurisdictions and 35 Cohort D exclusions, "
                f"got {len(sup1_isos)} and {len(cohort_d_35)}."
            )
    eval_150 = sup1_isos - cohort_d_35
    orig_166 = sup1_isos - set(LEGACY_EXCLUDED_19)

    return (
        gid1_list,
        iso3_list,
        gid_to_idx,
        {
            "eval_150": eval_150,
            "cohort_d_35": cohort_d_35,
            "orig_166": orig_166,
            "all_185": sup1_isos,
            "cat1_isos": cat1_isos,
            "cat2_isos": cat2_isos,
            "cat3_isos": cat3_isos,
        },
    )


def load_benchmark_data(
    paths: ProjectPaths,
    gid1_list: np.ndarray,
    _iso3_list: np.ndarray,
    gid_to_idx: dict[str, int],
    validate_schemas: bool = True,
) -> dict[str, Any]:
    """Load all derived Admin-1 exceedance matrices and Catalog A/B event tables into RAM."""
    t0 = time.time()
    n_g = len(gid1_list)

    # 1. CAP Admin-1 & Admin-0 daily matrices
    cap1_path = paths.resolve_data_file("cap_gid1_daily_2020_2025.parquet", "02_admin1_exceedance_matrices")
    cap1 = pd.read_parquet(cap1_path)
    if validate_schemas:
        validate_cap_gid1_matrix(cap1, "cap_gid1_daily_2020_2025.parquet")

    cap_perm = np.zeros((n_g, D_TOTAL), dtype=np.int8)
    cap_rest = np.zeros((n_g, D_TOTAL), dtype=np.int8)
    cap_perm_cont = np.zeros((n_g, D_TOTAL), dtype=np.int16)
    cap_rest_cont = np.zeros((n_g, D_TOTAL), dtype=np.int16)

    gi = cap1["gid_1"].map(gid_to_idx)
    valid = gi.notna().to_numpy()
    g_arr = gi[valid].astype(np.int32).to_numpy()
    d_arr = (pd.to_datetime(cap1["alert_date"].to_numpy()[valid]) - DATES[0]).days.astype(np.int32)
    m_d = (d_arr >= 0) & (d_arr < D_TOTAL)
    gv, dv = g_arr[m_d], d_arr[m_d]
    ps_v = cap1["perm_max_sev"].to_numpy()[valid][m_d].astype(np.int16)
    rs_v = cap1["rest_max_sev"].to_numpy()[valid][m_d].astype(np.int16)
    cap_perm[gv, dv] = ps_v.astype(np.int8)
    cap_rest[gv, dv] = rs_v.astype(np.int8)

    psc64 = cap1["perm_max_score64"].to_numpy()[valid][m_d].astype(np.int16)
    rsc64 = cap1["rest_max_score64"].to_numpy()[valid][m_d].astype(np.int16)
    cap_perm_cont[gv, dv] = np.where(ps_v > 0, psc64, 0)
    cap_rest_cont[gv, dv] = np.where(rs_v > 0, rsc64, 0)

    # Exact severity boundary subsumption invariant (64-rank lexicographic ordering)
    for sv, rc_b in ((1, 1), (2, 17), (3, 33), (4, 49)):
        if not np.array_equal(cap_perm >= sv, cap_perm_cont >= rc_b):
            raise AssertionError(f"CAP broad severity>={sv} mismatch vs score64>={rc_b}.")
        if not np.array_equal(cap_rest >= sv, cap_rest_cont >= rc_b):
            raise AssertionError(f"CAP flood-coded severity>={sv} mismatch vs score64>={rc_b}.")

    cap0_path = paths.resolve_data_file("cap_gid0_daily_2020_2025.parquet", "02_admin1_exceedance_matrices")
    cap0_df = pd.read_parquet(cap0_path)
    if validate_schemas:
        validate_cap_gid0_matrix(cap0_df, "cap_gid0_daily_2020_2025.parquet")

    # 2. FFM (0-24h)
    ffm_path = paths.resolve_data_file("ffm_gid1_daily_2021_2025.parquet", "02_admin1_exceedance_matrices")
    ffm = pd.read_parquet(ffm_path)
    if validate_schemas:
        validate_ffm_gid1_matrix(ffm, "ffm_gid1_daily_2021_2025.parquet")
    ffm_prob = np.zeros((n_g, D_TOTAL), dtype=np.float32)
    fgi = ffm["gid_1"].map(gid_to_idx)
    fv = fgi.notna().to_numpy()
    fg = fgi[fv].astype(np.int32).to_numpy()
    fd = (pd.to_datetime(ffm["valid_date"].to_numpy()[fv]) - DATES[0]).days.astype(np.int32)
    fm = (fd >= 0) & (fd < D_TOTAL)
    ffm_prob[fg[fm], fd[fm]] = ffm["max_prob"].to_numpy()[fv][fm]

    # 3. DHM (0d + 0-10d multi-lead)
    dhm_path = paths.resolve_data_file("dhm_10d_gid1_multilead_2021_2025.npz", "02_admin1_exceedance_matrices")
    with np.load(dhm_path, allow_pickle=False) as dhm_npz:
        dhm_dict = {k: dhm_npz[k] for k in dhm_npz.files}
    if validate_schemas:
        validate_tier_cube(dhm_dict, 3, gid1_list, "dhm_10d_gid1_multilead_2021_2025.npz", expected_days=D_TOTAL)
    dhm_tiers = dhm_dict["tiers"]

    # 4. GeoGLOWS v2 Retrospective (0d) & Operational (2024-04-01..2025-12-31)
    geo_r_path = paths.resolve_data_file("geoglows_v2_retro_gid1_daily_2021_2025.npz", "02_admin1_exceedance_matrices")
    with np.load(geo_r_path, allow_pickle=False) as geo_r_npz:
        geo_r_dict = {k: geo_r_npz[k] for k in geo_r_npz.files}
    if validate_schemas:
        validate_tier_cube(
            geo_r_dict, 2, gid1_list, "geoglows_v2_retro_gid1_daily_2021_2025.npz", expected_days=D_TOTAL
        )
    geo_retro = geo_r_dict["tiers"]

    geo_o_path = paths.resolve_data_file(
        "geoglows_v2_oper_gid1_multilead_2024_2025.npz", "02_admin1_exceedance_matrices"
    )
    with np.load(geo_o_path, allow_pickle=False) as geo_o_npz:
        geo_o_dict = {k: geo_o_npz[k] for k in geo_o_npz.files}
    if validate_schemas:
        validate_tier_cube(
            geo_o_dict, 4, gid1_list, "geoglows_v2_oper_gid1_multilead_2024_2025.npz", expected_days=None
        )
    geo_oper = geo_o_dict["tiers"]
    geo_oper_d0 = date_to_idx(geo_o_dict["issue_dates"][0])

    # 5. GloFAS v5 (0d)
    gf_path = paths.resolve_data_file("glofas_v5_gid1_daily_2021_2025.npz", "02_admin1_exceedance_matrices")
    with np.load(gf_path, allow_pickle=False) as gf_npz:
        gf_dict = {k: gf_npz[k] for k in gf_npz.files}
    if validate_schemas:
        validate_tier_cube(gf_dict, 2, gid1_list, "glofas_v5_gid1_daily_2021_2025.npz", expected_days=D_TOTAL)
    glofas_tiers = gf_dict["tiers"]

    # 6. Event Catalog A & Catalog B (all audited jurisdictions)
    cat_a_path = _find_first_existing(
        paths,
        [
            ("catalog_without_gs_all_185_2021_2025.parquet", ("01_event_catalogs",)),
            ("catalog_a_all_2021_2025.parquet", ("01_event_catalogs",)),
        ],
    )
    cat_b_path = _find_first_existing(
        paths,
        [
            ("catalog_with_gs_all_185_2021_2025.parquet", ("01_event_catalogs",)),
            ("catalog_b_all_2021_2025.parquet", ("01_event_catalogs",)),
        ],
    )
    cat_no_gs = pd.read_parquet(cat_a_path)
    cat_with_gs = pd.read_parquet(cat_b_path)
    if validate_schemas:
        validate_event_catalog(cat_no_gs, cat_a_path.name)
        validate_event_catalog(cat_with_gs, cat_b_path.name)

    elapsed = time.time() - t0
    print(f"Loaded all Admin-1 exceedance cubes & catalogs into RAM in {elapsed:.1f}s.", flush=True)
    return {
        "cap_perm": cap_perm,
        "cap_rest": cap_rest,
        "cap_perm_cont": cap_perm_cont,
        "cap_rest_cont": cap_rest_cont,
        "cap0_df": cap0_df,
        "ffm_prob": ffm_prob,
        "dhm_tiers": dhm_tiers,
        "geo_retro": geo_retro,
        "geo_oper": geo_oper,
        "geo_oper_d0": geo_oper_d0,
        "glofas_tiers": glofas_tiers,
        "cat_without_gs": cat_no_gs,
        "cat_with_gs": cat_with_gs,
    }


def prepare_catalog_slice(
    cat_df: pd.DataFrame,
    gid_to_idx: dict[str, int],
    allowed_isos: set[str],
    bench_start_str: str,
    bench_end_str: str,
    lead_n: int,
) -> dict[str, Any]:
    """Prepare integer index arrays for O(1) prefix-sum event recall & episode precision scoring."""
    b0 = date_to_idx(bench_start_str)
    b1 = date_to_idx(bench_end_str)
    w_min = b0 - lead_n
    if w_min < 0:
        raise ValueError(f"Window start {w_min} precedes calendar start {DATES[0].date()}.")

    sd = (pd.to_datetime(cat_df["start_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    ed = (pd.to_datetime(cat_df["end_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    mask = (cat_df["country_iso3"].isin(allowed_isos).to_numpy()) & (sd >= b0) & (sd <= b1)

    sub = cat_df[mask]
    ev_g = sub["gid_1"].map(gid_to_idx).astype(np.int32).to_numpy()
    ev_s = sd[mask]
    ev_e = np.maximum(ev_s, np.minimum(ed[mask], b1))
    ev_w0 = np.maximum(w_min, ev_s - lead_n)
    ev_w1 = ev_e
    ev_pop = np.nan_to_num(sub["waterfall_pop"].to_numpy(dtype=np.float64), nan=0.0)
    ev_iso = sub["country_iso3"].to_numpy()

    return {
        "b0": b0,
        "b1": b1,
        "w_min": w_min,
        "n_events": len(ev_g),
        "ev_g": ev_g,
        "ev_s": ev_s,
        "ev_e": ev_e,
        "ev_w0": ev_w0,
        "ev_w1": ev_w1,
        "ev_pop": ev_pop,
        "ev_iso": ev_iso,
    }


def build_fcover(n_g: int, cat_slice: dict[str, Any]) -> np.ndarray:
    """Build ``(n_g, D_TOTAL)`` boolean mask marking all Admin-1 days inside ``[ev_w0, ev_w1]``."""
    diff = np.zeros((n_g, D_TOTAL + 1), dtype=np.int32)
    np.add.at(diff, (cat_slice["ev_g"], cat_slice["ev_w0"]), 1)
    np.add.at(diff, (cat_slice["ev_g"], cat_slice["ev_w1"] + 1), -1)
    return np.cumsum(diff[:, :D_TOTAL], axis=1) > 0


def score_boolean_matrix(
    w_full: np.ndarray,
    cat_slice: dict[str, Any],
    fcover: np.ndarray,
    g_mask: np.ndarray,
    max_gap: int = 2,
) -> dict[str, Any]:
    """Score a boolean ``(n_g, D_TOTAL)`` alert matrix for event recall and episode precision."""
    w0, w1 = cat_slice["w_min"], cat_slice["b1"]
    win_len = w1 - w0 + 1
    n_g = w_full.shape[0]

    g_idx = np.flatnonzero(g_mask)
    if len(g_idx) == 0:
        return {
            "n_events": cat_slice["n_events"],
            "tp_ev": 0,
            "tp_pre": 0,
            "recall_pct": 0.0,
            "pre_recall_pct": 0.0,
            "pop_recall_pct": 0.0,
            "n_warn_days": 0,
            "n_episodes": 0,
            "tp_ep": 0,
            "fp_ep": 0,
            "precision_pct": np.nan,
            "f1_pct": np.nan,
        }

    w_win = w_full[g_idx, w0 : w1 + 1]
    flat = np.flatnonzero(w_win)
    n_warn_days = len(flat)
    if n_warn_days == 0:
        return {
            "n_events": cat_slice["n_events"],
            "tp_ev": 0,
            "tp_pre": 0,
            "recall_pct": 0.0,
            "pre_recall_pct": 0.0,
            "pop_recall_pct": 0.0,
            "n_warn_days": 0,
            "n_episodes": 0,
            "tp_ep": 0,
            "fp_ep": 0,
            "precision_pct": np.nan,
            "f1_pct": np.nan,
        }

    ev_g = cat_slice["ev_g"]
    ev_w0 = cat_slice["ev_w0"]
    ev_w1 = cat_slice["ev_w1"]
    ev_s = cat_slice["ev_s"]
    ev_pop = cat_slice["ev_pop"]

    if len(ev_g) > 0:
        g_remap = np.full(n_g, -1, dtype=np.int32)
        g_remap[g_idx] = np.arange(len(g_idx), dtype=np.int32)
        ev_r = g_remap[ev_g]
        valid_ev = ev_r >= 0
        if np.any(valid_ev):
            ev_r_safe = np.where(valid_ev, ev_r, 0).astype(np.int64)
            row_base = ev_r_safe * win_len
            key_lo = row_base + (ev_w0 - w0)
            key_hi = row_base + (ev_w1 - w0)
            key_pre = row_base + (ev_s - w0)
            pos_lo = np.searchsorted(flat, key_lo, side="left")
            det = valid_ev & (pos_lo < np.searchsorted(flat, key_hi, side="right"))
            pre_det = valid_ev & (pos_lo < np.searchsorted(flat, key_pre, side="right"))
            tp_ev = int(det.sum())
            tp_pre = int(pre_det.sum())
            tot_pop = float(ev_pop.sum())
            pop_rec = (float(ev_pop[det].sum()) / tot_pop * 100.0) if tot_pop > 0 else 0.0
        else:
            tp_ev, tp_pre, pop_rec = 0, 0, 0.0
    else:
        tp_ev, tp_pre, pop_rec = 0, 0, 0.0

    gi_sub, di_rel = np.divmod(flat, win_len)
    gi = g_idx[gi_sub]
    di = di_rel + w0
    is_new_ep = np.empty(n_warn_days, dtype=bool)
    is_new_ep[0] = True
    is_new_ep[1:] = (gi_sub[1:] != gi_sub[:-1]) | ((di_rel[1:] - di_rel[:-1]) > (max_gap + 1))

    ep_ids = np.cumsum(is_new_ep) - 1
    n_ep = int(ep_ids[-1]) + 1
    day_hit = fcover[gi, di]
    ep_tp = np.zeros(n_ep, dtype=bool)
    np.logical_or.at(ep_tp, ep_ids, day_hit)

    tp_ep = int(ep_tp.sum())
    fp_ep = n_ep - tp_ep

    if (tp_ev == 0) != (tp_ep == 0):
        raise AssertionError(f"Bipartite consistency mismatch: tp_ev={tp_ev}, tp_ep={tp_ep}")

    rec_pct = 100.0 * tp_ev / max(1, cat_slice["n_events"])
    pre_pct = 100.0 * tp_pre / max(1, cat_slice["n_events"])
    prec_pct = 100.0 * tp_ep / max(1, n_ep)
    f1_pct = (2.0 * rec_pct * prec_pct / (rec_pct + prec_pct)) if (rec_pct + prec_pct) > 0 else 0.0

    return {
        "n_events": cat_slice["n_events"],
        "tp_ev": tp_ev,
        "tp_pre": tp_pre,
        "recall_pct": round(rec_pct, 3),
        "pre_recall_pct": round(pre_pct, 3),
        "pop_recall_pct": round(pop_rec, 3),
        "n_warn_days": n_warn_days,
        "n_episodes": n_ep,
        "tp_ep": tp_ep,
        "fp_ep": fp_ep,
        "precision_pct": round(prec_pct, 3),
        "f1_pct": round(f1_pct, 3),
    }


def score_admin0_cap_audit(
    cap0_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    allowed_isos: set[str],
    bench_start_str: str,
    bench_end_str: str,
    lead_n: int,
    sev_thresh: int,
    kind: str = "perm",
    max_gap: int = 2,
) -> dict[str, Any]:
    """Country-level (Admin-0) CAP matching diagnostic for spatial attribution sensitivity."""
    b0 = date_to_idx(bench_start_str)
    b1 = date_to_idx(bench_end_str)
    w_min = b0 - lead_n

    sd = (pd.to_datetime(cat_df["start_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    ed = (pd.to_datetime(cat_df["end_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    mask = (cat_df["country_iso3"].isin(allowed_isos).to_numpy()) & (sd >= b0) & (sd <= b1)
    sub_cat = cat_df[mask].copy()
    sub_cat["sd_i"] = sd[mask]
    sub_cat["ed_i"] = np.maximum(sd[mask], np.minimum(ed[mask], b1))
    sub_cat["w0_i"] = np.maximum(w_min, sub_cat["sd_i"] - lead_n)

    iso_list = sorted(allowed_isos)
    iso_to_i = {c: i for i, c in enumerate(iso_list)}
    n_c = len(iso_list)

    col = "perm_max_sev" if kind == "perm" else "rest_max_sev"
    c0_sub = cap0_df[cap0_df["country_iso3"].isin(allowed_isos) & (cap0_df[col] >= sev_thresh)]
    w0 = np.zeros((n_c, D_TOTAL), dtype=bool)
    ci = c0_sub["country_iso3"].map(iso_to_i).to_numpy(dtype=np.int32)
    di = (pd.to_datetime(c0_sub["alert_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    m = (di >= w_min) & (di <= b1)
    w0[ci[m], di[m]] = True

    p0 = np.zeros((n_c, D_TOTAL + 1), dtype=np.int32)
    np.cumsum(w0, axis=1, out=p0[:, 1:])
    ev_c = sub_cat["country_iso3"].map(iso_to_i).to_numpy(dtype=np.int32)
    det = (p0[ev_c, sub_cat["ed_i"].to_numpy() + 1] - p0[ev_c, sub_cat["w0_i"].to_numpy()]) > 0
    tp_ev = int(det.sum())

    diff0 = np.zeros((n_c, D_TOTAL + 1), dtype=np.int32)
    np.add.at(diff0, (ev_c, sub_cat["w0_i"].to_numpy()), 1)
    np.add.at(diff0, (ev_c, sub_cat["ed_i"].to_numpy() + 1), -1)
    fcov0 = np.cumsum(diff0[:, :D_TOTAL], axis=1) > 0
    cgi, cdi_rel = np.where(w0[:, w_min : b1 + 1])
    if len(cgi) == 0:
        prec = np.nan
        n_ep, tp_ep = 0, 0
    else:
        cdi = cdi_rel + w_min
        new_ep = np.empty(len(cgi), dtype=bool)
        new_ep[0] = True
        new_ep[1:] = (cgi[1:] != cgi[:-1]) | ((cdi[1:] - cdi[:-1]) > (max_gap + 1))
        ep_ids = np.cumsum(new_ep) - 1
        n_ep = int(ep_ids[-1]) + 1
        ep_tp = np.zeros(n_ep, dtype=bool)
        np.logical_or.at(ep_tp, ep_ids, fcov0[cgi, cdi])
        tp_ep = int(ep_tp.sum())
        prec = 100.0 * tp_ep / n_ep

    return {
        "n_events": len(sub_cat),
        "tp_ev": tp_ev,
        "recall_pct": round(100.0 * tp_ev / max(1, len(sub_cat)), 3),
        "n_episodes": n_ep,
        "tp_ep": tp_ep,
        "precision_pct": round(prec, 3) if np.isfinite(prec) else np.nan,
    }


def make_issue_indexed_w_matrix(
    data: dict[str, Any], sys_key: str, thr_idx: float | int, lead_n: int, n_g: int
) -> np.ndarray:
    """Issue-indexed multi-lead max-pooling (sensitivity comparison only)."""
    if sys_key == "DHM_Oper":
        l_max = min(lead_n, data["dhm_tiers"].shape[2] - 1)
        return data["dhm_tiers"][:, :, : l_max + 1].max(axis=2) >= thr_idx
    if sys_key.startswith("GeoGLOWS_Oper_"):
        prod_map = {"EnsMedian": 0, "EnsMean": 1, "EnsP75": 2, "HRES": 3}
        p_idx = prod_map[sys_key.replace("GeoGLOWS_Oper_", "")]
        max_l = 9 if p_idx == 3 else (data["geo_oper"].shape[3] - 1)
        l_max = min(lead_n, max_l)
        w = np.zeros((n_g, D_TOTAL), dtype=bool)
        d0 = data["geo_oper_d0"]
        n_d = data["geo_oper"].shape[1]
        end_d = min(D_TOTAL, d0 + n_d)
        w[:, d0:end_d] = data["geo_oper"][:, : end_d - d0, p_idx, : l_max + 1].max(axis=2) >= thr_idx
        return w
    return make_w_matrix(data, sys_key, thr_idx, lead_n, n_g)


def make_valid_aligned_w_matrix(
    data: dict[str, Any], sys_key: str, thr_idx: float | int, lead_n: int, n_g: int
) -> np.ndarray:
    """Align multi-lead forecasts strictly by predicted flood date (d_valid = d_issue + L)."""
    w = np.zeros((n_g, D_TOTAL), dtype=bool)
    if sys_key == "DHM_Oper":
        tiers = data["dhm_tiers"]
        l_max = min(lead_n, tiers.shape[2] - 1)
        for lead_day in range(l_max + 1):
            if lead_day == 0:
                w |= tiers[:, :, 0] >= thr_idx
            else:
                w[:, lead_day:] |= tiers[:, :-lead_day, lead_day] >= thr_idx
        return w
    if sys_key.startswith("GeoGLOWS_Oper_"):
        prod_map = {"EnsMedian": 0, "EnsMean": 1, "EnsP75": 2, "HRES": 3}
        p_idx = prod_map[sys_key.replace("GeoGLOWS_Oper_", "")]
        max_l = 9 if p_idx == 3 else (data["geo_oper"].shape[3] - 1)
        l_max = min(lead_n, max_l)
        sub_cube = data["geo_oper"][:, :, p_idx, :]
        d0 = data["geo_oper_d0"]
        n_d = sub_cube.shape[1]
        for lead_day in range(l_max + 1):
            s_i = d0 + lead_day
            e_i = min(D_TOTAL, d0 + n_d + lead_day)
            if e_i > s_i:
                w[:, s_i:e_i] |= sub_cube[:, : (e_i - s_i), lead_day] >= thr_idx
        return w
    return make_w_matrix(data, sys_key, thr_idx, lead_n, n_g)


def make_w_matrix(data: dict[str, Any], sys_key: str, thr_idx: float | int, lead_n: int, n_g: int) -> np.ndarray:
    """Return ``(n_g, D_TOTAL)`` boolean trigger matrix (valid-date aligned for multi-lead models)."""
    if sys_key == "CAP_Perm":
        return data["cap_perm"] >= thr_idx
    if sys_key == "CAP_Rest":
        return data["cap_rest"] >= thr_idx
    if sys_key == "CAP_Perm_Cont":
        return data["cap_perm_cont"] >= thr_idx
    if sys_key == "CAP_Rest_Cont":
        return data["cap_rest_cont"] >= thr_idx
    if sys_key == "FFM":
        return data["ffm_prob"] >= thr_idx
    if sys_key == "DHM_0d":
        return data["dhm_tiers"][:, :, 0] >= thr_idx
    if sys_key == "DHM_Oper" or sys_key.startswith("GeoGLOWS_Oper_"):
        return make_valid_aligned_w_matrix(data, sys_key, thr_idx, lead_n, n_g)
    if sys_key == "GloFAS_v5_0d":
        return data["glofas_tiers"] >= thr_idx
    if sys_key == "GeoGLOWS_Retro_0d":
        return data["geo_retro"] >= thr_idx
    raise ValueError(f"Unknown sys_key={sys_key!r}")


def score_single_lead_curve(
    data: dict[str, Any],
    cat_df: pd.DataFrame,
    gid_to_idx: dict[str, int],
    allowed_isos: set[str],
    g_mask: np.ndarray,
    bench_start_str: str,
    bench_end_str: str,
    sys_name: str,
    max_lead: int,
    rp_tier: int = 2,
) -> list[dict[str, Any]]:
    """Score single-lead forecast skill decay L = 0..max_lead on predicted valid date."""
    b0 = date_to_idx(bench_start_str)
    b1 = date_to_idx(bench_end_str)
    sd = (pd.to_datetime(cat_df["start_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    ed = (pd.to_datetime(cat_df["end_date"].to_numpy()) - DATES[0]).days.astype(np.int32)
    mask = (cat_df["country_iso3"].isin(allowed_isos).to_numpy()) & (sd >= b0) & (sd <= b1)
    sub = cat_df[mask]
    ev_g = sub["gid_1"].map(gid_to_idx).astype(np.int32).to_numpy()
    ev_s = sd[mask]
    ev_e = np.maximum(ev_s, np.minimum(ed[mask], b1))
    n_g = len(g_mask)

    diff = np.zeros((n_g, D_TOTAL + 1), dtype=np.int32)
    np.add.at(diff, (ev_g, ev_s), 1)
    np.add.at(diff, (ev_g, ev_e + 1), -1)
    fcov = np.cumsum(diff[:, :D_TOTAL], axis=1) > 0

    rows = []
    for lead_day in range(max_lead + 1):
        wv = np.zeros((n_g, D_TOTAL), dtype=bool)
        if sys_name == "DHM_10D":
            wv[:, b0 : b1 + 1] = data["dhm_tiers"][:, b0 - lead_day : b1 - lead_day + 1, lead_day] >= rp_tier
        elif sys_name == "GeoGLOWS_EnsMedian":
            d0 = data["geo_oper_d0"]
            i0 = b0 - lead_day - d0
            i1 = b1 - lead_day - d0
            if i0 < 0:
                raise AssertionError(f"GeoGLOWS issue slice starts before archive start: i0={i0}")
            wv[:, b0 : b1 + 1] = data["geo_oper"][:, i0 : i1 + 1, 0, lead_day] >= rp_tier
        elif sys_name == "GeoGLOWS_HRES":
            d0 = data["geo_oper_d0"]
            i0 = b0 - lead_day - d0
            i1 = b1 - lead_day - d0
            if i0 < 0:
                raise AssertionError(f"GeoGLOWS HRES issue slice starts before archive start: i0={i0}")
            wv[:, b0 : b1 + 1] = data["geo_oper"][:, i0 : i1 + 1, 3, lead_day] >= rp_tier
        else:
            raise ValueError(f"Unsupported single-lead system: {sys_name!r}")

        w_m = wv & g_mask[:, None]
        prefix = np.zeros((n_g, D_TOTAL + 1), dtype=np.int32)
        np.cumsum(w_m, axis=1, out=prefix[:, 1:])
        det = (prefix[ev_g, ev_e + 1] - prefix[ev_g, ev_s]) > 0
        tp_ev = int(det.sum())

        gi, di_rel = np.where(w_m[:, b0 : b1 + 1])
        if len(gi) == 0:
            n_ep, tp_ep, prec = 0, 0, np.nan
        else:
            di = di_rel + b0
            is_new = np.empty(len(gi), dtype=bool)
            is_new[0] = True
            is_new[1:] = (gi[1:] != gi[:-1]) | ((di[1:] - di[:-1]) > 3)
            ep_ids = np.cumsum(is_new) - 1
            n_ep = int(ep_ids[-1]) + 1
            ep_tp = np.zeros(n_ep, dtype=bool)
            np.logical_or.at(ep_tp, ep_ids, fcov[gi, di])
            tp_ep = int(ep_tp.sum())
            prec = 100.0 * tp_ep / n_ep
        rows.append(
            {
                "system": sys_name,
                "rp_tier": rp_tier,
                "rp_label": RETURN_PERIODS[rp_tier - 1],
                "lead_day": lead_day,
                "n_events": len(ev_g),
                "tp_ev": tp_ev,
                "recall_pct": round(100.0 * tp_ev / max(1, len(ev_g)), 3),
                "n_episodes": n_ep,
                "tp_ep": tp_ep,
                "precision_pct": round(prec, 3) if np.isfinite(prec) else np.nan,
            }
        )
    return rows


def extract_pareto_frontier(grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract the non-dominated (max recall, max precision) Pareto frontier sorted by ascending recall."""
    valid = [r for r in grid_rows if np.isfinite(r["precision_pct"])]
    valid.sort(key=lambda r: (-r["recall_pct"], -r["precision_pct"]))
    frontier = []
    best_prec = -1.0
    for row in valid:
        if row["precision_pct"] > best_prec + 1e-9:
            frontier.append(dict(row))
            best_prec = row["precision_pct"]
    frontier.reverse()
    return frontier


def bin_envelope(df_pts: pd.DataFrame, bin_width: float = 2.5) -> pd.DataFrame:
    """Compute smooth upper envelope taking max precision inside each recall bin."""
    if df_pts.empty:
        return df_pts
    work = df_pts.copy()
    work["rbin"] = np.floor(work["recall_pct"].astype(float) / bin_width) * bin_width
    work = work.sort_values(["rbin", "precision_pct", "recall_pct"], ascending=[True, False, False])
    binned = work.groupby("rbin", as_index=False).first()
    return binned.sort_values("recall_pct").reset_index(drop=True)


def score_ffm_groundsource_slices(
    data: dict[str, Any],
    gid1_list: np.ndarray,
    iso3_list: np.ndarray,
    gid_to_idx: dict[str, int],
    eval_150: set[str],
    df_pr_base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score FFM, CAP, DHM, and 2D hybrid grids across Catalog A, Catalog B, and media-only clusters."""
    n_g = len(gid1_list)
    g_mask = np.isin(iso3_list, list(eval_150))

    cat_wo = data["cat_without_gs"]
    cat_w = data["cat_with_gs"]
    cat_gs_only = cat_w[
        cat_w["has_groundsource"].astype(bool) & ~cat_w["has_idmc"].astype(bool) & ~cat_w["has_montandon"].astype(bool)
    ].copy()

    catalogs = {
        "Without_GS": cat_wo,
        "With_GS": cat_w,
        "GS_Media_Only": cat_gs_only,
    }
    slices: dict[tuple[str, str], dict[str, Any]] = {}
    fcovers: dict[tuple[str, str], np.ndarray] = {}
    for cname, cdf in catalogs.items():
        for bname, bstart, bend in BENCHMARKS:
            sl = prepare_catalog_slice(cdf, gid_to_idx, eval_150, bstart, bend, 7)
            fc = build_fcover(n_g, sl)
            slices[(cname, bname)] = sl
            fcovers[(cname, bname)] = fc

    w_cap_perm_all = make_w_matrix(data, "CAP_Perm", 1, 7, n_g)
    w_cap_perm_cont = [(k, make_w_matrix(data, "CAP_Perm_Cont", k, 7, n_g)) for k in CAP_CONTINUOUS_THRESHOLDS]
    w_cap_perm_sevs = [(sv, lbl, make_w_matrix(data, "CAP_Perm", sv, 7, n_g)) for sv, lbl in CAP_SEVERITIES]
    w_cap_rest_cont = [(k, make_w_matrix(data, "CAP_Rest_Cont", k, 7, n_g)) for k in CAP_CONTINUOUS_THRESHOLDS]
    w_cap_rest_sevs = [(sv, lbl, make_w_matrix(data, "CAP_Rest", sv, 7, n_g)) for sv, lbl in CAP_SEVERITIES]
    w_dhm_rps = [(rp, make_w_matrix(data, "DHM_Oper", ri, 7, n_g)) for ri, rp in enumerate(RETURN_PERIODS, start=1)]
    w_ffm_taus = [(tau, make_w_matrix(data, "FFM", tau, 7, n_g)) for tau in FFM_THRESHOLDS]
    w_f050 = dict(w_ffm_taus)[0.50]

    def _score_row(
        w_mat: np.ndarray, cname: str, bname: str, sys_name: str, thr_label: str, thr_order: int
    ) -> dict[str, Any]:
        res = score_boolean_matrix(w_mat, slices[(cname, bname)], fcovers[(cname, bname)], g_mask, max_gap=2)
        return {
            "n_events": int(res["n_events"]),
            "tp_ev": int(res["tp_ev"]),
            "tp_pre": int(res["tp_pre"]),
            "recall_pct": round(float(res["recall_pct"]), 3),
            "pre_recall_pct": round(float(res["pre_recall_pct"]), 3),
            "pop_recall_pct": round(float(res["pop_recall_pct"]), 3),
            "n_warn_days": int(res["n_warn_days"]),
            "n_episodes": int(res["n_episodes"]),
            "tp_ep": int(res["tp_ep"]),
            "fp_ep": int(res["fp_ep"]),
            "precision_pct": round(float(res["precision_pct"]), 3) if np.isfinite(res["precision_pct"]) else np.nan,
            "f1_pct": round(float(res["f1_pct"]), 3) if np.isfinite(res["f1_pct"]) else np.nan,
            "catalog": cname,
            "benchmark": bname,
            "lead_N": 7,
            "system": sys_name,
            "thr_label": thr_label,
            "thr_order": int(thr_order),
        }

    out_rows = []
    for cname in ("Without_GS", "With_GS", "GS_Media_Only"):
        for bname, _, _ in BENCHMARKS:
            for idx, (k, w_mat) in enumerate(w_cap_perm_cont, start=1):
                out_rows.append(_score_row(w_mat, cname, bname, "CAP_Perm_Cont", f"CAP_Score64>={k}", idx))
            for sv, lbl, w_mat in w_cap_perm_sevs:
                out_rows.append(_score_row(w_mat, cname, bname, "CAP_Perm", lbl, sv))
            for idx, (k, w_mat) in enumerate(w_cap_rest_cont, start=1):
                out_rows.append(_score_row(w_mat, cname, bname, "CAP_Rest_Cont", f"CAP_Score64>={k}", idx))
            for sv, lbl, w_mat in w_cap_rest_sevs:
                out_rows.append(_score_row(w_mat, cname, bname, "CAP_Rest", lbl, sv))
            for idx, (rp, w_mat) in enumerate(w_dhm_rps, start=1):
                out_rows.append(_score_row(w_mat, cname, bname, "DHM_Oper", f"RP={rp:g}", idx))
            for idx, (tau, w_mat) in enumerate(w_ffm_taus, start=1):
                out_rows.append(_score_row(w_mat, cname, bname, "FFM", f"tau={tau:.2f}", idx))
                out_rows.append(
                    _score_row(w_cap_perm_all | w_mat, cname, bname, "CAP_Perm_U_FFM", f"tau={tau:.2f}", idx)
                )
            for idx, (rp, w_d) in enumerate(w_dhm_rps, start=1):
                out_rows.append(_score_row(w_d | w_f050, cname, bname, "DHM_Oper_U_FFM050", f"RP={rp:g}", idx))
                out_rows.append(
                    _score_row(
                        w_cap_perm_all | w_d | w_f050, cname, bname, "CAP_Perm_U_DHM_Oper_U_FFM050", f"RP={rp:g}", idx
                    )
                )

            grid_dhm_ffm = []
            grid_cap_dhm_ffm = []
            rp_opts: list[tuple[float | None, np.ndarray | None]] = [(None, None)] + list(w_dhm_rps)
            tau_opts: list[tuple[float | None, np.ndarray | None]] = [(None, None)] + list(w_ffm_taus)
            gidx = 0
            for rp, w_d in rp_opts:
                for tau, w_f in tau_opts:
                    if rp is None and tau is None:
                        continue
                    gidx += 1
                    lbl_rp = "RP=off" if rp is None else f"RP={rp:g}"
                    lbl_tau = "tau=off" if tau is None else f"tau={tau:.2f}"
                    lbl = f"{lbl_rp}|{lbl_tau}"
                    if w_d is None:
                        assert w_f is not None
                        w_df = w_f
                    elif w_f is None:
                        w_df = w_d
                    else:
                        w_df = w_d | w_f
                    r_df = _score_row(w_df, cname, bname, "DHM_Oper_U_FFM_Grid", lbl, gidx)
                    grid_dhm_ffm.append(r_df)
                    r_cdf = _score_row(w_cap_perm_all | w_df, cname, bname, "CAP_Perm_U_DHM_Oper_U_FFM_Grid", lbl, gidx)
                    grid_cap_dhm_ffm.append(r_cdf)

            out_rows.extend(grid_dhm_ffm)
            out_rows.extend(grid_cap_dhm_ffm)

            for idx, pr in enumerate(extract_pareto_frontier(grid_dhm_ffm), start=1):
                cp = dict(pr)
                cp["system"] = "DHM_Oper_U_FFM_Pareto"
                cp["thr_order"] = idx
                out_rows.append(cp)

            for idx, pr in enumerate(extract_pareto_frontier(grid_cap_dhm_ffm), start=1):
                cp = dict(pr)
                cp["system"] = "CAP_Perm_U_DHM_Oper_U_FFM_Pareto"
                cp["thr_order"] = idx
                out_rows.append(cp)

    df_out = pd.DataFrame(out_rows)
    if df_pr_base is not None and not df_pr_base.empty:
        df_base7 = df_pr_base[df_pr_base["lead_N"] == 7]
        shared_syss = [
            "CAP_Perm",
            "CAP_Perm_Cont",
            "CAP_Rest",
            "CAP_Rest_Cont",
            "DHM_Oper",
            "FFM",
            "DHM_Oper_U_FFM050",
            "CAP_Perm_U_FFM",
        ]
        m_chk = pd.merge(
            df_out[df_out["system"].isin(shared_syss)],
            df_base7[df_base7["system"].isin(shared_syss)],
            on=["catalog", "benchmark", "lead_N", "system", "thr_label"],
            suffixes=("_new", "_base"),
        )
        if not m_chk.empty:
            rec_diff = (m_chk["recall_pct_new"] - m_chk["recall_pct_base"]).abs().max()
            prec_diff = (m_chk["precision_pct_new"] - m_chk["precision_pct_base"]).abs().dropna()
            max_prec = float(prec_diff.max()) if not prec_diff.empty else 0.0
            if float(rec_diff) > 0.002 or max_prec > 0.002:
                raise AssertionError(
                    f"FFM slice parity mismatch vs pr_curves_summary: rec_diff={rec_diff}, prec_diff={max_prec}"
                )
    return df_out


def run_full_scoring(
    paths: ProjectPaths,
    strict_185: bool = True,
    validate_schemas: bool = True,
) -> dict[str, Any]:
    """Run the full Tier-1 scoring suite and write all clean CSV outputs to ``paths.results_dir``."""
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    gid1_list, iso3_list, gid_to_idx, universes = load_universe_sets(paths, strict_185=strict_185)
    data = load_benchmark_data(paths, gid1_list, iso3_list, gid_to_idx, validate_schemas=validate_schemas)
    n_g = len(gid1_list)

    eval_150 = universes["eval_150"]
    g_mask_150 = np.isin(iso3_list, list(eval_150))

    # 1. Full PR sweep at N=7d and N=15d across both benchmarks & catalogs
    pr_rows = []
    cap_cont_rows = []
    grid2d_rows = []
    catalogs = [
        ("Without_GS", data["cat_without_gs"]),
        ("With_GS", data["cat_with_gs"]),
    ]

    for cat_name, cat_df in catalogs:
        for b_name, b0_s, b1_s in BENCHMARKS:
            for lead_n in (7, 15):
                c_sl = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, b0_s, b1_s, lead_n)
                fcov = build_fcover(n_g, c_sl)

                for cap_k in ("CAP_Perm", "CAP_Rest"):
                    for sev_i, sev_lbl in CAP_SEVERITIES:
                        w_mat = make_w_matrix(data, cap_k, sev_i, lead_n, n_g)
                        res = score_boolean_matrix(w_mat, c_sl, fcov, g_mask_150, max_gap=2)
                        res.update(
                            {
                                "catalog": cat_name,
                                "benchmark": b_name,
                                "lead_N": lead_n,
                                "system": cap_k,
                                "thr_label": sev_lbl,
                                "thr_order": sev_i,
                            }
                        )
                        pr_rows.append(res)
                    cont_k = f"{cap_k}_Cont"
                    for ord_i, r_thr in enumerate(CAP_CONTINUOUS_THRESHOLDS, start=1):
                        w_mat = make_w_matrix(data, cont_k, r_thr, lead_n, n_g)
                        res = score_boolean_matrix(w_mat, c_sl, fcov, g_mask_150, max_gap=2)
                        res.update(
                            {
                                "catalog": cat_name,
                                "benchmark": b_name,
                                "lead_N": lead_n,
                                "system": cont_k,
                                "thr_label": f"CAP_Score64>={r_thr}",
                                "thr_order": ord_i,
                            }
                        )
                        pr_rows.append(res)
                        cap_cont_rows.append(dict(res))

                w_ffm_map = {tau: make_w_matrix(data, "FFM", tau, lead_n, n_g) for tau in FFM_THRESHOLDS}
                r_ffm_map = {}
                for ti, tau in enumerate(FFM_THRESHOLDS):
                    res = score_boolean_matrix(w_ffm_map[tau], c_sl, fcov, g_mask_150, max_gap=2)
                    r_ffm_map[tau] = res
                    row_f = dict(res)
                    row_f.update(
                        {
                            "catalog": cat_name,
                            "benchmark": b_name,
                            "lead_N": lead_n,
                            "system": "FFM",
                            "thr_label": f"tau={tau:.2f}",
                            "thr_order": ti,
                        }
                    )
                    pr_rows.append(row_f)

                rp_models = ["DHM_0d", "DHM_Oper", "GeoGLOWS_Retro_0d", "GloFAS_v5_0d"]
                if b_name == "2024_2025":
                    rp_models += [
                        "GeoGLOWS_Oper_EnsMedian",
                        "GeoGLOWS_Oper_EnsMean",
                        "GeoGLOWS_Oper_EnsP75",
                        "GeoGLOWS_Oper_HRES",
                    ]

                w_dhm_cache: dict[tuple[str, int], np.ndarray] = {}
                r_dhm_cache: dict[tuple[str, int], dict[str, Any]] = {}
                for sys_k in rp_models:
                    rp_pairs = (
                        GEOGLOWS_OPER_RP_TIERS
                        if sys_k.startswith("GeoGLOWS_Oper_")
                        else tuple(enumerate(RETURN_PERIODS, start=1))
                    )
                    for ri, rp in rp_pairs:
                        w_mat = make_w_matrix(data, sys_k, ri, lead_n, n_g)
                        res = score_boolean_matrix(w_mat, c_sl, fcov, g_mask_150, max_gap=2)
                        row_r = dict(res)
                        row_r.update(
                            {
                                "catalog": cat_name,
                                "benchmark": b_name,
                                "lead_N": lead_n,
                                "system": sys_k,
                                "thr_label": f"RP={rp:g}",
                                "thr_order": ri,
                            }
                        )
                        pr_rows.append(row_r)
                        if sys_k in ("DHM_0d", "DHM_Oper"):
                            w_dhm_cache[(sys_k, ri)] = w_mat
                            r_dhm_cache[(sys_k, ri)] = res

                w_cap_p = make_w_matrix(data, "CAP_Perm", 1, lead_n, n_g)
                r_cap_p = score_boolean_matrix(w_cap_p, c_sl, fcov, g_mask_150, max_gap=2)
                r_cap_r = score_boolean_matrix(
                    make_w_matrix(data, "CAP_Rest", 1, lead_n, n_g), c_sl, fcov, g_mask_150, max_gap=2
                )
                if r_cap_p["recall_pct"] < r_cap_r["recall_pct"] - 1e-6:
                    raise AssertionError("Broad CAP recall fell below explicit flood-coded CAP recall.")

                for d_key, u_fixed_key, u_pareto_key in (
                    ("DHM_0d", "DHM_0d_U_FFM050", "DHM_0d_U_FFM_Pareto"),
                    ("DHM_Oper", "DHM_Oper_U_FFM050", "DHM_Oper_U_FFM_Pareto"),
                ):
                    full_grid = []
                    cap_comb_grid = []
                    for tau in FFM_THRESHOLDS:
                        f_only = dict(r_ffm_map[tau])
                        f_only.update(
                            {
                                "catalog": cat_name,
                                "benchmark": b_name,
                                "lead_N": lead_n,
                                "system": u_pareto_key,
                                "thr_label": f"RP=off|tau={tau:.2f}",
                            }
                        )
                        full_grid.append(f_only)
                        if d_key == "DHM_Oper":
                            cf_only = score_boolean_matrix(w_cap_p | w_ffm_map[tau], c_sl, fcov, g_mask_150, max_gap=2)
                            cf_only.update(
                                {
                                    "catalog": cat_name,
                                    "benchmark": b_name,
                                    "lead_N": lead_n,
                                    "system": "CAP_Perm_U_DHM_Oper_U_FFM_Pareto",
                                    "thr_label": f"RP=off|tau={tau:.2f}",
                                }
                            )
                            cap_comb_grid.append(cf_only)
                    for ri, rp in enumerate(RETURN_PERIODS, start=1):
                        d_only = dict(r_dhm_cache[(d_key, ri)])
                        d_only.update(
                            {
                                "catalog": cat_name,
                                "benchmark": b_name,
                                "lead_N": lead_n,
                                "system": u_pareto_key,
                                "thr_label": f"RP={rp:g}|tau=off",
                            }
                        )
                        full_grid.append(d_only)
                        if d_key == "DHM_Oper":
                            cd_only = score_boolean_matrix(
                                w_cap_p | w_dhm_cache[(d_key, ri)], c_sl, fcov, g_mask_150, max_gap=2
                            )
                            cd_only.update(
                                {
                                    "catalog": cat_name,
                                    "benchmark": b_name,
                                    "lead_N": lead_n,
                                    "system": "CAP_Perm_U_DHM_Oper_U_FFM_Pareto",
                                    "thr_label": f"RP={rp:g}|tau=off",
                                }
                            )
                            cap_comb_grid.append(cd_only)

                    for ri, rp in enumerate(RETURN_PERIODS, start=1):
                        w_d = w_dhm_cache[(d_key, ri)]
                        r_d = r_dhm_cache[(d_key, ri)]
                        for tau in FFM_THRESHOLDS:
                            w_u = w_d | w_ffm_map[tau]
                            res = score_boolean_matrix(w_u, c_sl, fcov, g_mask_150, max_gap=2)
                            if res["recall_pct"] < max(r_d["recall_pct"], r_ffm_map[tau]["recall_pct"]) - 1e-6:
                                raise AssertionError("Union recall violated monotonicity.")
                            g_row = dict(res)
                            g_row.update(
                                {
                                    "catalog": cat_name,
                                    "benchmark": b_name,
                                    "lead_N": lead_n,
                                    "system": u_pareto_key,
                                    "thr_label": f"RP={rp:g}|tau={tau:.2f}",
                                }
                            )
                            full_grid.append(g_row)
                            cu_row = None
                            if d_key == "DHM_Oper":
                                grid2d_rows.append(dict(g_row))
                                res_cu = score_boolean_matrix(w_cap_p | w_u, c_sl, fcov, g_mask_150, max_gap=2)
                                if res_cu["recall_pct"] < max(res["recall_pct"], r_cap_p["recall_pct"]) - 1e-6:
                                    raise AssertionError("Hybrid union recall violated monotonicity.")
                                cu_row = dict(res_cu)
                                cu_row.update(
                                    {
                                        "catalog": cat_name,
                                        "benchmark": b_name,
                                        "lead_N": lead_n,
                                        "system": "CAP_Perm_U_DHM_Oper_U_FFM_Pareto",
                                        "thr_label": f"RP={rp:g}|tau={tau:.2f}",
                                    }
                                )
                                cap_comb_grid.append(cu_row)
                            if abs(tau - 0.50) < 1e-9:
                                f_row = dict(res)
                                f_row.update(
                                    {
                                        "catalog": cat_name,
                                        "benchmark": b_name,
                                        "lead_N": lead_n,
                                        "system": u_fixed_key,
                                        "thr_label": f"RP={rp:g}",
                                        "thr_order": ri,
                                    }
                                )
                                pr_rows.append(f_row)
                                if d_key == "DHM_Oper" and cu_row is not None:
                                    cm_row = dict(cu_row)
                                    cm_row.update(
                                        {
                                            "system": "CAP_Perm_U_DHM_Oper_U_FFM050",
                                            "thr_label": f"RP={rp:g}",
                                            "thr_order": ri,
                                        }
                                    )
                                    pr_rows.append(cm_row)

                    pareto = extract_pareto_frontier(full_grid)
                    for idx_p, pr_pt in enumerate(pareto, start=1):
                        pr_pt["thr_order"] = idx_p
                        pr_rows.append(pr_pt)
                    for bnd_pt in [r_ffm_map[t] for t in FFM_THRESHOLDS] + [
                        r_dhm_cache[(d_key, ri)] for ri in range(1, len(RETURN_PERIODS) + 1)
                    ]:
                        if np.isfinite(bnd_pt["precision_pct"]):
                            if not any(
                                p["recall_pct"] >= bnd_pt["recall_pct"] - 1e-6
                                and p["precision_pct"] >= bnd_pt["precision_pct"] - 1e-6
                                for p in pareto
                            ):
                                raise AssertionError(f"Pareto dominance invariant violated on {u_pareto_key}.")
                    if d_key == "DHM_Oper":
                        for idx_p, cp_pt in enumerate(extract_pareto_frontier(cap_comb_grid), start=1):
                            cp_pt["thr_order"] = idx_p
                            pr_rows.append(cp_pt)

                all_rp_pairs = tuple(enumerate(RETURN_PERIODS, start=1))
                ffm_pairs = tuple((i + 1, tau) for i, tau in enumerate(FFM_THRESHOLDS))
                union_targets: list[tuple[str, str, tuple[tuple[int, float], ...]]] = [
                    ("CAP_Perm_U_DHM_0d", "DHM_0d", all_rp_pairs),
                    ("CAP_Perm_U_DHM_Oper", "DHM_Oper", all_rp_pairs),
                    ("CAP_Perm_U_FFM", "FFM", ffm_pairs),
                    ("CAP_Perm_U_GeoGLOWS_Retro_0d", "GeoGLOWS_Retro_0d", all_rp_pairs),
                    ("CAP_Perm_U_GloFAS_v5_0d", "GloFAS_v5_0d", all_rp_pairs),
                ]
                if b_name == "2024_2025":
                    union_targets.append(
                        ("CAP_Perm_U_GeoGLOWS_Oper_EnsMedian", "GeoGLOWS_Oper_EnsMedian", GEOGLOWS_OPER_RP_TIERS)
                    )
                    union_targets.append(
                        ("CAP_Perm_U_GeoGLOWS_Oper_HRES", "GeoGLOWS_Oper_HRES", GEOGLOWS_OPER_RP_TIERS)
                    )

                for u_name, base_k, thr_pairs in union_targets:
                    for thr_ord, t_val in thr_pairs:
                        if base_k == "FFM":
                            w_m = w_ffm_map[t_val]
                        elif (base_k, thr_ord) in w_dhm_cache:
                            w_m = w_dhm_cache[(base_k, thr_ord)]
                        else:
                            w_m = make_w_matrix(data, base_k, thr_ord, lead_n, n_g)
                        res = score_boolean_matrix(w_cap_p | w_m, c_sl, fcov, g_mask_150, max_gap=2)
                        if res["recall_pct"] < r_cap_p["recall_pct"] - 1e-6:
                            raise AssertionError(f"Union {u_name} recall below CAP baseline.")
                        lbl = f"tau={t_val:.2f}" if base_k == "FFM" else f"RP={t_val:g}"
                        res.update(
                            {
                                "catalog": cat_name,
                                "benchmark": b_name,
                                "lead_N": lead_n,
                                "system": u_name,
                                "thr_label": lbl,
                                "thr_order": thr_ord,
                            }
                        )
                        pr_rows.append(res)

    df_pr = pd.DataFrame(pr_rows)
    df_cap_cont = pd.DataFrame(cap_cont_rows)
    df_grid2d = pd.DataFrame(grid2d_rows)
    df_pr.to_csv(paths.results_dir / "pr_curves_summary.csv", index=False)
    df_cap_cont.to_csv(paths.results_dir / "cap_continuous_pr.csv", index=False)
    df_grid2d.to_csv(paths.results_dir / "dhm_ffm_2d_grid.csv", index=False)

    # 2. Lead-window sweep (0..15d) & single-lead decay (0..14d)
    lead_rows = []
    single_rows = []
    op_points = [
        ("CAP_Perm", 1, "All CAP (Permissive)"),
        ("CAP_Rest", 1, "All CAP (Restrictive)"),
        ("FFM", 0.50, "FFM (tau=0.50)"),
        ("DHM_0d", 2, "DHM 0d (RP=2)"),
        ("DHM_Oper", 2, "DHM Oper (RP=2)"),
        ("GeoGLOWS_Retro_0d", 2, "GeoGLOWS Retro 0d (RP=2)"),
        ("GloFAS_v5_0d", 2, "GloFAS v5 0d (RP=2)"),
    ]
    for cat_name, cat_df in catalogs:
        for b_name, b0_s, b1_s in BENCHMARKS:
            pts = list(op_points)
            if b_name == "2024_2025":
                pts += [
                    ("GeoGLOWS_Oper_EnsMedian", 2, "GeoGLOWS Oper EnsMedian (RP=2)"),
                    ("GeoGLOWS_Oper_HRES", 2, "GeoGLOWS Oper HRES (RP=2)"),
                ]
            for lead_n in LEAD_WINDOWS_DAYS:
                c_sl = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, b0_s, b1_s, lead_n)
                fcov = build_fcover(n_g, c_sl)
                for sk, targ, lbl in pts:
                    w_mat = make_w_matrix(data, sk, targ, lead_n, n_g)
                    res = score_boolean_matrix(w_mat, c_sl, fcov, g_mask_150, max_gap=2)
                    res.update(
                        {"catalog": cat_name, "benchmark": b_name, "lead_N": lead_n, "system": sk, "op_label": lbl}
                    )
                    lead_rows.append(res)
                w_u = make_w_matrix(data, "DHM_Oper", 2, lead_n, n_g) | make_w_matrix(data, "FFM", 0.50, lead_n, n_g)
                ru = score_boolean_matrix(w_u, c_sl, fcov, g_mask_150, max_gap=2)
                ru.update(
                    {
                        "catalog": cat_name,
                        "benchmark": b_name,
                        "lead_N": lead_n,
                        "system": "DHM_Oper_U_FFM050",
                        "op_label": "Combined DHM(RP=2) U FFM(0.5)",
                    }
                )
                lead_rows.append(ru)

            for rp_t in (2, 3):
                for row in score_single_lead_curve(
                    data, cat_df, gid_to_idx, eval_150, g_mask_150, b0_s, b1_s, "DHM_10D", 10, rp_tier=rp_t
                ):
                    row.update({"catalog": cat_name, "benchmark": b_name})
                    single_rows.append(row)
                if b_name == "2024_2025":
                    for sname, mxl in (("GeoGLOWS_EnsMedian", 14), ("GeoGLOWS_HRES", 9)):
                        for row in score_single_lead_curve(
                            data, cat_df, gid_to_idx, eval_150, g_mask_150, b0_s, b1_s, sname, mxl, rp_tier=rp_t
                        ):
                            row.update({"catalog": cat_name, "benchmark": b_name})
                            single_rows.append(row)

    df_leads = pd.DataFrame(lead_rows)
    df_single = pd.DataFrame(single_rows)
    df_leads.to_csv(paths.results_dir / "lead_window_sweep.csv", index=False)
    df_single.to_csv(paths.results_dir / "single_lead_decay.csv", index=False)

    # 3. Sensitivity audits (spatial attribution, cohorts, episode gaps, windows, lead sweep)
    audit_rows = []
    metric_keys = ("n_events", "tp_ev", "recall_pct", "n_episodes", "tp_ep", "precision_pct")
    for cat_name, cat_df in catalogs:
        for lead_n in (7, 15):
            c_sl = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, "2021-01-01", "2025-12-31", lead_n)
            fcov = build_fcover(n_g, c_sl)
            for kind, sk in (("perm", "CAP_Perm"), ("rest", "CAP_Rest")):
                r_gid1 = score_boolean_matrix(
                    make_w_matrix(data, sk, 1, lead_n, n_g), c_sl, fcov, g_mask_150, max_gap=2
                )
                r_gid0 = score_admin0_cap_audit(
                    data["cap0_df"], cat_df, eval_150, "2021-01-01", "2025-12-31", lead_n, 1, kind=kind, max_gap=2
                )
                audit_rows.append(
                    {
                        "audit_type": "Spatial_Attribution",
                        "catalog": cat_name,
                        "lead_N": lead_n,
                        "variant": f"{kind}_strict_GID1",
                        **{k: r_gid1[k] for k in metric_keys},
                    }
                )
                audit_rows.append(
                    {
                        "audit_type": "Spatial_Attribution",
                        "catalog": cat_name,
                        "lead_N": lead_n,
                        "variant": f"{kind}_legacy_Admin0",
                        **r_gid0,
                    }
                )

    all_non_d = set(np.unique(iso3_list)) - universes["cohort_d_35"]
    univ_map = [
        ("Eval_150_CohortD35Excl", eval_150),
        ("Orig_166_Legacy19Excl", universes["orig_166"]),
        ("All_185_Jurisdictions", universes["all_185"]),
        ("All_NonD_Territories", all_non_d),
    ]
    for cat_name, cat_df in catalogs:
        for uname, uset in univ_map:
            gm = np.isin(iso3_list, list(uset))
            c_sl = prepare_catalog_slice(cat_df, gid_to_idx, uset, "2021-01-01", "2025-12-31", 7)
            fcov = build_fcover(n_g, c_sl)
            for sk, targ, vname in (
                ("CAP_Perm", 1, "CAP_Perm_All"),
                ("CAP_Rest", 1, "CAP_Rest_All"),
                ("DHM_Oper", 2, "DHM_Oper_RP2"),
                ("FFM", 0.50, "FFM_050"),
            ):
                r = score_boolean_matrix(make_w_matrix(data, sk, targ, 7, n_g), c_sl, fcov, gm, max_gap=2)
                audit_rows.append(
                    {
                        "audit_type": "Cohort_Exclusion",
                        "catalog": cat_name,
                        "lead_N": 7,
                        "variant": f"{uname}__{vname}",
                        **{k: r[k] for k in metric_keys},
                    }
                )

    for cat_name, cat_df in catalogs:
        c_sl = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, "2021-01-01", "2025-12-31", 7)
        fcov = build_fcover(n_g, c_sl)
        for gap_d in (0, 2, 7):
            for sk, targ, vname in (
                ("CAP_Perm", 1, "CAP_Perm_All"),
                ("CAP_Rest", 1, "CAP_Rest_All"),
                ("DHM_0d", 2, "DHM_0d_RP2"),
                ("DHM_Oper", 2, "DHM_Oper_RP2"),
                ("FFM", 0.50, "FFM_050"),
                ("GeoGLOWS_Retro_0d", 2, "GeoGLOWS_Retro_RP2"),
                ("GloFAS_v5_0d", 2, "GloFAS_v5_RP2"),
            ):
                r = score_boolean_matrix(make_w_matrix(data, sk, targ, 7, n_g), c_sl, fcov, g_mask_150, max_gap=gap_d)
                audit_rows.append(
                    {
                        "audit_type": f"Episode_Gap_{gap_d}d",
                        "catalog": cat_name,
                        "lead_N": 7,
                        "variant": vname,
                        **{k: r[k] for k in metric_keys},
                    }
                )

    for cat_name, cat_df in catalogs:
        c_sl = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, "2021-01-01", "2025-12-31", 7)
        fcov = build_fcover(n_g, c_sl)
        ev_g, ev_s, ev_w0 = c_sl["ev_g"], c_sl["ev_s"], c_sl["ev_w0"]
        for sk, targ, vname in (
            ("CAP_Perm", 1, "CAP_Perm_All"),
            ("DHM_0d", 2, "DHM_0d_RP2"),
            ("DHM_Oper", 2, "DHM_Oper_RP2"),
            ("FFM", 0.50, "FFM_050"),
            ("GeoGLOWS_Retro_0d", 2, "GeoGLOWS_Retro_RP2"),
            ("GloFAS_v5_0d", 2, "GloFAS_v5_RP2"),
        ):
            w_i = make_w_matrix(data, sk, targ, 7, n_g)
            r_prim = score_boolean_matrix(w_i, c_sl, fcov, g_mask_150, max_gap=2)
            w_masked = w_i & g_mask_150[:, None]
            prefix = np.zeros((n_g, D_TOTAL + 1), dtype=np.int32)
            np.cumsum(w_masked, axis=1, out=prefix[:, 1:])
            tp_before = int(((prefix[ev_g, ev_s] - prefix[ev_g, np.minimum(ev_w0, ev_s)]) > 0).sum())
            audit_rows.append(
                {
                    "audit_type": "Event_Window_Rule",
                    "catalog": cat_name,
                    "lead_N": 7,
                    "variant": f"{vname}__Primary_[start-7d,end]",
                    **{k: r_prim[k] for k in metric_keys},
                }
            )
            audit_rows.append(
                {
                    "audit_type": "Event_Window_Rule",
                    "catalog": cat_name,
                    "lead_N": 7,
                    "variant": f"{vname}__PreOnset_[start-7d,start]",
                    "n_events": r_prim["n_events"],
                    "tp_ev": r_prim["tp_pre"],
                    "recall_pct": r_prim["pre_recall_pct"],
                    "n_episodes": r_prim["n_episodes"],
                    "tp_ep": r_prim["tp_ep"],
                    "precision_pct": r_prim["precision_pct"],
                }
            )
            audit_rows.append(
                {
                    "audit_type": "Event_Window_Rule",
                    "catalog": cat_name,
                    "lead_N": 7,
                    "variant": f"{vname}__StrictlyBefore_[start-7d,start-1d]",
                    "n_events": r_prim["n_events"],
                    "tp_ev": tp_before,
                    "recall_pct": round(100.0 * tp_before / max(1, r_prim["n_events"]), 3),
                    "n_episodes": r_prim["n_episodes"],
                    "tp_ep": r_prim["tp_ep"],
                    "precision_pct": r_prim["precision_pct"],
                }
            )
            if sk == "DHM_Oper":
                w_iss = make_issue_indexed_w_matrix(data, sk, targ, 7, n_g)
                r_iss = score_boolean_matrix(w_iss, c_sl, fcov, g_mask_150, max_gap=2)
                audit_rows.append(
                    {
                        "audit_type": "Event_Window_Rule",
                        "catalog": cat_name,
                        "lead_N": 7,
                        "variant": f"{vname}__ValidDateAligned_[start-7d,end]",
                        **{k: r_prim[k] for k in metric_keys},
                    }
                )
                audit_rows.append(
                    {
                        "audit_type": "Event_Window_Rule",
                        "catalog": cat_name,
                        "lead_N": 7,
                        "variant": f"{vname}__IssueIndexed_[start-7d,end]",
                        **{k: r_iss[k] for k in metric_keys},
                    }
                )

        c_sl24 = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, "2024-04-16", "2025-12-31", 7)
        fcov24 = build_fcover(n_g, c_sl24)
        for sk24, vn24 in (
            ("DHM_Oper", "DHM_Oper_RP2_2024_2025"),
            ("GeoGLOWS_Oper_EnsMedian", "GeoGLOWS_Oper_EnsMed_RP2_2024_2025"),
            ("GeoGLOWS_Oper_HRES", "GeoGLOWS_Oper_HRES_RP2_2024_2025"),
        ):
            ri24 = score_boolean_matrix(
                make_issue_indexed_w_matrix(data, sk24, 2, 7, n_g), c_sl24, fcov24, g_mask_150, max_gap=2
            )
            rv24 = score_boolean_matrix(
                make_valid_aligned_w_matrix(data, sk24, 2, 7, n_g), c_sl24, fcov24, g_mask_150, max_gap=2
            )
            audit_rows.append(
                {
                    "audit_type": "Event_Window_Rule",
                    "catalog": cat_name,
                    "lead_N": 7,
                    "variant": f"{vn24}__IssueIndexed",
                    **{k: ri24[k] for k in metric_keys},
                }
            )
            audit_rows.append(
                {
                    "audit_type": "Event_Window_Rule",
                    "catalog": cat_name,
                    "lead_N": 7,
                    "variant": f"{vn24}__ValidDateAligned",
                    **{k: rv24[k] for k in metric_keys},
                }
            )

    for cat_name, cat_df in catalogs:
        for n_sw in (0, 3, 7, 15):
            c_sl_sw = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, "2021-01-01", "2025-12-31", n_sw)
            fcov_sw = build_fcover(n_g, c_sl_sw)
            w_cap_sw = make_w_matrix(data, "CAP_Perm", 1, n_sw, n_g)
            w_dhm_sw = make_w_matrix(data, "DHM_Oper", 2, n_sw, n_g)
            w_ffm_sw = make_w_matrix(data, "FFM", 0.50, n_sw, n_g)
            for vname, w_sw in (
                ("CAP_Perm_All", w_cap_sw),
                ("DHM_Oper_RP2", w_dhm_sw),
                ("FFM_050", w_ffm_sw),
                ("CAP_Perm_U_DHM_Oper_U_FFM050", w_cap_sw | w_dhm_sw | w_ffm_sw),
            ):
                r_sw = score_boolean_matrix(w_sw, c_sl_sw, fcov_sw, g_mask_150, max_gap=2)
                audit_rows.append(
                    {
                        "audit_type": "Lead_Window_Sweep",
                        "catalog": cat_name,
                        "lead_N": n_sw,
                        "variant": vname,
                        **{k: r_sw[k] for k in metric_keys},
                    }
                )

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(paths.results_dir / "sensitivity_audits.csv", index=False)

    # 4. Standalone FFM & Groundsource slices (Catalog A, Catalog B, GS_Media_Only)
    df_ffm_slices = score_ffm_groundsource_slices(data, gid1_list, iso3_list, gid_to_idx, eval_150, df_pr_base=df_pr)
    df_ffm_slices.to_csv(paths.results_dir / "ffm_gs_slices_pr.csv", index=False)

    # 5. Fine-resolution (0.02-step) empirical FFM & DHM+FFM curves + 2D Pareto ceiling
    df_fine_ffm = compute_fine_ffm_pr_curves(data, gid1_list, iso3_list, gid_to_idx, eval_150)
    df_fine_ffm.to_csv(paths.results_dir / "fine_ffm_pr_curves.csv", index=False)

    return {
        "gid1_list": gid1_list,
        "iso3_list": iso3_list,
        "gid_to_idx": gid_to_idx,
        "universes": universes,
        "data": data,
        "df_pr": df_pr,
        "df_cap_cont": df_cap_cont,
        "df_grid2d": df_grid2d,
        "df_leads": df_leads,
        "df_single": df_single,
        "df_audit": df_audit,
        "df_ffm_slices": df_ffm_slices,
        "df_fine_ffm": df_fine_ffm,
    }


def compute_fine_ffm_pr_curves(
    data: dict[str, Any],
    gid1_list: list[str],
    iso3_list: np.ndarray,
    gid_to_idx: dict[str, int],
    eval_150: set[str],
    lead_n: int = 7,
) -> pd.DataFrame:
    """Compute exact 0.02-step empirical FFM, DHM(Q2yr)+FFM, and fine 2D Pareto PR curves."""
    n_g = len(gid1_list)
    g_mask_150 = np.isin(iso3_list, list(eval_150))

    cat_a = data["cat_without_gs"]
    cat_b = data["cat_with_gs"]
    cat_gs_only = cat_b[~cat_b["has_idmc"] & ~cat_b["has_montandon"] & cat_b["has_groundsource"]].copy()

    slices: dict[tuple[str, str], dict[str, Any]] = {}
    fcovers: dict[tuple[str, str], np.ndarray] = {}
    for cat_name, cat_df in (("Without_GS", cat_a), ("With_GS", cat_b), ("GS_Media_Only", cat_gs_only)):
        for bm_name, b_start, b_end in BENCHMARKS:
            sl = prepare_catalog_slice(cat_df, gid_to_idx, eval_150, b_start, b_end, lead_n=lead_n)
            slices[(cat_name, bm_name)] = sl
            fcovers[(cat_name, bm_name)] = build_fcover(n_g, sl)

    taus = sorted(
        {round(float(x), 2) for x in np.arange(0.05, 0.991, 0.02)}
        | {0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99}
    )

    w_cap_perm = data["cap_perm"] >= 1
    dhm_oper_mats = {
        rp_i: make_w_matrix(data, "DHM_Oper", rp_i, lead_n=lead_n, n_g=n_g)
        for rp_i in range(1, len(RETURN_PERIODS) + 1)
    }
    w_dhm_q2 = dhm_oper_mats[2]
    ffm_mats = {tau: (data["ffm_prob"] >= tau) for tau in taus}

    curve_rows: list[dict[str, Any]] = []
    for tau in taus:
        wf = ffm_mats[tau]
        combos = (
            ("FFM_fine", wf),
            ("CAP_Perm_U_FFM_fine", w_cap_perm | wf),
            ("DHM_Oper_Q2_U_FFM_fine", w_dhm_q2 | wf),
            ("CAP_Perm_U_DHM_Oper_Q2_U_FFM_fine", w_cap_perm | w_dhm_q2 | wf),
        )
        for sys_name, wm in combos:
            for (cat_name, bm_name), sl in slices.items():
                sc = score_boolean_matrix(wm, sl, fcovers[(cat_name, bm_name)], g_mask_150, max_gap=2)
                curve_rows.append(
                    {
                        "catalog": cat_name,
                        "benchmark": bm_name,
                        "system": sys_name,
                        "tau": tau,
                        "rp": 2 if "Q2" in sys_name else np.nan,
                        "thr_label": f"tau={tau:.2f}",
                        **sc,
                    }
                )

    pareto_rows: list[dict[str, Any]] = []
    rp_options: list[tuple[int, str, np.ndarray | None]] = [(0, "off", None)] + [
        (rp_i, str(RETURN_PERIODS[rp_i - 1]), dhm_oper_mats[rp_i]) for rp_i in range(1, len(RETURN_PERIODS) + 1)
    ]
    for mode_name, base_cap in (
        ("DHM_Oper_U_FFM_Pareto_fine", None),
        ("CAP_Perm_U_DHM_Oper_U_FFM_Pareto_fine", w_cap_perm),
    ):
        grid_by_slice: dict[tuple[str, str], list[dict[str, Any]]] = {
            k: [] for k in slices if k[0] != "GS_Media_Only"
        }
        for rp_i, rp_lbl, w_dhm in rp_options:
            for tau in taus:
                wm = ffm_mats[tau] if w_dhm is None else (w_dhm | ffm_mats[tau])
                if base_cap is not None:
                    wm = base_cap | wm
                for key in grid_by_slice:
                    sc = score_boolean_matrix(wm, slices[key], fcovers[key], g_mask_150, max_gap=2)
                    grid_by_slice[key].append(
                        {
                            "catalog": key[0],
                            "benchmark": key[1],
                            "system": mode_name,
                            "tau": tau,
                            "rp": RETURN_PERIODS[rp_i - 1] if rp_i > 0 else np.nan,
                            "thr_label": f"RP={rp_lbl}|tau={tau:.2f}",
                            **sc,
                        }
                    )
        for grows in grid_by_slice.values():
            pareto_rows.extend(extract_pareto_frontier(grows))

    return pd.DataFrame(curve_rows + pareto_rows)
