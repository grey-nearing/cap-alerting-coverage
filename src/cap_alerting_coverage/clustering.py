"""Subnational (GID_1) flood episode clustering, population waterfall, and Catalog B nesting."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from cap_alerting_coverage.config import LEGACY_EXCLUDED_19

PopRule = Literal["priority_fallback", "max", "max_with_killed_fallback"]


def collapse_warning_episodes(active_days: list[int], max_gap_days: int = 2) -> list[tuple[int, int]]:
    """Cluster sorted warning day indices separated by <= max_gap_days quiet days.

    Two active days ``d_prev`` and ``d`` belong to the same episode when
    ``(d - d_prev) <= (max_gap_days + 1)``.
    """
    if not active_days:
        return []
    days = sorted(set(int(d) for d in active_days))
    episodes: list[tuple[int, int]] = []
    start = prev = days[0]
    for day in days[1:]:
        if (day - prev) <= (max_gap_days + 1):
            prev = day
        else:
            episodes.append((start, prev))
            start = prev = day
    episodes.append((start, prev))
    return episodes


def compute_waterfall_population(
    num_affected: float,
    displacements: float,
    num_killed: float,
    pop_rule: PopRule = "priority_fallback",
) -> float:
    """Compute event-level affected population without double counting.

    - ``priority_fallback`` (manuscript pipeline baseline): uses ``num_affected`` if > 0,
      otherwise ``displacements`` if > 0, otherwise ``num_killed``.
    - ``max`` (pure ``max(num_affected, displacements)`` sensitivity variant, no fatality fallback).
    - ``max_with_killed_fallback``: uses ``max(num_affected, displacements)`` if > 0,
      otherwise ``num_killed``.
    """
    aff = float(num_affected or 0.0)
    disp = float(displacements or 0.0)
    kill = float(num_killed or 0.0)
    if pop_rule == "priority_fallback":
        return aff if aff > 0.0 else (disp if disp > 0.0 else kill)
    if pop_rule == "max":
        return max(aff, disp)
    if pop_rule == "max_with_killed_fallback":
        top = max(aff, disp)
        return top if top > 0.0 else kill
    raise ValueError(
        f"Unsupported pop_rule={pop_rule!r}; expected 'priority_fallback', 'max', or 'max_with_killed_fallback'."
    )


def cluster_gid1_episodes(
    records_df: pd.DataFrame,
    gap_days: int = 7,
    pop_rule: PopRule = "priority_fallback",
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Collapse subnational flood records sharing the same ``gid_1`` within ``<= gap_days``.

    Connected intervals sharing ``gid_1`` where ``next_start <= current_end + gap_days`` are
    merged into a single episode. Impact counts take the per-source maximum across merged rows
    before applying ``compute_waterfall_population``.
    """
    if records_df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "gid_1",
                "country_iso3",
                "start_date",
                "end_date",
                "has_idmc",
                "has_montandon",
                "has_groundsource",
                "waterfall_pop",
            ]
        )

    df = records_df.sort_values(["gid_1", "start_date", "end_date"]).reset_index(drop=True)
    gids = df["gid_1"].values
    c_isos = df["country_iso3"].values
    starts = pd.to_datetime(df["start_date"]).values.astype("datetime64[D]").astype(np.int64)
    ends = np.maximum(starts, pd.to_datetime(df["end_date"]).values.astype("datetime64[D]").astype(np.int64))
    has_idmc = df["has_idmc"].values
    has_mont = df["has_montandon"].values
    has_gs = df["has_groundsource"].values
    disp = df["displacements"].values
    aff = df["num_affected"].values
    kill = df["num_killed"].values

    episodes = []
    n = len(df)
    i = 0
    ep_counter = 0
    origin_dt = np.datetime64("1970-01-01", "D")

    while i < n:
        cur_g = gids[i]
        cur_iso = c_isos[i]
        cur_s = starts[i]
        cur_e = ends[i]
        cur_idmc = bool(has_idmc[i])
        cur_mont = bool(has_mont[i])
        cur_gs = bool(has_gs[i])
        cur_disp = float(disp[i])
        cur_aff = float(aff[i])
        cur_kill = float(kill[i])
        j = i + 1
        while j < n and gids[j] == cur_g and starts[j] <= cur_e + gap_days:
            if starts[j] < cur_s:
                cur_s = starts[j]
            if ends[j] > cur_e:
                cur_e = ends[j]
            cur_idmc = cur_idmc or bool(has_idmc[j])
            cur_mont = cur_mont or bool(has_mont[j])
            cur_gs = cur_gs or bool(has_gs[j])
            cur_disp = max(cur_disp, float(disp[j]))
            cur_aff = max(cur_aff, float(aff[j]))
            cur_kill = max(cur_kill, float(kill[j]))
            j += 1

        w_pop = compute_waterfall_population(cur_aff, cur_disp, cur_kill, pop_rule=pop_rule)
        ep_counter += 1
        episodes.append(
            {
                "event_id": f"{cur_g}_{ep_counter}",
                "gid_1": cur_g,
                "country_iso3": cur_iso,
                "start_date": pd.Timestamp(origin_dt + np.timedelta64(int(cur_s), "D")),
                "end_date": pd.Timestamp(origin_dt + np.timedelta64(int(cur_e), "D")),
                "has_idmc": cur_idmc,
                "has_montandon": cur_mont,
                "has_groundsource": cur_gs,
                "waterfall_pop": w_pop,
            }
        )
        i = j

    out = pd.DataFrame(episodes)
    out = out[(out["start_date"] >= start_date) & (out["start_date"] <= end_date)].reset_index(drop=True)
    return out


def merge_strictly_nested_catalog_b(
    cat_a_df: pd.DataFrame,
    gs_daily_df: pd.DataFrame,
    cohort_d_35: set[str],
    eval_150: set[str],
    audited_185: set[str],
    match_window_days: int = 7,
    pop_rule: PopRule = "priority_fallback",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly nested Catalog B merge preserving all Catalog A rows and population totals.

    Groundsource daily Admin-1 footprints are clustered per ``gid_1`` (<=7 days) and matched
    against Catalog A episodes sharing the exact same ``gid_1`` within ±7 days. Matched
    Catalog A rows receive ``has_groundsource=True`` with unchanged dates and population;
    only unmatched Groundsource clusters are appended with ``event_id`` prefixed by ``GS_``
    and ``waterfall_pop = 0.0``.
    """
    gs_eps = cluster_gid1_episodes(gs_daily_df, gap_days=match_window_days, pop_rule=pop_rule)
    a_by_g: dict[str, list[tuple[int, int, int]]] = {}
    cat_a_s = pd.to_datetime(cat_a_df["start_date"]).values.astype("datetime64[D]").astype(np.int64)
    cat_a_e = pd.to_datetime(cat_a_df["end_date"]).values.astype("datetime64[D]").astype(np.int64)
    for pos_i, (a_idx, gid_val) in enumerate(zip(cat_a_df.index, cat_a_df["gid_1"].values)):
        a_by_g.setdefault(str(gid_val), []).append((int(cat_a_s[pos_i]), int(cat_a_e[pos_i]), int(a_idx)))

    matched_a_idx: set[int] = set()
    unmatched_gs_mask = np.ones(len(gs_eps), dtype=bool)
    if not gs_eps.empty:
        gs_g = gs_eps["gid_1"].values
        gs_s = pd.to_datetime(gs_eps["start_date"]).values.astype("datetime64[D]").astype(np.int64)
        gs_e = pd.to_datetime(gs_eps["end_date"]).values.astype("datetime64[D]").astype(np.int64)
        for k in range(len(gs_eps)):
            cand = a_by_g.get(str(gs_g[k]))
            if cand:
                hit = False
                for s_a, e_a, a_idx in cand:
                    if gs_s[k] <= e_a + match_window_days and gs_e[k] >= s_a - match_window_days:
                        hit = True
                        matched_a_idx.add(a_idx)
                if hit:
                    unmatched_gs_mask[k] = False

    cat_a_parent = cat_a_df.copy()
    cat_a_parent["has_groundsource"] = cat_a_parent.index.isin(matched_a_idx)

    gs_unmatched = gs_eps[unmatched_gs_mask].copy()
    gs_unmatched["event_id"] = ["GS_" + str(eid) for eid in gs_unmatched["event_id"]]
    gs_unmatched["waterfall_pop"] = 0.0
    gs_unmatched["in_cohort_d_35"] = gs_unmatched["country_iso3"].isin(cohort_d_35)
    gs_unmatched["in_eval_150"] = gs_unmatched["country_iso3"].isin(eval_150)
    gs_unmatched["in_audited_185"] = gs_unmatched["country_iso3"].isin(audited_185)
    gs_unmatched["in_orig_excl_19"] = gs_unmatched["country_iso3"].isin(LEGACY_EXCLUDED_19)

    cat_b_all = pd.concat([cat_a_parent, gs_unmatched[cat_a_parent.columns]], ignore_index=True)
    return cat_b_all, gs_unmatched
