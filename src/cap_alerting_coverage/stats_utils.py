"""Statistical inference, bootstrap CIs, and Log-Pearson Type III return-period utilities."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from cap_alerting_coverage.config import RETURN_PERIODS


def mcnemar_exact_p(n_b: int, n_c: int = 0) -> float:
    """Compute the exact two-sided binomial McNemar p-value for discordant pairs (n_b, n_c)."""
    total = int(n_b) + int(n_c)
    if total == 0:
        return 1.0
    if int(n_b) < 0 or int(n_c) < 0:
        raise ValueError(f"Discordant pair counts must be non-negative, got n_b={n_b}, n_c={n_c}.")
    k = min(int(n_b), int(n_c))
    return float(stats.binomtest(k, total, 0.5).pvalue)


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return exact Clopper-Pearson binomial confidence interval (low, high) in [0, 1]."""
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}.")
    if not (0 <= k <= n):
        raise ValueError(f"Success count k={k} outside [0, {n}].")
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=confidence, method="exact")
    return float(ci.low), float(ci.high)


def wald_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return two-sided Wald normal-approximation confidence interval clamped to [0, 1]."""
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}.")
    if not (0 <= k <= n):
        raise ValueError(f"Success count k={k} outside [0, {n}].")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = k / n
    hw = z * math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - hw), min(1.0, p + hw)


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Return (z_stat, two_sided_p_value) for a pooled two-proportion z-test."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes n1 and n2 must be positive.")
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if denom == 0.0:
        return 0.0, 1.0
    z_stat = (p1 - p2) / denom
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
    return float(z_stat), p_val


def fisher_exact_2x2(tp_sel: int, n_sel: int, tp_excl: int, n_excl: int) -> tuple[float, float]:
    """Return (odds_ratio, two_sided_p_value) for a disjoint 2x2 exact Fisher test."""
    if n_sel <= 0 or n_excl <= 0:
        raise ValueError("Group totals n_sel and n_excl must be positive.")
    fp_sel = int(n_sel) - int(tp_sel)
    tn_excl = int(n_excl) - int(tp_excl)
    if min(int(tp_sel), fp_sel, int(tp_excl), tn_excl) < 0:
        raise ValueError("Contingency table cells must be non-negative.")
    res = stats.fisher_exact([[int(tp_sel), fp_sel], [int(tp_excl), tn_excl]], alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def cluster_design_effect(
    cluster_ids: Sequence[Any] | np.ndarray,
    indicators: np.ndarray,
) -> tuple[float, float]:
    """Compute Taylor-linearization cluster design effect (DEFF) and effective sample size N_eff.

    Uses the ratio-estimator cluster variance for a proportion ``p_hat = sum(y_c) / sum(n_c)``:
    ``Var_cluster = (M / (M - 1)) * sum_c (y_c - p_hat * n_c)^2 / N^2``, divided by
    ``Var_SRS = p_hat * (1 - p_hat) / N``.
    """
    y = np.asarray(indicators, dtype=np.float64)
    c = np.asarray(cluster_ids)
    if y.shape != c.shape:
        raise ValueError(f"Shape mismatch: cluster_ids {c.shape} vs indicators {y.shape}.")
    n_tot = len(y)
    if n_tot == 0:
        return float("nan"), float("nan")
    grp = pd.DataFrame({"cluster": c, "y": y}).groupby("cluster", sort=True)["y"].agg(["count", "sum"])
    m = len(grp)
    if m < 2:
        return float("nan"), float("nan")
    p_hat = float(grp["sum"].sum() / n_tot)
    var_srs = p_hat * (1.0 - p_hat) / n_tot
    if var_srs <= 0.0:
        return 1.0, float(n_tot)
    u = grp["sum"].to_numpy(dtype=np.float64) - p_hat * grp["count"].to_numpy(dtype=np.float64)
    var_cl = (m / (m - 1.0)) * float(np.sum(u**2)) / (float(n_tot) ** 2)
    deff = float(var_cl / var_srs)
    n_eff = float(n_tot / deff) if deff > 0.0 else float("inf")
    return deff, n_eff


def cluster_bootstrap_proportion_ci(
    cluster_ids: Sequence[Any] | np.ndarray,
    indicators: np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute cluster-level percentile bootstrap (estimate, ci_low, ci_high) for a proportion."""
    y = np.asarray(indicators, dtype=np.float64)
    c = np.asarray(cluster_ids)
    if y.shape != c.shape:
        raise ValueError(f"Shape mismatch: cluster_ids {c.shape} vs indicators {y.shape}.")
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    if weights is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError(f"weights shape {w.shape} != indicators shape {y.shape}.")

    df = pd.DataFrame({"cluster": c, "num": y * w, "den": w})
    grp = df.groupby("cluster", sort=True)[["num", "den"]].sum()
    num_arr = grp["num"].to_numpy(dtype=np.float64)
    den_arr = grp["den"].to_numpy(dtype=np.float64)
    den_tot = float(den_arr.sum())
    point = float(num_arr.sum() / den_tot) if den_tot > 0.0 else 0.0

    m = len(grp)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_resamples, m))
    den_b = den_arr[idx].sum(axis=1)
    num_b = num_arr[idx].sum(axis=1)
    samples = np.where(den_b > 0.0, num_b / np.where(den_b > 0.0, den_b, 1.0), 0.0)

    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def country_paired_wilcoxon(
    country_ids: Sequence[Any] | np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    """Run a two-sided paired Wilcoxon signed-rank test on per-country coverage percentages."""
    c = np.asarray(country_ids)
    b = np.asarray(baseline, dtype=np.float64)
    k = np.asarray(candidate, dtype=np.float64)
    if not (c.shape == b.shape == k.shape):
        raise ValueError(f"Shape mismatch: country_ids {c.shape}, baseline {b.shape}, candidate {k.shape}.")
    if len(c) == 0:
        return {
            "n_countries": 0,
            "n_improved": 0,
            "n_declined": 0,
            "n_tied": 0,
            "median_pp_gain": float("nan"),
            "mean_pp_gain": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
        }

    grp = (
        pd.DataFrame({"country": c, "base": b, "cand": k})
        .groupby("country", sort=True)
        .agg(n=("base", "size"), base=("base", "sum"), cand=("cand", "sum"))
    )
    pct_base = 100.0 * grp["base"].to_numpy(dtype=np.float64) / grp["n"].to_numpy(dtype=np.float64)
    pct_cand = 100.0 * grp["cand"].to_numpy(dtype=np.float64) / grp["n"].to_numpy(dtype=np.float64)
    diff = pct_cand - pct_base

    n_improved = int(np.sum(diff > 0.0))
    n_declined = int(np.sum(diff < 0.0))
    n_tied = int(np.sum(diff == 0.0))
    if np.all(diff == 0.0):
        w_stat, p_val = 0.0, 1.0
    else:
        res = stats.wilcoxon(diff, alternative="two-sided")
        w_stat, p_val = float(res.statistic), float(res.pvalue)

    return {
        "n_countries": int(len(grp)),
        "n_improved": n_improved,
        "n_declined": n_declined,
        "n_tied": n_tied,
        "median_pp_gain": float(np.median(diff)),
        "mean_pp_gain": float(np.mean(diff)),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": p_val,
    }


def country_clustered_coverage_inference(
    country_ids: Sequence[Any] | np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Compute country-level paired Wilcoxon tests, cluster DEFF, and clustered 95% bootstrap CIs.

    Clusters are sorted lexicographically by ``country_ids`` prior to resampling with
    ``numpy.random.default_rng(seed)`` so results are deterministic across runs.
    """
    c = np.asarray(country_ids)
    b = np.asarray(baseline, dtype=np.float64)
    k = np.asarray(candidate, dtype=np.float64)
    if not (c.shape == b.shape == k.shape):
        raise ValueError(f"Shape mismatch: country_ids {c.shape}, baseline {b.shape}, candidate {k.shape}.")

    wilcox = country_paired_wilcoxon(c, b, k)
    deff_base, neff_base = cluster_design_effect(c, b)
    deff_cand, neff_cand = cluster_design_effect(c, k)

    n_tot = len(c)
    if n_tot == 0:
        return {
            "n_events": 0,
            **wilcox,
            "baseline_deff": deff_base,
            "baseline_n_eff": neff_base,
            "candidate_deff": deff_cand,
            "candidate_n_eff": neff_cand,
            "baseline_pct": float("nan"),
            "baseline_ci_low_pct": float("nan"),
            "baseline_ci_high_pct": float("nan"),
            "candidate_pct": float("nan"),
            "candidate_ci_low_pct": float("nan"),
            "candidate_ci_high_pct": float("nan"),
            "pp_gain": float("nan"),
            "pp_gain_ci_low": float("nan"),
            "pp_gain_ci_high": float("nan"),
            "gap_closed_pct": float("nan"),
            "gap_closed_ci_low_pct": float("nan"),
            "gap_closed_ci_high_pct": float("nan"),
            "pop_baseline_pct": float("nan"),
            "pop_baseline_ci_low_pct": float("nan"),
            "pop_baseline_ci_high_pct": float("nan"),
            "pop_candidate_pct": float("nan"),
            "pop_candidate_ci_low_pct": float("nan"),
            "pop_candidate_ci_high_pct": float("nan"),
        }

    w = np.zeros_like(b, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != b.shape:
        raise ValueError(f"weights shape {w.shape} != baseline shape {b.shape}.")

    df = pd.DataFrame(
        {
            "country": c,
            "n": 1.0,
            "base": b,
            "cand": k,
            "pop": w,
            "pop_base": b * w,
            "pop_cand": k * w,
        }
    )
    grp = df.groupby("country", sort=True)[["n", "base", "cand", "pop", "pop_base", "pop_cand"]].sum()
    n_arr = grp["n"].to_numpy(dtype=np.float64)
    b_arr = grp["base"].to_numpy(dtype=np.float64)
    k_arr = grp["cand"].to_numpy(dtype=np.float64)
    p_arr = grp["pop"].to_numpy(dtype=np.float64)
    pb_arr = grp["pop_base"].to_numpy(dtype=np.float64)
    pk_arr = grp["pop_cand"].to_numpy(dtype=np.float64)

    n_sum = float(n_arr.sum())
    b_sum = float(b_arr.sum())
    k_sum = float(k_arr.sum())
    p_sum = float(p_arr.sum())
    base_pct = 100.0 * b_sum / n_sum
    cand_pct = 100.0 * k_sum / n_sum
    pp_gain = cand_pct - base_pct
    unalerted = n_sum - b_sum
    gap_closed_pct = (100.0 * (k_sum - b_sum) / unalerted) if unalerted > 0.0 else float("nan")
    pop_base_pct = (100.0 * float(pb_arr.sum()) / p_sum) if p_sum > 0.0 else float("nan")
    pop_cand_pct = (100.0 * float(pk_arr.sum()) / p_sum) if p_sum > 0.0 else float("nan")

    m = len(grp)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_resamples, m))
    nb = n_arr[idx].sum(axis=1)
    bb = b_arr[idx].sum(axis=1)
    kb = k_arr[idx].sum(axis=1)
    pb = p_arr[idx].sum(axis=1)
    pbb = pb_arr[idx].sum(axis=1)
    pkb = pk_arr[idx].sum(axis=1)

    base_s = 100.0 * bb / nb
    cand_s = 100.0 * kb / nb
    gain_s = cand_s - base_s
    unalerted_b = nb - bb
    gap_s = np.where(unalerted_b > 0.0, 100.0 * (kb - bb) / np.where(unalerted_b > 0.0, unalerted_b, 1.0), np.nan)

    alpha = 1.0 - confidence
    qs = [alpha / 2.0, 1.0 - alpha / 2.0]
    b_lo, b_hi = np.quantile(base_s, qs)
    k_lo, k_hi = np.quantile(cand_s, qs)
    g_lo, g_hi = np.quantile(gain_s, qs)
    gap_valid = gap_s[np.isfinite(gap_s)]
    gc_lo, gc_hi = np.quantile(gap_valid, qs) if len(gap_valid) > 0 else (float("nan"), float("nan"))

    if weights is not None and p_sum > 0.0:
        pop_b_s = np.where(pb > 0.0, 100.0 * pbb / np.where(pb > 0.0, pb, 1.0), np.nan)
        pop_k_s = np.where(pb > 0.0, 100.0 * pkb / np.where(pb > 0.0, pb, 1.0), np.nan)
        pb_lo, pb_hi = np.quantile(pop_b_s[np.isfinite(pop_b_s)], qs)
        pk_lo, pk_hi = np.quantile(pop_k_s[np.isfinite(pop_k_s)], qs)
    else:
        pb_lo = pb_hi = pk_lo = pk_hi = float("nan")

    return {
        "n_events": int(n_tot),
        **wilcox,
        "baseline_deff": deff_base,
        "baseline_n_eff": neff_base,
        "candidate_deff": deff_cand,
        "candidate_n_eff": neff_cand,
        "baseline_pct": float(base_pct),
        "baseline_ci_low_pct": float(b_lo),
        "baseline_ci_high_pct": float(b_hi),
        "candidate_pct": float(cand_pct),
        "candidate_ci_low_pct": float(k_lo),
        "candidate_ci_high_pct": float(k_hi),
        "pp_gain": float(pp_gain),
        "pp_gain_ci_low": float(g_lo),
        "pp_gain_ci_high": float(g_hi),
        "gap_closed_pct": float(gap_closed_pct),
        "gap_closed_ci_low_pct": float(gc_lo),
        "gap_closed_ci_high_pct": float(gc_hi),
        "pop_baseline_pct": float(pop_base_pct),
        "pop_baseline_ci_low_pct": float(pb_lo),
        "pop_baseline_ci_high_pct": float(pb_hi),
        "pop_candidate_pct": float(pop_cand_pct),
        "pop_candidate_ci_low_pct": float(pk_lo),
        "pop_candidate_ci_high_pct": float(pk_hi),
    }


def bootstrap_proportion_ci(
    indicators: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute percentile bootstrap (estimate, ci_low, ci_high) for a binary or weighted proportion."""
    arr = np.asarray(indicators, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if weights is None:
        point = float(arr.mean())
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != arr.shape:
            raise ValueError(f"weights shape {w.shape} != indicators shape {arr.shape}.")
        w_sum = float(w.sum())
        point = float((arr * w).sum() / w_sum) if w_sum > 0 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    if weights is None:
        samples = arr[idx].mean(axis=1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w_samp = w[idx]
        w_tot = w_samp.sum(axis=1)
        samples = np.where(w_tot > 0, (arr[idx] * w_samp).sum(axis=1) / np.where(w_tot > 0, w_tot, 1.0), 0.0)

    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def paired_bootstrap_diff_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute paired percentile bootstrap CI for (candidate - baseline) proportion difference."""
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    if b.shape != c.shape:
        raise ValueError(f"Shape mismatch: baseline {b.shape} vs candidate {c.shape}.")
    n = b.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diff = c - b
    return bootstrap_proportion_ci(diff, n_resamples=n_resamples, confidence=confidence, seed=seed, weights=weights)


def fit_lp3_return_periods(
    annual_maxima: np.ndarray,
    return_periods: Sequence[float] = RETURN_PERIODS,
    skew_clip: tuple[float, float] = (-2.0, 2.0),
    min_flow: float = 1e-3,
) -> np.ndarray:
    """Fit USGS Bulletin 17C Log-Pearson Type III (Wilson-Hilferty) return-period quantiles.

    Args:
        annual_maxima: Array of shape ``(n_years,)`` or ``(n_years, n_sites)`` containing
            annual peak streamflow values (NaNs are ignored per site).
        return_periods: Sequence of return periods in years (default canonical 8-tier grid).
        skew_clip: Lower/upper bounds on sample log-space skewness coefficient.
        min_flow: Positive floor applied before base-10 logarithm.

    Returns:
        Array of return-period thresholds with shape ``(n_rps,)`` for 1D input or
        ``(n_sites, n_rps)`` for 2D input.
    """
    arr = np.asarray(annual_maxima, dtype=np.float64)
    squeeze_1d = arr.ndim == 1
    if squeeze_1d:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"annual_maxima must be 1D or 2D, got shape {arr.shape}.")

    rps = np.asarray(return_periods, dtype=np.float64)
    if np.any(rps <= 1.0):
        raise ValueError("All return periods must be strictly greater than 1.0 year.")
    z_norms = stats.norm.ppf(1.0 - 1.0 / rps)  # (n_rps,)

    log_peaks = np.where(np.isfinite(arr), np.log10(np.maximum(arr, min_flow)), np.nan)
    valid = np.isfinite(log_peaks)
    n_v = valid.sum(axis=0).astype(np.float64)
    lp_z = np.where(valid, log_peaks, 0.0)
    mu = lp_z.sum(axis=0) / np.maximum(n_v, 1.0)
    dev = np.where(valid, lp_z - mu[None, :], 0.0)
    m2 = (dev**2).sum(axis=0) / np.maximum(n_v, 1.0)
    m3 = (dev**3).sum(axis=0) / np.maximum(n_v, 1.0)
    sd = np.sqrt(np.where(n_v > 1.0, m2 * n_v / np.maximum(n_v - 1.0, 1.0), 0.0))

    denom = np.power(np.maximum(m2, 1e-18), 1.5)
    g_raw = np.where(
        (n_v >= 3.0) & (m2 > 1e-12),
        (np.sqrt(n_v * (n_v - 1.0)) / np.maximum(n_v - 2.0, 1.0)) * (m3 / denom),
        0.0,
    )
    g = np.clip(g_raw, skew_clip[0], skew_clip[1])

    z = z_norms[:, None]  # (n_rps, 1)
    g2 = g[None, :]  # (1, n_sites)
    g_nz = np.where(np.abs(g2) > 1e-5, g2, 1.0)
    k_wh = np.where(
        np.abs(g2) > 1e-5,
        (2.0 / g_nz) * (np.power(np.maximum(1.0 + (g2 * z / 6.0) - (g2**2 / 36.0), 1e-6), 3) - 1.0),
        z,
    )
    q_rp = np.power(10.0, mu[None, :] + k_wh * sd[None, :]).T  # (n_sites, n_rps)
    q_rp = np.where((n_v[:, None] >= 2.0) & np.isfinite(q_rp) & (q_rp > 0), q_rp, np.nan)
    return q_rp[0] if squeeze_1d else q_rp


def format_p_value(p_val: float | None) -> str:
    """Format a p-value concisely for publication tables."""
    if p_val is None or not np.isfinite(p_val):
        return "—"
    if p_val == 0.0 or p_val < 1e-300:
        return "< 10^-300"
    exp = int(math.floor(math.log10(p_val)))
    mant = p_val / (10**exp)
    if mant >= 9.95:
        mant, exp = 1.0, exp + 1
    return f"{mant:.1f}×10^{exp}"
