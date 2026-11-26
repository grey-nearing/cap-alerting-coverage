"""Statistical inference, bootstrap CIs, and Log-Pearson Type III return-period utilities."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
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
