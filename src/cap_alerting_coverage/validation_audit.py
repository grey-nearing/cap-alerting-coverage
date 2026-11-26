"""Deterministic tabulation of the de-identified N=400 CAP validation audit.

Supports the full 30-column de-identified public schema (``validation_n400_deidentified.csv``:
boolean/categorical audit labels across review rounds, strata, and filter predictors) as well as
minimal boolean audit fixtures. Enforces strict non-redistribution checks so no raw CAP text,
identifiers, URLs, or reviewer notes are ever ingested or emitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cap_alerting_coverage.schemas import (
    DEIDENTIFIED_VALIDATION_COLUMNS,
    validate_deidentified_validation_table,
)
from cap_alerting_coverage.stats_utils import (
    clopper_pearson_ci,
    two_proportion_z_test,
    wald_ci,
)

Z_95: float = 1.959963984540054
LEGACY_FPC_N: int = 8508


@dataclass(frozen=True)
class ValidationCounts:
    """Aggregate confusion-matrix counts for an N-row CAP alert validation audit."""

    n_total: int
    n_country_only_tp: int
    n_exclusion_tp: int
    n_inclusion_selected: int
    n_inclusion_tp: int


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score confidence interval for a binomial proportion (in [0, 1])."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wald_ci_pct(k: int, n: int, fpc_n: int | None = None, z: float = Z_95) -> tuple[float, float]:
    """Wald 95% confidence interval in percentage points, with optional finite-population factor."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    hw = z * math.sqrt(p * (1.0 - p) / n)
    if fpc_n is not None and fpc_n > 1:
        hw *= math.sqrt((fpc_n - n) / (fpc_n - 1))
    return (100.0 * (p - hw), 100.0 * (p + hw))


def _r4(val: float | None) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), 4)


def _to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "tp", "valid", "match"})


def is_full_n400_schema(df: pd.DataFrame) -> bool:
    """Return True if ``df`` matches the full 30-column de-identified N=400 audit schema."""
    return DEIDENTIFIED_VALIDATION_COLUMNS.issubset(set(df.columns))


def load_deidentified_validation_table(path: str | Path) -> pd.DataFrame:
    """Load and privacy-validate a de-identified N=400 CSV or Parquet table."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        for col in df.columns:
            uniq = set(df[col].unique())
            if uniq and uniq <= {"True", "False"}:
                df[col] = df[col].eq("True")
        int_cols = [
            "sample_id",
            "sample_round",
            "event_year",
            "alert_year",
            "alert_group_id",
            "occurrences_in_v4_v6",
        ]
        for col in int_cols:
            if col in df.columns and df[col].astype(str).str.isdigit().all():
                df[col] = df[col].astype(int)
    validate_deidentified_validation_table(df, dataset_name=p.name)
    return df


def _label_versions(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    ones = pd.Series(1, index=df.index)
    all_rows = pd.Series(True, index=df.index)
    return {
        "v1_initial200": (df["thematic_match_v1"].eq("Yes"), ones, df["sample_round"].eq(1), df["spatial_match_v1"]),
        "v3": (df["thematic_match_v3"].eq("Yes"), ones, all_rows, df["spatial_match_v3"]),
        "v4_reviewed": (df["thematic_match_v4"].eq("Yes"), ones, all_rows, df["spatial_match_v4"]),
        "v4_v6_raw403": (
            df["thematic_match_v4"].eq("Yes"),
            df["occurrences_in_v4_v6"].astype(int),
            all_rows,
            df["spatial_match_v4"],
        ),
    }


def _predictors(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "exclusion_keep_all_sampled": pd.Series(True, index=df.index),
        "sheet_keyword_filter_status_Both": df["keyword_filter_status_sheet"].eq("Both"),
        "filter02_not_event_excluded": ~_to_bool_series(df["filter02_event_excluded"]),
        "filter02_broad_flood_storm": _to_bool_series(df["filter02_broad_flood_storm"]),
        "filter02_broad_flood_storm_with_sheet_translation": _to_bool_series(
            df["filter02_broad_flood_storm_with_sheet_translation"]
        ),
        "filter02_explicit_flood": _to_bool_series(df["filter02_explicit_flood"]),
        "annex_inclusion_keyword_any_text": _to_bool_series(df["annex_inclusion_keyword_any_text"]),
    }


def _subsets(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=df.index),
        "round1_rows1_200": df["sample_round"].eq(1),
        "round2_rows201_400": df["sample_round"].eq(2),
        "stratum_large_gt_500k_km2": df["country_area_stratum"].eq("large_gt_500k_km2"),
        "stratum_small_medium_le_500k_km2": df["country_area_stratum"].eq("small_medium_le_500k_km2"),
        "study_frame_2021_2025_and_150_countries": (
            _to_bool_series(df["event_in_study_period_2021_2025"]) & ~_to_bool_series(df["cohort_d_excluded_country"])
        ),
    }


def compute_confusion_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute confusion matrices (version × subset × predictor) with Wald, Wilson, and CP 95% CIs."""
    validate_deidentified_validation_table(df, dataset_name="validation_n400_deidentified")
    if not is_full_n400_schema(df):
        raise ValueError("Full 30-column validation_n400_deidentified schema is required.")

    rows = []
    for vname, (truth, weights, vmask, _) in _label_versions(df).items():
        for sname, smask in _subsets(df).items():
            mask = vmask & smask
            for pname, pred in _predictors(df).items():
                ww = weights[mask]
                t, p = truth[mask], pred[mask]
                tp = int(ww[p & t].sum())
                fp = int(ww[p & ~t].sum())
                fn = int(ww[~p & t].sum())
                tn = int(ww[~p & ~t].sum())
                npos = tp + fp
                ntrue = tp + fn
                nrows = int(ww.sum())

                pw = wald_ci_pct(tp, npos)
                pwi_raw = wilson_ci(tp, npos)
                pwi = (100.0 * pwi_raw[0], 100.0 * pwi_raw[1])
                pcp_raw = clopper_pearson_ci(tp, npos) if npos > 0 else (float("nan"), float("nan"))
                pcp = (100.0 * pcp_raw[0], 100.0 * pcp_raw[1])

                rw = wald_ci_pct(tp, ntrue)
                rcp_raw = clopper_pearson_ci(tp, ntrue) if ntrue > 0 else (float("nan"), float("nan"))
                rcp = (100.0 * rcp_raw[0], 100.0 * rcp_raw[1])

                rows.append(
                    {
                        "label_version": vname,
                        "subset": sname,
                        "predictor": pname,
                        "n_rows": nrows,
                        "n_predicted_positive": npos,
                        "n_truth_positive": ntrue,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "precision_pct": _r4(100.0 * tp / npos if npos else np.nan),
                        "recall_pct": _r4(100.0 * tp / ntrue if ntrue else np.nan),
                        "precision_wald_lo": _r4(pw[0]),
                        "precision_wald_hi": _r4(pw[1]),
                        "precision_wilson_lo": _r4(pwi[0]),
                        "precision_wilson_hi": _r4(pwi[1]),
                        "precision_cp_lo": _r4(pcp[0]),
                        "precision_cp_hi": _r4(pcp[1]),
                        "recall_wald_lo": _r4(rw[0]),
                        "recall_wald_hi": _r4(rw[1]),
                        "recall_cp_lo": _r4(rcp[0]),
                        "recall_cp_hi": _r4(rcp[1]),
                    }
                )
    return pd.DataFrame(rows)


def compute_spatial_precision_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute country-level pairing spatial precision (Yes / Partial / No) per version & subset."""
    validate_deidentified_validation_table(df, dataset_name="validation_n400_deidentified")
    if not is_full_n400_schema(df):
        raise ValueError("Full 30-column validation_n400_deidentified schema is required.")

    rows = []
    for vname, (_, weights, vmask, spatial) in _label_versions(df).items():
        for sname, smask in _subsets(df).items():
            mask = vmask & smask
            ww = weights[mask]
            n = int(ww.sum())
            yes = int(ww[spatial[mask].eq("Yes")].sum())
            partial = int(ww[spatial[mask].eq("Partial")].sum())
            wl = wald_ci_pct(yes, n)
            wfpc = wald_ci_pct(yes, n, fpc_n=LEGACY_FPC_N)
            cp_raw = clopper_pearson_ci(yes, n) if n > 0 else (float("nan"), float("nan"))
            cp = (100.0 * cp_raw[0], 100.0 * cp_raw[1])
            rows.append(
                {
                    "label_version": vname,
                    "subset": sname,
                    "n_rows": n,
                    "spatial_yes": yes,
                    "spatial_partial": partial,
                    "spatial_no": n - yes - partial,
                    "precision_yes_pct": _r4(100.0 * yes / n if n else np.nan),
                    "precision_yes_or_partial_pct": _r4(100.0 * (yes + partial) / n if n else np.nan),
                    "wald_halfwidth_pct": _r4((wl[1] - wl[0]) / 2.0) if n else None,
                    "wald_fpc8508_halfwidth_pct": _r4((wfpc[1] - wfpc[0]) / 2.0) if n else None,
                    "cp_lo": _r4(cp[0]),
                    "cp_hi": _r4(cp[1]),
                }
            )
    return pd.DataFrame(rows)


def counts_from_deidentified_table(
    df: pd.DataFrame,
    label_version: str = "v4_reviewed",
    predictor: str = "filter02_broad_flood_storm",
) -> ValidationCounts:
    """Compute ``ValidationCounts`` deterministically from a de-identified audit DataFrame.

    Supports both:
      1. The full 30-column ``validation_n400_deidentified.csv`` schema (selecting ``label_version``
         in ``{"v1_initial200", "v3", "v4_reviewed", "v4_v6_raw403"}`` and ``predictor``).
      2. A minimal boolean audit DataFrame with ``is_true_flood_alert``,
         ``matches_inclusion_filter``, and optional ``is_country_only_tp``.
    """
    validate_deidentified_validation_table(df, dataset_name="validation_audit_deidentified")

    if is_full_n400_schema(df):
        versions = _label_versions(df)
        preds = _predictors(df)
        if label_version not in versions:
            raise ValueError(f"Unknown label_version '{label_version}'. Choose from {sorted(versions)}.")
        if predictor not in preds:
            raise ValueError(f"Unknown predictor '{predictor}'. Choose from {sorted(preds)}.")
        truth, weights, vmask, spatial = versions[label_version]
        pred = preds[predictor]
        ww = weights[vmask]
        t, p, sp = truth[vmask], pred[vmask], spatial[vmask]
        return ValidationCounts(
            n_total=int(ww.sum()),
            n_country_only_tp=int(ww[sp.eq("Yes")].sum()),
            n_exclusion_tp=int(ww[t].sum()),
            n_inclusion_selected=int(ww[p].sum()),
            n_inclusion_tp=int(ww[p & t].sum()),
        )

    required = {"is_true_flood_alert", "matches_inclusion_filter"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"De-identified validation table missing required columns: {missing}")

    n_total = int(len(df))
    y_true = _to_bool_series(df["is_true_flood_alert"])
    y_incl = _to_bool_series(df["matches_inclusion_filter"])
    n_excl_tp = int(y_true.sum())
    n_incl_sel = int(y_incl.sum())
    n_incl_tp = int((y_true & y_incl).sum())
    n_country_tp = int(_to_bool_series(df["is_country_only_tp"]).sum()) if "is_country_only_tp" in df.columns else 0

    return ValidationCounts(
        n_total=n_total,
        n_country_only_tp=n_country_tp,
        n_exclusion_tp=n_excl_tp,
        n_inclusion_selected=n_incl_sel,
        n_inclusion_tp=n_incl_tp,
    )


def build_validation_summary_table(counts: ValidationCounts) -> pd.DataFrame:
    """Build Extended Data Table 4 metrics with Clopper-Pearson, Wilson, and Wald 95% CIs."""
    n = counts.n_total
    excl_tp = counts.n_exclusion_tp
    incl_n = counts.n_inclusion_selected
    incl_tp = counts.n_inclusion_tp
    c0_tp = counts.n_country_only_tp

    c0_prec = 100.0 * c0_tp / max(1, n)
    c0_cp_lo, c0_cp_hi = clopper_pearson_ci(c0_tp, n) if n > 0 else (0.0, 1.0)
    c0_w_lo, c0_w_hi = wald_ci(c0_tp, n) if n > 0 else (0.0, 1.0)
    c0_wil_lo, c0_wil_hi = wilson_ci(c0_tp, n) if n > 0 else (0.0, 1.0)

    excl_prec = 100.0 * excl_tp / max(1, n)
    excl_w_lo, excl_w_hi = wald_ci(excl_tp, n) if n > 0 else (0.0, 1.0)
    excl_cp_lo, excl_cp_hi = clopper_pearson_ci(excl_tp, n) if n > 0 else (0.0, 1.0)
    excl_wil_lo, excl_wil_hi = wilson_ci(excl_tp, n) if n > 0 else (0.0, 1.0)
    excl_rec_w_lo, excl_rec_w_hi = wald_ci(excl_tp, excl_tp) if excl_tp > 0 else (float("nan"), float("nan"))
    excl_rec_cp_lo, excl_rec_cp_hi = (
        clopper_pearson_ci(excl_tp, excl_tp) if excl_tp > 0 else (float("nan"), float("nan"))
    )

    incl_prec = 100.0 * incl_tp / max(1, incl_n)
    incl_w_lo, incl_w_hi = wald_ci(incl_tp, incl_n) if incl_n > 0 else (0.0, 1.0)
    incl_cp_lo, incl_cp_hi = clopper_pearson_ci(incl_tp, incl_n) if incl_n > 0 else (0.0, 1.0)
    incl_wil_lo, incl_wil_hi = wilson_ci(incl_tp, incl_n) if incl_n > 0 else (0.0, 1.0)

    incl_rec = 100.0 * incl_tp / max(1, excl_tp)
    incl_rec_w_lo, incl_rec_w_hi = wald_ci(incl_tp, excl_tp) if excl_tp > 0 else (0.0, 1.0)
    incl_rec_cp_lo, incl_rec_cp_hi = clopper_pearson_ci(incl_tp, excl_tp) if excl_tp > 0 else (0.0, 1.0)

    z_stat, p_val = two_proportion_z_test(incl_tp, incl_n, excl_tp, n) if (incl_n > 0 and n > 0) else (0.0, 1.0)
    max_moe_n = 100.0 * Z_95 * math.sqrt(0.25 / max(1, n))
    max_moe_incl = 100.0 * Z_95 * math.sqrt(0.25 / max(1, incl_n))

    rows = [
        {
            "strategy": "Country-Only Matching (Admin-0 Baseline)",
            "sample_n": n,
            "true_positives": c0_tp,
            "precision_pct": round(c0_prec, 2),
            "precision_wald_ci_low_pct": round(100.0 * c0_w_lo, 2),
            "precision_wald_ci_high_pct": round(100.0 * c0_w_hi, 2),
            "precision_wilson_ci_low_pct": round(100.0 * c0_wil_lo, 2),
            "precision_wilson_ci_high_pct": round(100.0 * c0_wil_hi, 2),
            "precision_cp_ci_low_pct": round(100.0 * c0_cp_lo, 2),
            "precision_cp_ci_high_pct": round(100.0 * c0_cp_hi, 2),
            "recall_pct": float("nan"),
            "recall_wald_ci_low_pct": float("nan"),
            "recall_wald_ci_high_pct": float("nan"),
            "recall_cp_ci_low_pct": float("nan"),
            "recall_cp_ci_high_pct": float("nan"),
            "max_margin_of_error_pct": round(max_moe_n, 2),
            "precision_diff_vs_exclusion_pp": round(c0_prec - excl_prec, 2),
            "two_prop_z_stat_vs_exclusion": float("nan"),
            "two_prop_p_vs_exclusion": float("nan"),
        },
        {
            "strategy": "Exclusion Filter",
            "sample_n": n,
            "true_positives": excl_tp,
            "precision_pct": round(excl_prec, 2),
            "precision_wald_ci_low_pct": round(100.0 * excl_w_lo, 2),
            "precision_wald_ci_high_pct": round(100.0 * excl_w_hi, 2),
            "precision_wilson_ci_low_pct": round(100.0 * excl_wil_lo, 2),
            "precision_wilson_ci_high_pct": round(100.0 * excl_wil_hi, 2),
            "precision_cp_ci_low_pct": round(100.0 * excl_cp_lo, 2),
            "precision_cp_ci_high_pct": round(100.0 * excl_cp_hi, 2),
            "recall_pct": 100.0 if excl_tp > 0 else float("nan"),
            "recall_wald_ci_low_pct": round(100.0 * excl_rec_w_lo, 2) if excl_tp > 0 else float("nan"),
            "recall_wald_ci_high_pct": round(100.0 * excl_rec_w_hi, 2) if excl_tp > 0 else float("nan"),
            "recall_cp_ci_low_pct": round(100.0 * excl_rec_cp_lo, 2) if excl_tp > 0 else float("nan"),
            "recall_cp_ci_high_pct": round(100.0 * excl_rec_cp_hi, 2) if excl_tp > 0 else float("nan"),
            "max_margin_of_error_pct": round(max_moe_n, 2),
            "precision_diff_vs_exclusion_pp": 0.0,
            "two_prop_z_stat_vs_exclusion": 0.0,
            "two_prop_p_vs_exclusion": 1.0,
        },
        {
            "strategy": "Inclusion Filter",
            "sample_n": incl_n,
            "true_positives": incl_tp,
            "precision_pct": round(incl_prec, 2),
            "precision_wald_ci_low_pct": round(100.0 * incl_w_lo, 2),
            "precision_wald_ci_high_pct": round(100.0 * incl_w_hi, 2),
            "precision_wilson_ci_low_pct": round(100.0 * incl_wil_lo, 2),
            "precision_wilson_ci_high_pct": round(100.0 * incl_wil_hi, 2),
            "precision_cp_ci_low_pct": round(100.0 * incl_cp_lo, 2),
            "precision_cp_ci_high_pct": round(100.0 * incl_cp_hi, 2),
            "recall_pct": round(incl_rec, 2),
            "recall_wald_ci_low_pct": round(100.0 * incl_rec_w_lo, 2),
            "recall_wald_ci_high_pct": round(100.0 * incl_rec_w_hi, 2),
            "recall_cp_ci_low_pct": round(100.0 * incl_rec_cp_lo, 2),
            "recall_cp_ci_high_pct": round(100.0 * incl_rec_cp_hi, 2),
            "max_margin_of_error_pct": round(max_moe_incl, 2),
            "precision_diff_vs_exclusion_pp": round(incl_prec - excl_prec, 2),
            "two_prop_z_stat_vs_exclusion": round(z_stat, 4),
            "two_prop_p_vs_exclusion": p_val,
        },
    ]
    return pd.DataFrame(rows)


def run_validation_audit(
    deidentified_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    label_version: str = "v4_reviewed",
    predictor: str = "filter02_broad_flood_storm",
) -> pd.DataFrame:
    """Tabulate the de-identified N=400 validation audit from an explicit CSV or Parquet file.

    Raises ``FileNotFoundError`` if ``deidentified_csv`` is omitted or does not exist (never
    falls back to hard-coded summary counts). When ``deidentified_csv`` uses the full 30-column
    ``validation_n400_deidentified.csv`` schema and ``output_csv`` is specified, also writes
    ``validation_confusion_counts.csv`` and ``validation_spatial_precision.csv`` in the same
    output directory.
    """
    if deidentified_csv is None:
        raise FileNotFoundError(
            "deidentified_csv is required; pass the path to "
            "04_validation_deidentified/validation_n400_deidentified.csv."
        )
    in_path = Path(deidentified_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"De-identified validation table not found: {in_path}")

    df_raw = load_deidentified_validation_table(in_path)
    counts = counts_from_deidentified_table(df_raw, label_version=label_version, predictor=predictor)

    summary = build_validation_summary_table(counts)
    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_p, index=False)
        if is_full_n400_schema(df_raw):
            compute_confusion_matrix_table(df_raw).to_csv(out_p.parent / "validation_confusion_counts.csv", index=False)
            compute_spatial_precision_table(df_raw).to_csv(
                out_p.parent / "validation_spatial_precision.csv", index=False
            )
    return summary
