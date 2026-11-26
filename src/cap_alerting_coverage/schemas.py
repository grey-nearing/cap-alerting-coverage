"""Strict I/O schema validation and non-redistribution privacy guardrails."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from cap_alerting_coverage.config import D_TOTAL

FORBIDDEN_PUBLIC_COLUMNS: frozenset[str] = frozenset(
    {
        "raw_xml",
        "headline",
        "description",
        "instruction",
        "translated_headline",
        "translated_description",
        "translated_instruction",
        "source_url",
        "url",
        "sender",
        "author_email",
        "author_name",
        "polygon",
        "polygon_geojson",
        "geometry",
        "article_text",
        "news_text",
        "groundsource_summary",
        "raw_discharge",
        "streamflow_m3s",
    }
)

CATALOG_REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "gid_1",
    "country_iso3",
    "start_date",
    "end_date",
    "has_idmc",
    "has_montandon",
    "has_groundsource",
    "waterfall_pop",
)

CAP_GID1_REQUIRED_COLUMNS: tuple[str, ...] = (
    "gid_1",
    "country_iso3",
    "alert_date",
    "perm_max_sev",
    "rest_max_sev",
    "perm_max_score64",
    "rest_max_score64",
    "n_perm",
    "n_rest",
)

CAP_GID0_REQUIRED_COLUMNS: tuple[str, ...] = (
    "country_iso3",
    "alert_date",
    "perm_max_sev",
    "rest_max_sev",
    "perm_max_score64",
    "rest_max_score64",
    "n_perm",
    "n_rest",
)

FFM_GID1_REQUIRED_COLUMNS: tuple[str, ...] = (
    "gid_1",
    "country_iso3",
    "valid_date",
    "max_prob",
)


def assert_no_forbidden_columns(columns: Iterable[str], dataset_name: str) -> None:
    """Reject any raw CAP text, URL, geometry, PII, or raw discharge column in derived public tables."""
    col_lower = {str(c).strip().lower(): str(c) for c in columns}
    leaked = sorted(col_lower[k] for k in (set(col_lower) & FORBIDDEN_PUBLIC_COLUMNS))
    if leaked:
        raise ValueError(f"Privacy/governance violation in {dataset_name}: forbidden columns present: {leaked}")


def _assert_required_columns(df: pd.DataFrame, required: tuple[str, ...], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Schema mismatch in {dataset_name}: missing required columns {missing}")


def validate_event_catalog(df: pd.DataFrame, dataset_name: str = "event_catalog") -> None:
    """Validate Catalog A / Catalog B schema, date ordering, non-negative population, and privacy."""
    assert_no_forbidden_columns(df.columns, dataset_name)
    _assert_required_columns(df, CATALOG_REQUIRED_COLUMNS, dataset_name)
    if df.empty:
        return
    if df["gid_1"].isna().any() or df["country_iso3"].isna().any():
        raise ValueError(f"{dataset_name} contains null gid_1 or country_iso3 values.")
    sd = pd.to_datetime(df["start_date"])
    ed = pd.to_datetime(df["end_date"])
    if (ed < sd).any():
        raise ValueError(f"{dataset_name} has rows with end_date < start_date.")
    pop = pd.to_numeric(df["waterfall_pop"], errors="coerce")
    if pop.isna().any() or (pop < 0).any():
        raise ValueError(f"{dataset_name} has null or negative waterfall_pop values.")


def validate_cap_gid1_matrix(df: pd.DataFrame, dataset_name: str = "cap_gid1_daily") -> None:
    """Validate derived CAP Admin-1 daily exceedance table and OASIS score64 boundary subsumption."""
    assert_no_forbidden_columns(df.columns, dataset_name)
    _assert_required_columns(df, CAP_GID1_REQUIRED_COLUMNS, dataset_name)
    if df.empty:
        return
    for col in ("perm_max_sev", "rest_max_sev"):
        vals = df[col].to_numpy()
        if np.any((vals < 0) | (vals > 4)):
            raise ValueError(f"{dataset_name}.{col} out of valid range [0, 4].")
    for col in ("perm_max_score64", "rest_max_score64"):
        vals = df[col].to_numpy()
        if np.any((vals < 0) | (vals > 64)):
            raise ValueError(f"{dataset_name}.{col} out of valid range [0, 64].")
    for sev_v, sc_v in ((1, 1), (2, 17), (3, 33), (4, 49)):
        if not np.array_equal(df["perm_max_sev"].to_numpy() >= sev_v, df["perm_max_score64"].to_numpy() >= sc_v):
            raise ValueError(f"{dataset_name}: perm severity>={sev_v} does not match score64>={sc_v}.")
        if not np.array_equal(df["rest_max_sev"].to_numpy() >= sev_v, df["rest_max_score64"].to_numpy() >= sc_v):
            raise ValueError(f"{dataset_name}: rest severity>={sev_v} does not match score64>={sc_v}.")


def validate_cap_gid0_matrix(df: pd.DataFrame, dataset_name: str = "cap_gid0_daily") -> None:
    """Validate derived CAP Admin-0 daily sensitivity audit table."""
    assert_no_forbidden_columns(df.columns, dataset_name)
    _assert_required_columns(df, CAP_GID0_REQUIRED_COLUMNS, dataset_name)


def validate_ffm_gid1_matrix(df: pd.DataFrame, dataset_name: str = "ffm_gid1_daily") -> None:
    """Validate derived FFM Admin-1 daily max-probability table."""
    assert_no_forbidden_columns(df.columns, dataset_name)
    _assert_required_columns(df, FFM_GID1_REQUIRED_COLUMNS, dataset_name)
    if df.empty:
        return
    probs = df["max_prob"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"{dataset_name}.max_prob contains values outside [0, 1].")


def validate_tier_cube(
    npz_obj: dict[str, np.ndarray],
    expected_ndim: int,
    gid1_list: np.ndarray,
    dataset_name: str,
    expected_days: int | None = D_TOTAL,
) -> None:
    """Validate a riverine exceedance tier NPZ archive (discrete uint8 tiers 0..8 only)."""
    for req_key in ("tiers", "gid_1", "country_iso3"):
        if req_key not in npz_obj:
            raise ValueError(f"{dataset_name} missing required NPZ array '{req_key}'.")
    tiers = npz_obj["tiers"]
    if tiers.ndim != expected_ndim:
        raise ValueError(f"{dataset_name}.tiers expected {expected_ndim}D, got shape {tiers.shape}.")
    if not np.array_equal(npz_obj["gid_1"], gid1_list):
        raise ValueError(f"{dataset_name}.gid_1 ordering does not match reference GADM Admin-1 index.")
    if expected_days is not None and tiers.shape[1] != expected_days:
        raise ValueError(f"{dataset_name}.tiers expected {expected_days} days on axis 1, got {tiers.shape[1]}.")
    if tiers.size > 0 and (int(tiers.min()) < 0 or int(tiers.max()) > 8):
        raise ValueError(f"{dataset_name}.tiers contains values outside valid discrete tier range [0, 8].")


DEIDENTIFIED_VALIDATION_COLUMNS: frozenset[str] = frozenset(
    {
        "sample_id",
        "sample_round",
        "country_area_stratum",
        "cohort_d_excluded_country",
        "event_year",
        "alert_year",
        "event_in_study_period_2021_2025",
        "alert_lead_window",
        "alert_group_id",
        "alert_shared_with_other_rows",
        "alert_cap_severity",
        "event_hazard_type",
        "keyword_filter_status_sheet",
        "spatial_match_v1",
        "thematic_match_v1",
        "spatial_match_v3",
        "thematic_match_v3",
        "alert_type_v3",
        "spatial_match_v4",
        "thematic_match_v4",
        "alert_type_v4",
        "reviewer2_reviewed_v4",
        "reviewer2_verdict_v4",
        "occurrences_in_v4_v6",
        "reviewer2_reviewed_v4_v6",
        "filter02_event_excluded",
        "filter02_broad_flood_storm",
        "filter02_broad_flood_storm_with_sheet_translation",
        "filter02_explicit_flood",
        "annex_inclusion_keyword_any_text",
    }
)

_SIMPLE_VALIDATION_COLUMNS: frozenset[str] = frozenset(
    {
        "sample_id",
        "is_true_flood_alert",
        "matches_inclusion_filter",
        "is_country_only_tp",
    }
)


def validate_public_summary_table(df: pd.DataFrame, dataset_name: str = "summary_csv") -> None:
    """Validate that a public summary or audit CSV (e.g., Montandon unit audit) contains no forbidden raw fields."""
    assert_no_forbidden_columns(df.columns, dataset_name)


def validate_deidentified_validation_table(df: pd.DataFrame, dataset_name: str = "validation_audit") -> None:
    """Validate that a public de-identified N=400 validation table contains zero verbatim CAP text or PII."""
    assert_no_forbidden_columns(df.columns, dataset_name)
    safe_cols = DEIDENTIFIED_VALIDATION_COLUMNS | _SIMPLE_VALIDATION_COLUMNS
    for col in df.columns:
        low = str(col).strip().lower()
        if low in safe_cols:
            continue
        if any(
            tok in low
            for tok in (
                "text",
                "headline",
                "desc",
                "xml",
                "url",
                "sender",
                "snippet",
                "area",
                "note",
                "comment",
                "uuid",
                "email",
                "reviewer_name",
            )
        ):
            raise ValueError(f"Privacy violation in {dataset_name}: suspicious column '{col}'.")

    # Content-level guard: verify no free text, URLs, emails, or XML snippets leaked into cells.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        if (vals.str.len() > 64).any():
            raise ValueError(f"Privacy violation in {dataset_name}.{col}: cell exceeds 64-char categorical limit.")
        if vals.str.contains(r"https?://|@|<alert|<info", case=False, regex=True).any():
            raise ValueError(f"Privacy violation in {dataset_name}.{col}: URL, email, or XML marker detected.")
