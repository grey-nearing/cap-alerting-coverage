"""Unit tests for LP3 return periods, exact McNemar, bootstrap CIs, validation audit, and schemas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cap_alerting_coverage.config import RETURN_PERIODS
from cap_alerting_coverage.pipeline.extract_glofas_v5 import score_glofas_pixel_timeseries
from cap_alerting_coverage.schemas import (
    assert_no_forbidden_columns,
    validate_cap_gid1_matrix,
    validate_deidentified_validation_table,
    validate_event_catalog,
    validate_ffm_gid1_matrix,
    validate_public_summary_table,
)
from cap_alerting_coverage.stats_utils import (
    bootstrap_proportion_ci,
    clopper_pearson_ci,
    cluster_bootstrap_proportion_ci,
    cluster_design_effect,
    country_clustered_coverage_inference,
    country_paired_wilcoxon,
    fisher_exact_2x2,
    fit_lp3_return_periods,
    format_p_value,
    mcnemar_exact_p,
    paired_bootstrap_diff_ci,
    two_proportion_z_test,
    wald_ci,
)
from cap_alerting_coverage.validation_audit import (
    ValidationCounts,
    build_validation_summary_table,
    counts_from_deidentified_table,
)


def test_lp3_monotonicity_and_glofas_flow_filter():
    rng = np.random.default_rng(123)
    ann_max = rng.lognormal(mean=4.5, sigma=0.45, size=(45, 3))
    ann_max[0, 1] = np.nan  # missing year tolerated cleanly
    q_rps = fit_lp3_return_periods(ann_max, return_periods=RETURN_PERIODS)
    assert q_rps.shape == (3, len(RETURN_PERIODS))
    assert np.all(np.isfinite(q_rps))
    # Return-period levels must increase strictly with return period
    assert np.all(np.diff(q_rps, axis=1) > 0)

    # Verify <10 m^3/s pixel filtering in GloFAS scorer
    n_days = 45 * 365
    yr_indices = [np.arange(y * 365, (y + 1) * 365) for y in range(45)]
    idx_target = np.arange(n_days - 30, n_days)
    q_mat = np.column_stack(
        [
            np.full(n_days, 2.0),  # below 10 m^3/s -> dropped
            np.full(n_days, 50.0) + rng.exponential(scale=25.0, size=n_days),  # kept
        ]
    )
    q_mat[-1, 1] = 5000.0  # extreme peak on final day -> tier 8
    keep, tiers = score_glofas_pixel_timeseries(q_mat, yr_indices, idx_target, min_mean_flow_m3s=10.0)
    assert list(keep) == [False, True]
    assert tiers.shape == (30, 1)
    assert int(tiers[-1, 0]) == 8


def test_exact_mcnemar_and_validation_statistics():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert abs(mcnemar_exact_p(10, 0) - 2.0 / 1024.0) < 1e-12
    assert mcnemar_exact_p(2362, 0) == 0.0
    assert format_p_value(mcnemar_exact_p(2362, 0)) == "< 10^-300"

    # Extended Data Table 4 verification (N=400 v4_reviewed counts)
    _, p_val = two_proportion_z_test(208, 235, 220, 400)
    assert p_val < 1e-16
    lo_cp, hi_cp = clopper_pearson_ci(220, 220)
    assert abs(lo_cp - 0.9834) < 0.001 and hi_cp == 1.0
    lo_w, hi_w = wald_ci(208, 235)
    assert abs(100 * lo_w - 84.43) < 0.1 and abs(100 * hi_w - 92.59) < 0.1

    # ValidationCounts must not provide hard-coded defaults
    with pytest.raises(TypeError):
        ValidationCounts()  # type: ignore[call-arg]

    summary = build_validation_summary_table(
        ValidationCounts(
            n_total=400,
            n_country_only_tp=86,
            n_exclusion_tp=220,
            n_inclusion_selected=235,
            n_inclusion_tp=208,
        )
    )
    c0_row = summary[summary["strategy"].str.startswith("Country-Only")].iloc[0]
    excl_row = summary[summary["strategy"] == "Exclusion Filter"].iloc[0]
    incl_row = summary[summary["strategy"] == "Inclusion Filter"].iloc[0]
    assert np.isnan(c0_row["recall_pct"])
    assert np.isnan(c0_row["recall_wald_ci_low_pct"])
    assert np.isnan(c0_row["recall_cp_ci_low_pct"])
    assert excl_row["recall_wald_ci_low_pct"] == 100.0
    assert excl_row["recall_wald_ci_high_pct"] == 100.0
    assert abs(excl_row["recall_cp_ci_low_pct"] - 98.34) < 0.05
    assert abs(incl_row["precision_pct"] - 88.51) < 0.05
    assert abs(incl_row["recall_pct"] - 94.55) < 0.05


def test_bootstrap_and_paired_bootstrap_cis():
    base = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0] * 20, dtype=float)
    cand = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1] * 20, dtype=float)
    weights = np.linspace(1.0, 5.0, len(base))

    est, lo, hi = bootstrap_proportion_ci(base, n_resamples=500, seed=7)
    assert 0.0 <= lo <= est <= hi <= 1.0
    assert abs(est - 0.30) < 1e-9

    d_est, d_lo, d_hi = paired_bootstrap_diff_ci(base, cand, n_resamples=500, seed=7, weights=weights)
    assert d_lo > 0.0 and d_lo <= d_est <= d_hi


def test_fisher_exact_deff_country_wilcoxon_and_clustered_cis():
    # Disjoint N=400 v4_reviewed filter validation: 208/235 selected vs 12/165 excluded
    odds, p_fisher = fisher_exact_2x2(208, 235, 12, 165)
    assert odds > 50.0
    assert abs(p_fisher - 8.289e-66) < 1e-67

    # Strongly clustered binary indicators across 20 countries (100% in half, 0% in half)
    countries = np.repeat([f"C{i:02d}" for i in range(20)], 10)
    base = np.concatenate([np.ones(100), np.zeros(100)])
    cand = np.concatenate([np.ones(100), np.tile([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 10)])
    pop = np.linspace(100.0, 5000.0, len(countries))

    deff, n_eff = cluster_design_effect(countries, base)
    assert deff > 5.0
    assert abs(n_eff - len(countries) / deff) < 1e-9

    est, lo, hi = cluster_bootstrap_proportion_ci(countries, base, n_resamples=500, seed=42)
    assert abs(est - 0.5) < 1e-9
    assert 0.0 <= lo < est < hi <= 1.0
    # Deterministic replay with identical seed
    assert cluster_bootstrap_proportion_ci(countries, base, n_resamples=500, seed=42) == (est, lo, hi)

    w_res = country_paired_wilcoxon(countries, base, cand)
    assert w_res["n_countries"] == 20
    assert w_res["n_improved"] == 10
    assert w_res["n_tied"] == 10
    assert w_res["n_declined"] == 0
    assert abs(float(w_res["mean_pp_gain"]) - 35.0) < 1e-9
    assert float(w_res["wilcoxon_p"]) < 0.01

    # Identical arrays yield p=1.0 cleanly without error
    tied_res = country_paired_wilcoxon(countries, base, base)
    assert tied_res["wilcoxon_p"] == 1.0 and tied_res["n_tied"] == 20

    inf = country_clustered_coverage_inference(
        country_ids=countries,
        baseline=base,
        candidate=cand,
        weights=pop,
        n_resamples=500,
        seed=42,
    )
    assert inf["n_events"] == 200
    assert inf["n_countries"] == 20
    assert inf["baseline_deff"] > 5.0
    assert inf["baseline_ci_low_pct"] <= inf["baseline_pct"] <= inf["baseline_ci_high_pct"]
    assert inf["candidate_ci_low_pct"] <= inf["candidate_pct"] <= inf["candidate_ci_high_pct"]
    assert inf["gap_closed_ci_low_pct"] <= inf["gap_closed_pct"] <= inf["gap_closed_ci_high_pct"]
    assert inf["pop_baseline_ci_low_pct"] <= inf["pop_baseline_pct"] <= inf["pop_baseline_ci_high_pct"]
    assert inf["pop_candidate_ci_low_pct"] <= inf["pop_candidate_pct"] <= inf["pop_candidate_ci_high_pct"]


def test_schema_and_privacy_validators_catch_violations():
    with pytest.raises(ValueError, match="forbidden columns"):
        assert_no_forbidden_columns(["gid_1", "headline"], "test_table")

    bad_cat = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "start_date": "2024-05-10",
                "end_date": "2024-05-09",  # end < start
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 10.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="end_date < start_date"):
        validate_event_catalog(bad_cat)

    bad_cap = pd.DataFrame(
        [
            {
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "alert_date": "2024-05-10",
                "perm_max_sev": 4,
                "rest_max_sev": 0,
                "perm_max_score64": 40,  # passes sev>=3, violates >=49 for sev=4
                "rest_max_score64": 0,
                "n_perm": 1,
                "n_rest": 0,
            }
        ]
    )
    with pytest.raises(ValueError, match="score64>=49"):
        validate_cap_gid1_matrix(bad_cap)

    bad_montandon_audit = pd.DataFrame(
        [{"iso3": "KEN", "unit": "houses", "description": "Raw third-party Montandon event narrative"}]
    )
    with pytest.raises(ValueError, match="Privacy/governance violation"):
        validate_public_summary_table(bad_montandon_audit, dataset_name="montandon_nonhuman_unit_audit.csv")

    bad_ffm = pd.DataFrame([{"gid_1": "USA.1_1", "country_iso3": "USA", "valid_date": "2024-05-10", "max_prob": 1.2}])
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        validate_ffm_gid1_matrix(bad_ffm)

    bad_val = pd.DataFrame(
        [{"sample_id": "1", "alert_text": "raw text", "is_true_flood_alert": True, "matches_inclusion_filter": True}]
    )
    with pytest.raises(ValueError, match="Privacy violation"):
        validate_deidentified_validation_table(bad_val)

    good_val = pd.DataFrame(
        [
            {
                "sample_id": "1",
                "is_true_flood_alert": True,
                "matches_inclusion_filter": True,
                "is_country_only_tp": True,
            },
            {
                "sample_id": "2",
                "is_true_flood_alert": False,
                "matches_inclusion_filter": False,
                "is_country_only_tp": False,
            },
        ]
    )
    counts = counts_from_deidentified_table(good_val)
    assert counts.n_total == 2 and counts.n_inclusion_tp == 1
