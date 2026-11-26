"""Unit tests for CAP >=5% GADM overlap geocoding, <=7d clustering, waterfall, and Catalog B merge."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from cap_alerting_coverage.clustering import (
    cluster_gid1_episodes,
    collapse_warning_episodes,
    compute_waterfall_population,
    merge_strictly_nested_catalog_b,
)
from cap_alerting_coverage.pipeline.extract_cap_alerts import CapAdmin1Geocoder, parse_cap_polygon


def _make_two_province_geocoder() -> CapAdmin1Geocoder:
    # Two adjacent 10x10 degree provinces in a synthetic country "TST"
    g1 = box(0.0, 0.0, 10.0, 10.0)  # area = 100
    g2 = box(10.0, 0.0, 20.0, 10.0)  # area = 100
    adm1 = gpd.GeoDataFrame(
        [
            {
                "gid_1": "TST.1_1",
                "country_iso3": "TST",
                "COUNTRY": "Testland",
                "area_deg2": 100.0,
                "country_area_deg2": 200.0,
                "n_units_in_country": 2,
                "geometry": g1,
            },
            {
                "gid_1": "TST.2_1",
                "country_iso3": "TST",
                "COUNTRY": "Testland",
                "area_deg2": 100.0,
                "country_area_deg2": 200.0,
                "n_units_in_country": 2,
                "geometry": g2,
            },
        ],
        crs="EPSG:4326",
    )
    gaz = {
        "adm1_alias": {"TST": {"alphaland": ["TST.1_1"], "betaland": ["TST.2_1"]}},
        "adm1_native": {},
        "adm2_alias": {},
        "adm2_native": {},
        "single_gid1": {},
        "abbrev_map": {},
        "c0_cents": {"TST": [[10.0, 5.0]]},
    }
    return CapAdmin1Geocoder(adm1, gaz)


def test_cap_polygon_5pct_overlap_and_whole_country_rejection():
    geocoder = _make_two_province_geocoder()

    # 1. Polygon mostly in TST.1_1 with only 2% sliver in TST.2_1 -> maps ONLY to TST.1_1
    # lon in [0.2, 10.2], lat in [0, 10] -> total area = 100, overlap with TST.2_1 = 0.2*10 = 2 (< 5% of 100)
    poly_sliver = "0,0.2 0,10.2 10,10.2 10,0.2 0,0.2"
    assert geocoder.resolve_polygon("TST", poly_sliver) == ("TST.1_1",)

    # 2. Polygon with 6% overlap in TST.2_1 -> maps to BOTH TST.1_1 and TST.2_1
    # lon in [0.6, 10.6], lat in [0, 10] -> overlap with TST.2_1 = 6 (>= 5% of 100)
    poly_both = "0,0.6 0,10.6 10,10.6 10,0.6 0,0.6"
    assert geocoder.resolve_polygon("TST", poly_both) == ("TST.1_1", "TST.2_1")

    # 3. Whole-country polygon covering >=80% of 2-unit country -> REJECTED (zero Admin-0 fallback!)
    # lon in [0, 19], lat in [0, 10] -> area = 190 / 200 = 95% >= 80%
    poly_countrywide = "0,0 0,19 10,19 10,0 0,0"
    assert geocoder.resolve_polygon("TST", poly_countrywide) == ()


def test_cap_point_centroid_rejection_and_coastal_nearest():
    geocoder = _make_two_province_geocoder()
    # Generic country centroid (lat=5.0, lon=10.0) must be rejected
    assert geocoder.resolve_point("TST", 5.01, 10.01) == ()
    # Interior point in TST.1_1 matches
    assert geocoder.resolve_point("TST", 5.0, 4.0) == ("TST.1_1",)
    # Coastal point 0.10 deg offshore west of TST.1_1 matches via <=0.15 deg fallback
    assert geocoder.resolve_point("TST", 5.0, -0.10) == ("TST.1_1",)
    # Far ocean point (>0.15 deg away) rejected
    assert geocoder.resolve_point("TST", 5.0, -0.50) == ()
    # Null island (0, 0) rejected
    assert geocoder.resolve_point("TST", 0.0, 0.0) == ()


def test_parse_cap_polygon_antimeridian_and_malformed():
    assert parse_cap_polygon("10,20 10,21 11,21 11,20") is not None
    assert parse_cap_polygon("10,-170 10,170 11,170 11,-170") is None  # spans >180 deg
    assert parse_cap_polygon("not_a_polygon") is None


def test_population_waterfall_modes():
    # Priority fallback (baseline paper code): num_affected wins if > 0 even when displacements > affected
    assert compute_waterfall_population(100, 500, 5, pop_rule="priority_fallback") == 100.0
    assert compute_waterfall_population(0, 500, 5, pop_rule="priority_fallback") == 500.0
    assert compute_waterfall_population(0, 0, 5, pop_rule="priority_fallback") == 5.0
    # Pure max sensitivity mode: max(affected, displaced) without killed fallback
    assert compute_waterfall_population(100, 500, 5, pop_rule="max") == 500.0
    assert compute_waterfall_population(0, 0, 5, pop_rule="max") == 0.0
    # Max with killed fallback mode
    assert compute_waterfall_population(100, 500, 5, pop_rule="max_with_killed_fallback") == 500.0
    assert compute_waterfall_population(0, 0, 5, pop_rule="max_with_killed_fallback") == 5.0
    with pytest.raises(ValueError):
        compute_waterfall_population(10, 20, 1, pop_rule="invalid")  # type: ignore[arg-type]


def test_per_gid1_7day_clustering_and_strictly_nested_catalog_b_merge():
    raw = pd.DataFrame(
        [
            # Same gid_1 separated by 5 days -> merge into 1 episode (2024-06-01 .. 2024-06-07)
            {
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "start_date": "2024-06-01",
                "end_date": "2024-06-02",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "displacements": 500.0,
                "num_affected": 0.0,
                "num_killed": 0.0,
            },
            {
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "start_date": "2024-06-07",
                "end_date": "2024-06-07",
                "has_idmc": False,
                "has_montandon": True,
                "has_groundsource": False,
                "displacements": 0.0,
                "num_affected": 120.0,
                "num_killed": 2.0,
            },
            # Same gid_1 8 days after end -> separate episode
            {
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "start_date": "2024-06-16",
                "end_date": "2024-06-16",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "displacements": 50.0,
                "num_affected": 0.0,
                "num_killed": 0.0,
            },
        ]
    )
    eps_prio = cluster_gid1_episodes(raw, gap_days=7, pop_rule="priority_fallback")
    eps_max = cluster_gid1_episodes(raw, gap_days=7, pop_rule="max")
    assert len(eps_prio) == 2
    assert eps_prio.loc[0, "waterfall_pop"] == 120.0
    assert eps_max.loc[0, "waterfall_pop"] == 500.0

    # Add cohort flags & test non-nanosecond datetime64[s]/[us] strict nesting
    cat_a = eps_prio.copy()
    cat_a["start_date"] = cat_a["start_date"].astype("datetime64[s]")
    cat_a["end_date"] = cat_a["end_date"].astype("datetime64[s]")
    cat_a["in_cohort_d_35"] = False
    cat_a["in_eval_150"] = True
    cat_a["in_audited_185"] = True
    cat_a["in_orig_excl_19"] = False

    gs_daily = pd.DataFrame(
        [
            # Within +-7d of first Catalog A episode on USA.1_1 -> flags parent, NOT appended
            {
                "gid_1": "USA.1_1",
                "country_iso3": "USA",
                "start_date": pd.Timestamp("2024-06-10"),
                "end_date": pd.Timestamp("2024-06-10"),
                "has_idmc": False,
                "has_montandon": False,
                "has_groundsource": True,
                "displacements": 0.0,
                "num_affected": 0.0,
                "num_killed": 0.0,
            },
            # Different gid_1 -> appended as standalone GS_ event with 0 pop
            {
                "gid_1": "USA.2_1",
                "country_iso3": "USA",
                "start_date": pd.Timestamp("2024-06-10"),
                "end_date": pd.Timestamp("2024-06-10"),
                "has_idmc": False,
                "has_montandon": False,
                "has_groundsource": True,
                "displacements": 0.0,
                "num_affected": 0.0,
                "num_killed": 0.0,
            },
        ]
    )
    gs_daily["start_date"] = gs_daily["start_date"].astype("datetime64[us]")
    gs_daily["end_date"] = gs_daily["end_date"].astype("datetime64[us]")

    cat_b, gs_unmatched = merge_strictly_nested_catalog_b(
        cat_a, gs_daily, cohort_d_35=set(), eval_150={"USA"}, audited_185={"USA"}
    )
    assert len(cat_b) == 3
    assert len(gs_unmatched) == 1
    assert abs(cat_b["waterfall_pop"].sum() - cat_a["waterfall_pop"].sum()) < 1e-9
    assert bool(cat_b.loc[0, "has_groundsource"]) is True


def test_collapse_warning_episodes():
    days = [10, 11, 13, 20, 22, 27, 40]
    assert len(collapse_warning_episodes(days, max_gap_days=0)) == 6
    assert len(collapse_warning_episodes(days, max_gap_days=2)) == 4
    assert len(collapse_warning_episodes(days, max_gap_days=7)) == 2
