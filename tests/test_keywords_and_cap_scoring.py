"""Unit tests for multilingual CAP keyword filtering and OASIS CAP v1.2 score64 invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cap_alerting_coverage.config import D_TOTAL, date_to_idx
from cap_alerting_coverage.keywords import (
    classify_cap_alert,
    classify_multi_hazard,
    compute_cap_score64,
    map_certainty,
    map_severity,
    map_urgency,
)
from cap_alerting_coverage.scoring import (
    build_fcover,
    build_pre_onset_fcover,
    build_strictly_before_fcover,
    prepare_catalog_slice,
    score_boolean_matrix,
    score_composite_alert_matrices,
    score_event_window_rules,
)


@pytest.mark.parametrize(
    ("event", "headline", "description", "tr_head", "tr_desc", "exp_perm", "exp_rest", "exp_excl"),
    [
        ("Flash Flood Warning", "Flash flood in valley", "", "", "", True, True, False),
        ("Heavy Rain Advisory", "Intense rainfall expected", "", "", "", True, False, False),
        ("High Wind Warning", "Damaging gusts up to 90 km/h", "Waterfront debris", "", "", False, False, True),
        ("Earthquake", "Magnitude 6.2 earthquake", "Riverbank damage", "", "", False, False, True),
        ("Alerte Météo", "Vigilance crue torrentielle", "", "", "", True, True, False),
        ("Aviso", "Lluvia intensa en provincia", "", "", "", True, False, False),
        ("Предупреждение", "Сильный паводок на реке", "", "", "", True, True, False),
        ("气象预警", "暴雨红色预警", "", "", "", True, False, False),
        ("هشدار جوی", "احتمال سیلاب و آبگرفتگی معابر", "", "", "", True, True, False),
        ("Peringatan Dini", "Waspada banjir bandang", "", "", "", True, True, False),
        ("Unwetterwarnung", "Hochwasser und Starkregen", "", "", "", True, True, False),
        ("Allerta", "Rischio esondazione fiume", "", "", "", True, True, False),
        ("Uyarı", "Şiddetli yağmur ve taşkın riski", "", "", "", True, True, False),
        ("ประกาศเตือนภัย", "เฝ้าระวังน้ำท่วมฉับพลัน", "", "", "", True, True, False),
        ("Cảnh báo", "Nguy cơ ngập lụt vùng trũng", "", "", "", True, True, False),
        ("기상특보", "호우경보 및 홍수 주의", "", "", "", True, True, False),
        ("चेतावनी", "बाढ़ और भारी बारिश की संभावना", "", "", "", True, True, False),
        # Translation fallback: raw event generic, translated headline has flood
        ("Alert", "General advisory", "", "Severe river flood warning", "", True, True, False),
        # Description-only broad match
        ("Weather Bulletin", "Regional notice", "Heavy precipitation expected tonight", "", "", True, False, False),
        # Exclusion in event takes precedence over broad description
        ("Forest Fire Warning", "Wildfire risk", "Do not waste water", "", "", False, False, True),
        # Explicit flood in headline still blocked if event is excluded hazard
        ("Snow Storm", "Flood risk from melting snow", "", "", "", False, False, True),
    ],
)
def test_multilingual_cap_classification(event, headline, description, tr_head, tr_desc, exp_perm, exp_rest, exp_excl):
    is_perm, is_rest, is_excl = classify_cap_alert(
        event=event,
        headline=headline,
        description=description,
        translated_headline=tr_head,
        translated_description=tr_desc,
    )
    assert (is_perm, is_rest, is_excl) == (exp_perm, exp_rest, exp_excl)


def test_oasis_score64_lexicographic_subsumption_proof():
    """Verify exact severity boundary cutoffs [1, 17, 33, 49] across all 64 OASIS CAP combinations."""
    boundaries = ((1, 1), (2, 17), (3, 33), (4, 49))
    seen = set()
    for sev in range(1, 5):
        for cert in range(1, 5):
            for urg in range(1, 5):
                s64 = compute_cap_score64(sev, cert, urg)
                assert 1 <= s64 <= 64
                seen.add(s64)
                for sev_cut, rank_cut in boundaries:
                    assert (s64 >= rank_cut) == (sev >= sev_cut)
    assert len(seen) == 64


def test_cap_field_ordinal_mappers():
    assert map_severity("Extreme") == 4
    assert map_severity("severe") == 3
    assert map_severity("Moderate") == 2
    assert map_severity("Minor") == 1
    assert map_severity(None) == 1
    assert map_certainty("Observed") == 4
    assert map_urgency("Immediate") == 4
    assert compute_cap_score64("Extreme", "Observed", "Immediate") == 64
    assert compute_cap_score64("Minor", "Unknown", "Unknown") == 1


def test_multi_hazard_classifier():
    hz = classify_multi_hazard(event="Wildfire warning", headline="Extreme heat wave", category="Fire")
    assert hz["wildfire"] is True
    assert hz["heat"] is True
    assert hz["earthquake"] is False


def test_event_window_rule_episode_precision_pre_onset_and_strictly_before():
    """PreOnset and StrictlyBefore score episode precision against narrowed onset masks."""
    gid_to_idx = {"TST.1_1": 0, "TST.2_1": 1}
    n_g = len(gid_to_idx)
    g_mask = np.ones(n_g, dtype=bool)

    cat_df = pd.DataFrame(
        [
            {
                "event_id": "E_AFTER",
                "gid_1": "TST.1_1",
                "country_iso3": "TST",
                "start_date": "2022-03-10",
                "end_date": "2022-03-14",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 100.0,
            },
            {
                "event_id": "E_ONSET",
                "gid_1": "TST.2_1",
                "country_iso3": "TST",
                "start_date": "2022-06-10",
                "end_date": "2022-06-14",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 100.0,
            },
        ]
    )
    c_sl = prepare_catalog_slice(cat_df, gid_to_idx, {"TST"}, "2021-01-01", "2025-12-31", lead_n=7)
    fcov = build_fcover(n_g, c_sl)
    fcov_pre = build_pre_onset_fcover(n_g, c_sl)
    fcov_before = build_strictly_before_fcover(n_g, c_sl)

    # Alerts occur ONLY during/after onset (start+2d for E_AFTER, exact start day for E_ONSET) + 1 FP episode.
    w_during_after = np.zeros((n_g, D_TOTAL), dtype=bool)
    w_during_after[0, date_to_idx("2022-03-12")] = True  # after onset only -> Primary TP only
    w_during_after[1, date_to_idx("2022-06-10")] = True  # exact onset day -> Primary + PreOnset TP
    w_during_after[0, date_to_idx("2023-01-15")] = True  # false alarm outside any event

    _, rows = score_event_window_rules(
        w_during_after,
        c_sl,
        fcov,
        fcov_pre,
        fcov_before,
        g_mask,
        cat_name="Without_GS",
        variant_name="CAP_Perm_All",
        lead_n=7,
        max_gap=2,
    )
    by_var = {r["variant"]: r for r in rows}
    prim = by_var["CAP_Perm_All__Primary_[start-7d,end]"]
    pre = by_var["CAP_Perm_All__PreOnset_[start-7d,start]"]
    before = by_var["CAP_Perm_All__StrictlyBefore_[start-7d,start-1d]"]

    assert prim["n_events"] == pre["n_events"] == before["n_events"] == 2
    assert prim["n_episodes"] == pre["n_episodes"] == before["n_episodes"] == 3
    assert (prim["tp_ev"], pre["tp_ev"], before["tp_ev"]) == (2, 1, 0)
    assert (prim["tp_ep"], pre["tp_ep"], before["tp_ep"]) == (2, 1, 0)
    assert prim["precision_pct"] > pre["precision_pct"] > before["precision_pct"]
    assert prim["precision_pct"] == pytest.approx(66.667, abs=1e-3)
    assert pre["precision_pct"] == pytest.approx(33.333, abs=1e-3)
    assert before["precision_pct"] == pytest.approx(0.0, abs=1e-6)

    # Boundary check: events with ev_s == 0 or ev_s <= ev_w0 are skipped in StrictlyBefore mask.
    edge_slice = {
        "ev_g": np.array([0, 1], dtype=np.int32),
        "ev_s": np.array([0, 10], dtype=np.int32),
        "ev_w0": np.array([0, 10], dtype=np.int32),
        "ev_w1": np.array([2, 12], dtype=np.int32),
    }
    edge_pre = build_pre_onset_fcover(2, edge_slice)
    edge_before = build_strictly_before_fcover(2, edge_slice)
    assert edge_pre[0, 0] and edge_pre[1, 10]
    assert int(edge_pre.sum()) == 2
    assert not np.any(edge_before)


def test_score_composite_alert_matrices_respects_1d_ffm_vs_7d_river():
    """FFM alerts at start-3d are rejected while start-1d and CAP/river start-5d are credited."""
    gid_to_idx = {"TST.1_1": 0, "TST.2_1": 1, "TST.3_1": 2}
    n_g = len(gid_to_idx)
    g_mask = np.ones(n_g, dtype=bool)

    cat_df = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "gid_1": "TST.1_1",
                "country_iso3": "TST",
                "start_date": "2022-04-10",
                "end_date": "2022-04-12",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 100.0,
            },
            {
                "event_id": "E2",
                "gid_1": "TST.2_1",
                "country_iso3": "TST",
                "start_date": "2022-05-10",
                "end_date": "2022-05-12",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 200.0,
            },
            {
                "event_id": "E3",
                "gid_1": "TST.3_1",
                "country_iso3": "TST",
                "start_date": "2022-06-10",
                "end_date": "2022-06-12",
                "has_idmc": True,
                "has_montandon": False,
                "has_groundsource": False,
                "waterfall_pop": 300.0,
            },
        ]
    )
    sl_7 = prepare_catalog_slice(cat_df, gid_to_idx, {"TST"}, "2021-01-01", "2025-12-31", lead_n=7)
    sl_1 = prepare_catalog_slice(cat_df, gid_to_idx, {"TST"}, "2021-01-01", "2025-12-31", lead_n=1)
    fc_7 = build_fcover(n_g, sl_7)
    fc_1 = build_fcover(n_g, sl_1)

    w_other = np.zeros((n_g, D_TOTAL), dtype=bool)
    w_ffm = np.zeros((n_g, D_TOTAL), dtype=bool)

    # E1: CAP/river fires at start-5d -> credited
    w_other[0, date_to_idx("2022-04-05")] = True
    # E2: FFM fires ONLY at start-3d -> must NOT be credited under 1d FFM rule (counts as FP episode)
    w_ffm[1, date_to_idx("2022-05-07")] = True
    # E3: FFM fires at start-1d -> credited
    w_ffm[2, date_to_idx("2022-06-09")] = True

    res = score_composite_alert_matrices(w_other, w_ffm, sl_7, fc_7, sl_1, fc_1, g_mask, max_gap=2)
    assert res["n_events"] == 3
    assert res["tp_ev"] == 2
    assert res["tp_pre"] == 2
    assert res["n_episodes"] == 3
    assert res["tp_ep"] == 2
    assert res["recall_pct"] == pytest.approx(66.667, abs=1e-3)
    assert res["precision_pct"] == pytest.approx(66.667, abs=1e-3)
    assert res["pop_recall_pct"] == pytest.approx(100.0 * 400.0 / 600.0, abs=1e-3)

    # Compare against uniform 7-day window: would over-credit E2
    res_uniform = score_boolean_matrix(w_other | w_ffm, sl_7, fc_7, g_mask, max_gap=2)
    assert res_uniform["tp_ev"] == 3
    assert res_uniform["tp_ep"] == 3
