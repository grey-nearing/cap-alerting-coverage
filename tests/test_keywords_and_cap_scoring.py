"""Unit tests for multilingual CAP keyword filtering and OASIS CAP v1.2 score64 invariants."""

from __future__ import annotations

import pytest

from cap_alerting_coverage.keywords import (
    classify_cap_alert,
    classify_multi_hazard,
    compute_cap_score64,
    map_certainty,
    map_severity,
    map_urgency,
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
