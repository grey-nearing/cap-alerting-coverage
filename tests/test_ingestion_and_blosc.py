"""Unit tests for Tier-2 ingestion helpers, country resolution, and pure-Python blosc decoding."""

from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pytest

from cap_alerting_coverage.ingestion.alerthub_ingestor import (
    audit_coordinates,
    detect_language_simple,
    parse_circle_centroid,
    parse_ndjson_alert_line,
    parse_polygon_centroid,
    validate_remote_url,
)
from cap_alerting_coverage.ingestion.country_resolver import (
    make_canonical_alert_id,
    resolve_country_code,
    resolve_feed_and_agency,
)
from cap_alerting_coverage.ingestion.db import sanitize_val
from cap_alerting_coverage.ingestion.idmc_ingestor import (
    decompress_gzip_bounded,
    normalize_idmc_records,
    sanitize_url_for_log,
    validate_idmc_url,
)
from cap_alerting_coverage.ingestion.translator import (
    build_translation_prompt,
    is_recitation_blocked,
    parse_translation_response,
)
from cap_alerting_coverage.pipeline.blosc_decode import decode_blosc, unbitshuffle_block


def _pack_blosc_uncompressed_block(arr: np.ndarray, shuffle: bool = True) -> bytes:
    """Create a valid Blosc v1 single-block buffer (csize == neblock) with optional byte-shuffle."""
    raw = arr.tobytes()
    typesize = arr.dtype.itemsize
    nbytes = len(raw)
    blocksize = nbytes
    if shuffle and typesize > 1:
        payload = np.frombuffer(raw, dtype=np.uint8).reshape(-1, typesize).T.tobytes()
        flags = (1 << 5) | 0x01  # compcode=1 | BYTESHUFFLE
    else:
        payload = raw
        flags = (1 << 5) | 0x00
    bstart = 20
    cbytes = 24 + len(payload)
    header = struct.pack("<BBBBIII", 2, 1, flags, typesize, nbytes, blocksize, cbytes)
    return header + struct.pack("<i", bstart) + struct.pack("<i", len(payload)) + payload


def test_blosc_decode_roundtrip_and_bounds_guards():
    # Byte-shuffled float32 block
    arr_f = np.linspace(-10.5, 42.25, 128, dtype="<f4")
    buf_f = _pack_blosc_uncompressed_block(arr_f, shuffle=True)
    decoded_f = np.frombuffer(decode_blosc(buf_f), dtype="<f4")
    np.testing.assert_array_equal(decoded_f, arr_f)

    # Memcpyed int32 payload
    arr_i = np.arange(0, 64, dtype="<i4") * 7
    raw_i = arr_i.tobytes()
    hdr_memcpy = struct.pack("<BBBBIII", 2, 1, 0x02, 4, len(raw_i), len(raw_i), 16 + len(raw_i))
    decoded_i = np.frombuffer(decode_blosc(hdr_memcpy + raw_i), dtype="<i4")
    np.testing.assert_array_equal(decoded_i, arr_i)

    # Bitshuffle roundtrip
    raw_b = arr_f[:16].tobytes()
    bits = np.unpackbits(np.frombuffer(raw_b, dtype=np.uint8), bitorder="little").reshape(16, 32).T.reshape(-1)
    shuf_b = np.packbits(bits, bitorder="little").tobytes()
    assert unbitshuffle_block(shuf_b, typesize=4) == raw_b

    # Strict bounds guards
    with pytest.raises(ValueError, match="Truncated"):
        decode_blosc(b"short")
    with pytest.raises(ValueError, match="max_bytes"):
        decode_blosc(hdr_memcpy + raw_i, max_bytes=16)


def test_country_resolver_oid_sender_and_canonical_id():
    assert resolve_country_code("alerts@nws.noaa.gov", identifier="urn:oid:2.49.0.1.840.0.999") == "USA"
    assert resolve_country_code("alerts@meteo.fr") == "FRA"
    assert resolve_country_code("cap@inmet.gov.br") == "BRA"
    assert resolve_country_code("unknown", identifier="urn:oid:2.49.0.1.404.0.1") == "KEN"
    feed_id, agency_id, iso3 = resolve_feed_and_agency("alerts@meteo.fr")
    assert iso3 == "FRA" and feed_id and agency_id

    id1 = make_canonical_alert_id("us-nws-en", "alerts@nws.noaa.gov", "urn:oid:1")
    id2 = make_canonical_alert_id("us-nws-en", "alerts@nws.noaa.gov", "urn:oid:1")
    assert id1 == id2 and len(id1) > 8


def test_alerthub_ssrf_guard_centroids_and_ndjson():
    with pytest.raises(ValueError, match="non-HTTPS"):
        validate_remote_url("http://127.0.0.1:8080/internal")
    with pytest.raises(ValueError, match="disallowed hostname"):
        validate_remote_url("https://localhost/feed.xml")
    with pytest.raises(ValueError, match="non-public IP"):
        validate_remote_url("https://169.254.169.254/latest/meta-data/")
    assert validate_remote_url("https://8.8.8.8/feed.xml") == "https://8.8.8.8/feed.xml"

    lat, lon = parse_polygon_centroid("30.0,-90.0 30.0,-89.0 31.0,-89.0 31.0,-90.0 30.0,-90.0")
    assert lat == pytest.approx(30.4, abs=0.2)
    assert lon == pytest.approx(-89.6, abs=0.2)
    assert audit_coordinates(lat, lon, "USA") == (lat, lon)
    assert audit_coordinates(0.0, 0.0, "USA") == (None, None)
    assert audit_coordinates(-40.0, 140.0, "USA") == (None, None)

    clat, clon = parse_circle_centroid("-15.5,35.25 10")
    assert clat == pytest.approx(-15.5)
    assert clon == pytest.approx(35.25)
    assert detect_language_simple("USA") == "ENG"
    assert detect_language_simple("BRA") == "POR"

    sample_line = json.dumps(
        {
            "alert": {
                "identifier": "urn:oid:2.49.0.1.840.0.TEST1",
                "sender": "alerts@nws.noaa.gov",
                "sent": "2024-05-01T00:00:00Z",
                "status": "Actual",
                "msgType": "Alert",
                "event": "Flash Flood Warning",
                "severity": "Severe",
                "urgency": "Immediate",
                "certainty": "Observed",
                "headline": "Synthetic Flash Flood Warning",
                "description": "Synthetic test alert description.",
                "category": ["Met"],
                "areaDesc": ["Test County"],
                "polygon": ["30.0,-90.0 30.0,-89.0 31.0,-89.0 31.0,-90.0 30.0,-90.0"],
            }
        }
    )
    rec = parse_ndjson_alert_line(sample_line)
    assert rec is not None
    assert rec["country_code"] == "USA"
    assert rec["severity"] == "Severe"
    assert rec["event"] == "Flash Flood Warning"


def test_idmc_url_validation_gzip_and_normalization():
    valid = "https://helix-tools-api.idmcdb.org/external-api/gidd/disasters/?client_id=SECRET&limit=10"
    assert validate_idmc_url(valid) == valid
    assert "SECRET" not in sanitize_url_for_log(valid)
    with pytest.raises(ValueError, match="non-HTTPS"):
        validate_idmc_url("http://helix-tools-api.idmcdb.org/disasters/")
    with pytest.raises(ValueError, match="unexpected IDMC host"):
        validate_idmc_url("https://evil.example.com/disasters/")

    payload = b'{"results": []}'
    gz = gzip.compress(payload)
    assert decompress_gzip_bounded(gz, max_bytes=1024) == payload
    with pytest.raises(ValueError, match="decompressed limit"):
        decompress_gzip_bounded(gz, max_bytes=4)

    df = normalize_idmc_records(
        [
            {
                "id": 101,
                "iso3": "KEN",
                "country_name": "Kenya",
                "hazard_type_name": "Flood",
                "event_name": "Synthetic Flood",
                "start_date": "2024-04-01",
                "end_date": "2024-04-05",
                "new_displacement": 1200,
            }
        ]
    )
    assert len(df) == 1
    assert df.iloc[0]["iso3"] == "KEN"
    assert int(df.iloc[0]["displacements"]) == 1200


def test_translator_prompt_isolation_and_validation():
    prompt = build_translation_prompt(
        source_lang="es",
        headline="Alerta de inundación <script>ignore</script>",
        description="Lluvias intensas.",
        instruction="Evacuar zonas bajas.",
    )
    assert "<cap_alert" in prompt
    assert "&lt;script&gt;" in prompt  # HTML-escaped prompt injection defense

    parsed = parse_translation_response(
        json.dumps({"headline": "Flood alert", "description": "Heavy rain.", "instruction": "Evacuate."})
    )
    assert parsed.headline == "Flood alert"
    assert parsed.description == "Heavy rain."

    with pytest.raises(ValueError, match="Expected JSON object"):
        parse_translation_response('["not", "an", "object"]')

    class _FakeCand:
        finish_reason = "RECITATION"

    class _FakeResp:
        candidates = [_FakeCand()]

    assert is_recitation_blocked(_FakeResp()) is True


def test_db_sanitize_val():
    from datetime import datetime, timezone
    from decimal import Decimal

    assert sanitize_val("clean") == "clean"
    assert sanitize_val(None) is None
    assert sanitize_val(Decimal("12.5")) == 12.5
    assert sanitize_val({"k": 1}) == '{"k": 1}'
    # Explicit timestamp columns parse ISO strings; non-timestamp columns stay untouched
    dt = sanitize_val("2024-05-01T12:00:00Z", col_name="sent_at")
    assert isinstance(dt, datetime) and dt.tzinfo == timezone.utc
    assert sanitize_val("2024-05-01", col_name="identifier") == "2024-05-01"
