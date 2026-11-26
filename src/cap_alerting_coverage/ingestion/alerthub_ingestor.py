"""Portable, SSRF- and XXE-hardened CAP Alert Hub / Filtered Alert Hub ingestor (Tier 2)."""

from __future__ import annotations

import gzip
import io
import ipaddress
import json
import socket
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

from cap_alerting_coverage.ingestion.country_resolver import (
    make_canonical_alert_id,
    resolve_feed_and_agency,
)

MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}

COUNTRY_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "USA": {"lat": (24.0, 50.0), "lon": (-125.0, -65.0)},
    "AUS": {"lat": (-45.0, -10.0), "lon": (110.0, 155.0)},
    "SVN": {"lat": (45.4, 46.9), "lon": (13.3, 16.6)},
    "CAN": {"lat": (41.0, 84.0), "lon": (-141.0, -52.0)},
    "GBR": {"lat": (49.0, 61.0), "lon": (-9.0, 2.0)},
    "PHL": {"lat": (4.0, 21.0), "lon": (116.0, 127.0)},
    "COL": {"lat": (-4.0, 13.0), "lon": (-79.0, -66.0)},
    "MEX": {"lat": (14.0, 33.0), "lon": (-118.0, -86.0)},
    "IDN": {"lat": (-11.0, 6.0), "lon": (95.0, 141.0)},
}

ENGLISH_COUNTRIES = frozenset({"USA", "GBR", "AUS", "CAN", "NZL", "IRL", "ZAF", "JAM", "TTO", "GUY", "BLZ"})


def _validate_ip_address(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, host_label: str):
    """Unwrap IPv4-mapped IPv6 addresses and reject private/loopback/link-local/CGNAT targets."""
    effective = getattr(ip_obj, "ipv4_mapped", None) or ip_obj
    if (
        not effective.is_global
        or effective.is_private
        or effective.is_loopback
        or effective.is_link_local
        or effective.is_multicast
        or effective.is_reserved
        or effective.is_unspecified
    ):
        raise ValueError(f"Rejected non-public IP {effective} for host {host_label!r}")
    return effective


def validate_remote_url(url: str) -> str:
    """Validate an external HTTPS URL against SSRF to loopback, private, CGNAT, or metadata hosts."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise ValueError(f"Rejected non-HTTPS URL scheme: {parsed.scheme!r}")
    host_clean = (parsed.hostname or "").lower().rstrip(".")
    if not host_clean or host_clean in _BLOCKED_HOSTNAMES or host_clean.endswith((".internal", ".local", ".localhost")):
        raise ValueError(f"Rejected disallowed hostname: {host_clean!r}")
    try:
        addrs = [ipaddress.ip_address(host_clean)]
    except ValueError:
        infos = socket.getaddrinfo(host_clean, parsed.port or 443, type=socket.SOCK_STREAM)
        addrs = [ipaddress.ip_address(info[4][0]) for info in infos]
    if not addrs:
        raise ValueError(f"Could not resolve hostname: {host_clean!r}")
    for addr in addrs:
        _validate_ip_address(addr, host_clean)
    return url.strip()


def parse_polygon_centroid(polygon: str | None) -> tuple[float | None, float | None]:
    """Compute mean (lat, lon) from valid 'lat,lon' pairs in a CAP polygon string."""
    if not polygon:
        return None, None
    valid_pts: list[tuple[float, float]] = []
    for token in polygon.split():
        parts = token.strip().split(",")
        if len(parts) == 2:
            try:
                valid_pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not valid_pts:
        return None, None
    lat = sum(p[0] for p in valid_pts) / len(valid_pts)
    lon = sum(p[1] for p in valid_pts) / len(valid_pts)
    return lat, lon


def parse_circle_centroid(circle: str | None) -> tuple[float | None, float | None]:
    """Safely extract (lat, lon) from a CAP circle string ('lat,lon radius')."""
    if not circle:
        return None, None
    tokens = circle.split()
    if not tokens:
        return None, None
    coords = tokens[0].split(",")
    if len(coords) < 2:
        return None, None
    try:
        return float(coords[0]), float(coords[1])
    except ValueError:
        return None, None


def audit_coordinates(
    lat: float | None, lon: float | None, country_code: str | None
) -> tuple[float | None, float | None]:
    """Suppress (0, 0) null-island points and gross out-of-country coordinate errors."""
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        return None, None
    if country_code in COUNTRY_BOUNDS:
        bounds = COUNTRY_BOUNDS[country_code]
        min_lat, max_lat = bounds["lat"]
        min_lon, max_lon = bounds["lon"]
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return None, None
    return lat, lon


def detect_language_simple(country_code: str | None) -> str:
    """Assign initial ISO-639-2 language hint from issuing country code."""
    if country_code in ENGLISH_COUNTRIES:
        return "ENG"
    if country_code == "FRA":
        return "FRA"
    if country_code in {"ESP", "ARG", "CHL", "COL", "MEX", "PRY", "URY", "PER", "ECU"}:
        return "SPA"
    if country_code in {"DEU", "AUT", "CHE"}:
        return "DEU"
    if country_code in {"BRA", "PRT"}:
        return "POR"
    if country_code in {"RUS", "UKR", "BLR"}:
        return "RUS"
    if country_code == "CHN":
        return "ZHO"
    if country_code == "JPN":
        return "JPN"
    if country_code == "KOR":
        return "KOR"
    return "OTH"


def _parse_xml_safe(raw_xml: str | bytes):
    from defusedxml import ElementTree

    return ElementTree.fromstring(
        raw_xml,
        forbid_dtd=True,
        forbid_entities=True,
        forbid_external=True,
    )


def parse_ndjson_alert_line(line_str: str, source_site: str = "Filtered_Alert_Hub") -> dict[str, Any] | None:
    """Parse one NDJSON Alert Hub archive line into a normalized CAP database row dict."""
    try:
        data = json.loads(line_str)
    except json.JSONDecodeError:
        return None
    alert = data.get("alert", {})
    raw_xml = data.get("capXml") or data.get("entry") or alert.get("capXml")
    identifier = alert.get("identifier")
    if not identifier and raw_xml:
        try:
            root = _parse_xml_safe(raw_xml)
            el = root.find("{*}identifier")
            if el is not None and el.text:
                identifier = el.text.strip()
        except Exception:
            pass
    if not identifier:
        return None

    sender = alert.get("sender")
    feed_code = alert.get("ownerFeedCode") or data.get("source", {}).get("ownerFeedCode")
    link_url = alert.get("link") or data.get("source", {}).get("link")
    resolved_feed_id, agency_id, country_code = resolve_feed_and_agency(
        sender, identifier=identifier, source_feed_id=feed_code
    )

    polygon_list = alert.get("polygon") or []
    polygon_str = polygon_list[0] if polygon_list else None
    lat, lon = parse_polygon_centroid(polygon_str)
    lat, lon = audit_coordinates(lat, lon, country_code)

    sent_at_dt = None
    sent_str = alert.get("sent")
    if sent_str:
        try:
            from dateutil import parser as dt_parser

            sent_at_dt = dt_parser.parse(sent_str)
            if sent_at_dt.tzinfo is None:
                sent_at_dt = sent_at_dt.replace(tzinfo=timezone.utc)
        except Exception:
            sent_at_dt = None

    event = alert.get("event")
    headline = alert.get("headline")
    description = alert.get("description")
    instruction = alert.get("instruction")
    if not event and not description and not headline:
        return None

    if raw_xml and (not description or not headline):
        try:
            root = _parse_xml_safe(raw_xml)
            info = root.find("{*}info")
            if info is not None:
                for field_name, tag in (
                    ("headline", "headline"),
                    ("description", "description"),
                    ("instruction", "instruction"),
                ):
                    if locals()[field_name]:
                        continue
                    el = info.find(f"{{*}}{tag}")
                    if el is not None and el.text:
                        if field_name == "headline":
                            headline = el.text.strip()
                        elif field_name == "description":
                            description = el.text.strip()
                        else:
                            instruction = el.text.strip()
        except Exception:
            pass

    cats = alert.get("category") or []
    areas = alert.get("areaDesc") or []
    return {
        "id": make_canonical_alert_id(resolved_feed_id, sender, identifier),
        "source_site": source_site,
        "identifier": identifier,
        "sender": sender,
        "sent_at": sent_at_dt,
        "status": alert.get("status"),
        "msg_type": alert.get("msgType"),
        "raw_xml": raw_xml,
        "created_at": datetime.now(timezone.utc),
        "event": event,
        "severity": alert.get("severity"),
        "urgency": alert.get("urgency"),
        "certainty": alert.get("certainty"),
        "scope": alert.get("scope"),
        "category": cats[0] if isinstance(cats, list) and cats else (cats if isinstance(cats, str) else None),
        "headline": headline,
        "description": description,
        "instruction": instruction,
        "area_desc": ", ".join(areas)
        if isinstance(areas, list) and areas
        else (areas if isinstance(areas, str) else None),
        "polygon": polygon_str,
        "latitude": lat,
        "longitude": lon,
        "country_code": country_code,
        "source_language": detect_language_simple(country_code),
        "is_from_cap": True,
        "is_from_filtered_hub": source_site == "Filtered_Alert_Hub",
        "is_from_ifrc": source_site == "IFRC_Alert_Hub",
        "is_from_google": False,
        "source_feed_id": resolved_feed_id,
        "agency_id": agency_id,
        "source_url": link_url,
    }


def iter_ndjson_archive(uri: str) -> Iterator[dict[str, Any]]:
    """Stream parsed CAP alert records from a local or fsspec-addressed ``.ndjson`` / ``.ndjson.gz`` URI."""
    import fsspec

    with fsspec.open(uri, "rb") as raw_fh:
        if uri.endswith(".gz"):
            stream: io.TextIOBase = io.TextIOWrapper(gzip.GzipFile(fileobj=raw_fh, mode="rb"), encoding="utf-8")
        else:
            stream = io.TextIOWrapper(raw_fh, encoding="utf-8")
        for line in stream:
            rec = parse_ndjson_alert_line(line)
            if rec is not None:
                yield rec
