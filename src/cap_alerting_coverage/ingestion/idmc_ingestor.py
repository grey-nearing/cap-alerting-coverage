"""Portable, origin-locked IDMC Helix GIDD Disasters API ingestor (Tier 2)."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import pandas as pd

ALLOWED_IDMC_HOST = "helix-tools-api.idmcdb.org"
MAX_COMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_PAGES = 500


def sanitize_url_for_log(url: str) -> str:
    """Strip query parameters (including ``client_id``) and fragments before logging."""
    parts = urllib.parse.urlsplit(str(url or ""))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def validate_idmc_url(url: str) -> str:
    """Ensure every pagination URL uses HTTPS on the exact IDMC GIDD API host."""
    parts = urllib.parse.urlsplit(str(url or "").strip())
    if parts.scheme != "https":
        raise ValueError(f"Rejected non-HTTPS IDMC URL scheme: {parts.scheme!r}")
    if (parts.hostname or "").lower() != ALLOWED_IDMC_HOST:
        raise ValueError(f"Rejected unexpected IDMC host: {parts.hostname!r}")
    return str(url).strip()


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        validate_idmc_url(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


_URL_OPENER = urllib.request.build_opener(_SameOriginRedirectHandler())


def _read_bounded(response, max_bytes: int = MAX_COMPRESSED_BYTES) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Response exceeded {max_bytes} byte compressed limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def decompress_gzip_bounded(raw_data: bytes, max_bytes: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    """Decompress a gzip payload with strict output byte ceiling."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = decompressor.decompress(raw_data, max_bytes + 1)
    if len(out) > max_bytes or decompressor.unconsumed_tail:
        raise ValueError(f"Gzip payload exceeded {max_bytes} byte decompressed limit.")
    tail = decompressor.flush()
    if len(out) + len(tail) > max_bytes:
        raise ValueError(f"Gzip payload exceeded {max_bytes} byte decompressed limit.")
    return out + tail


def fetch_idmc_page(url: str) -> dict:
    """Fetch and parse one JSON page from the IDMC GIDD API."""
    validated_url = validate_idmc_url(url)
    req = urllib.request.Request(
        validated_url,
        headers={"User-Agent": "cap-alerting-coverage/1.0", "Accept-Encoding": "gzip"},
    )
    with _URL_OPENER.open(req, timeout=30) as response:
        content_encoding = response.info().get("Content-Encoding")
        raw_data = _read_bounded(response)
        if content_encoding == "gzip" or raw_data.startswith(b"\x1f\x8b"):
            raw_data = decompress_gzip_bounded(raw_data)
        return json.loads(raw_data.decode("utf-8"))


def normalize_idmc_records(records: list[dict]) -> pd.DataFrame:
    """Normalize raw GIDD disaster API records into the canonical ``idmc_events`` DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=[
                "event_id",
                "iso3",
                "country",
                "hazard_type",
                "event_name",
                "start_date",
                "end_date",
                "displacements",
            ]
        )
    df = pd.DataFrame(records)
    df_clean = pd.DataFrame()
    if "id" in df.columns:
        df_clean["event_id"] = pd.to_numeric(df["id"], errors="coerce")
    elif "event_id" in df.columns:
        df_clean["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    else:
        df_clean["event_id"] = range(1, len(df) + 1)
    df_clean["iso3"] = df["iso3"]
    df_clean["country"] = df["country_name"]
    df_clean["hazard_type"] = df["hazard_type_name"].fillna("Unknown")
    df_clean["event_name"] = df["event_name"].fillna("")
    df_clean["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.tz_localize(None)
    df_clean["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.tz_localize(None)
    df_clean["displacements"] = pd.to_numeric(df["new_displacement"], errors="coerce").fillna(0).astype(int)
    return df_clean.dropna(subset=["start_date", "end_date", "iso3"]).reset_index(drop=True)


def fetch_idmc_floods(client_id: str, max_pages: int = MAX_PAGES) -> pd.DataFrame:
    """Fetch paginated IDMC GIDD flood disaster records (``hazard_type__in=10``)."""
    base_url = f"https://{ALLOWED_IDMC_HOST}/external-api/gidd/disasters/"
    query = urllib.parse.urlencode({"client_id": client_id, "limit": 500, "hazard_type__in": "10"})
    next_url: str | None = f"{base_url}?{query}"
    all_records: list[dict] = []
    page_count = 0
    while next_url and page_count < max_pages:
        page_count += 1
        data = fetch_idmc_page(next_url)
        all_records.extend(data.get("results") or [])
        raw_next = data.get("next")
        next_url = validate_idmc_url(raw_next) if raw_next else None
    return normalize_idmc_records(all_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch IDMC GIDD flood records to Parquet or SQL.")
    parser.add_argument("--output-parquet", required=True, help="Destination Parquet file path.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Maximum API pages to fetch.")
    args = parser.parse_args()

    client_id = os.environ.get("IDMC_GIDD_CLIENT_ID", "").strip()
    if not client_id:
        raise RuntimeError("Set IDMC_GIDD_CLIENT_ID to query the IDMC GIDD API.")
    df = fetch_idmc_floods(client_id, max_pages=args.max_pages)
    out_p = Path(args.output_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    print(f"Saved {len(df):,} IDMC flood events to {out_p}")


if __name__ == "__main__":
    main()
