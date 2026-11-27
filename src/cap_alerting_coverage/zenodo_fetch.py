"""Zenodo archive discovery, SHA-256 verification, safe ZIP extraction, and HTTP download."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import requests

from cap_alerting_coverage.config import DEFAULT_ZENODO_RECORD_ID

ZENODO_API_URL = "https://zenodo.org/api/records/{record_id}"
PLACEHOLDER_ZENODO_RECORD_ID = "24583418"

EXPECTED_ZENODO_SUBDIRS: tuple[str, ...] = (
    "01_event_catalogs",
    "02_admin1_exceedance_matrices",
    "03_summary_results",
    "04_validation_deidentified",
)


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def parse_sha256sums(manifest_path: Path) -> dict[str, str]:
    """Parse a standard ``SHA256SUMS.txt`` file into ``{relative_path: sha256_hex}``."""
    entries: dict[str, str] = {}
    for line_no, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed SHA256SUMS line {line_no}: {raw_line!r}")
        digest, rel_path = parts[0].lower(), parts[1].lstrip("*").strip()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA-256 digest on line {line_no}: {digest!r}")
        entries[rel_path] = digest
    return entries


def _is_within_dir(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a ZIP archive into ``dest_dir`` with strict zip-slip and symlink guards."""
    dest_root = dest_dir.expanduser().resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            # Reject symlinks (UNIX mode 0o120000 in upper 16 bits of external_attr)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"Rejected symlink member inside {zip_path.name}: {name!r}")
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Rejected unsafe zip member path in {zip_path.name}: {name!r}")
            target = (dest_root / rel).resolve()
            if not _is_within_dir(dest_root, target):
                raise ValueError(f"Rejected zip-slip member in {zip_path.name}: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            extracted.append(target)
    return extracted


def verify_zenodo_checksums(zenodo_dir: str | Path, manifest_name: str = "SHA256SUMS.txt") -> dict[str, str]:
    """Verify every file listed in ``SHA256SUMS.txt`` inside ``zenodo_dir``.

    Returns ``{relative_path: status}`` and raises ``FileNotFoundError`` if the manifest is missing
    or ``ValueError`` if any file is missing or fails SHA-256 verification.
    """
    root = Path(zenodo_dir).expanduser().resolve()
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checksum manifest not found: {manifest_path}")

    expected = parse_sha256sums(manifest_path)
    results: dict[str, str] = {}
    failures: list[str] = []
    for rel_path, expected_digest in expected.items():
        fpath = (root / rel_path).resolve()
        if not _is_within_dir(root, fpath):
            failures.append(f"Path traversal rejected: {rel_path}")
            continue
        if not fpath.exists():
            failures.append(f"Missing file: {rel_path}")
            continue
        actual_digest = compute_sha256(fpath)
        if actual_digest != expected_digest:
            failures.append(f"Checksum mismatch on {rel_path}: expected {expected_digest}, got {actual_digest}")
        else:
            results[rel_path] = "ok"

    if failures:
        raise ValueError("Zenodo archive integrity check failed:\n  - " + "\n  - ".join(failures))
    return results


def download_zenodo_archive(
    dest_dir: str | Path,
    record_id: str | None = None,
    verify: bool = True,
    timeout: int = 60,
    extract_zips: bool = True,
) -> Path:
    """Download all files from a published Zenodo record into ``dest_dir``.

    Verifies MD5 digests reported by Zenodo, safely extracts modular ``0?_*.zip`` archives
    (with zip-slip protection), and enforces ``SHA256SUMS.txt`` verification when ``verify=True``.
    """
    env_rec = os.environ.get("CAP_GAP_ZENODO_RECORD_ID", "").strip()
    rec = (record_id or (env_rec if env_rec and env_rec != PLACEHOLDER_ZENODO_RECORD_ID else "")).strip()
    if not rec:
        raise ValueError(
            "An explicit published Zenodo --record-id is required while the reserved archive DOI "
            f"(10.5281/zenodo.{DEFAULT_ZENODO_RECORD_ID}) is pending final publication. "
            "Pass --record-id <PUBLISHED_ID> (or set CAP_GAP_ZENODO_RECORD_ID) once published, "
            "or point --zenodo-dir / CAP_GAP_ZENODO_DIR at an unpacked local archive directory."
        )

    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    api_url = ZENODO_API_URL.format(record_id=rec)
    try:
        resp = requests.get(api_url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to contact Zenodo API for record {rec}: {exc}") from exc

    if resp.status_code == 404:
        raise RuntimeError(
            f"Zenodo record {rec} was not found (HTTP 404). If this record is a reserved placeholder "
            "awaiting publication, pass --zenodo-dir / CAP_GAP_ZENODO_DIR pointing to your local archive."
        )
    resp.raise_for_status()
    payload = resp.json()

    files = payload.get("files") or []
    if not files:
        raise RuntimeError(f"Zenodo record {rec} contains no downloadable files.")

    downloaded_zips: list[Path] = []
    for entry in files:
        key = entry["key"]
        target = (dest / key).resolve()
        if not _is_within_dir(dest, target):
            raise ValueError(f"Rejected unsafe archive member path: {key!r}")
        target.parent.mkdir(parents=True, exist_ok=True)

        dl_url = entry["links"]["self"]
        with requests.get(dl_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with target.open("wb") as out_fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_fh.write(chunk)

        checksum_spec = str(entry.get("checksum") or "")
        if checksum_spec.startswith("md5:"):
            expected_md5 = checksum_spec.split(":", 1)[1].lower()
            h_md5 = hashlib.md5()
            with target.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    h_md5.update(block)
            if h_md5.hexdigest() != expected_md5:
                raise ValueError(f"MD5 mismatch on downloaded file {key}.")

        if target.suffix.lower() == ".zip":
            downloaded_zips.append(target)

    if extract_zips:
        for zpath in downloaded_zips:
            safe_extract_zip(zpath, dest)

    if verify:
        verify_zenodo_checksums(dest)
    return dest
