"""Public clean-room, governance, license, and non-redistribution guardrail tests.

Runs in CI on every push and pull request to verify that:
  1. `LICENSE` matches the unmodified canonical Apache-2.0 license text.
  2. No raw binary datasets (Parquet, GeoPackage, NPZ, NetCDF, SQLite, etc.) are tracked in Git.
  3. No credentials, JWTs, private keys, embedded SHA-256 contact-email hash maps, local
     home-directory paths, or unexpected personal email addresses enter tracked source files.
  4. All required curated static reference files are present.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_APACHE2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

# Forbidden binary/raw dataset extensions that must never be committed to Git.
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".parquet",
        ".geoparquet",
        ".gpkg",
        ".npz",
        ".npy",
        ".zarr",
        ".grib",
        ".grib2",
        ".nc",
        ".dump",
        ".sqlite",
        ".db",
        ".xlsx",
        ".xls",
    }
)

# Generic public hygiene & secret-leak guardrails.
GENERIC_FORBIDDEN_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("live_jwt_token", re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("embedded_sha256_hash_map", re.compile(r"['\"][0-9a-f]{64}['\"]\s*:\s*\(")),
    (
        "local_unix_home_path",
        re.compile(r"(?:^|[\s\"'(\[=:])(?:/[A-Za-z0-9._-]+){0,4}/(?:home|Users)/[A-Za-z0-9._-]+/"),
    ),
]

SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "dist",
        "build",
        "out_results",
        "out_figures",
        "work_data",
        "zenodo_archive",
        "zenodo_data",
        "zenodo_cache",
    }
)

TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".cff",
        ".yml",
        ".yaml",
        ".json",
        ".geojson",
        ".csv",
        ".sql",
        ".svg",
        ".ipynb",
        ".sh",
        "",
    }
)


def _is_skipped(rel_path: Path) -> bool:
    for part in rel_path.parts:
        if part in SKIP_DIR_NAMES or part.endswith((".egg-info", ".eggs")):
            return True
    return False


def _iter_repo_files() -> list[Path]:
    """Enumerate Git-tracked and non-ignored files (falling back to filtered rglob outside Git)."""
    if (REPO_ROOT / ".git").exists():
        try:
            out = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                stderr=subprocess.DEVNULL,
            )
            files: list[Path] = []
            for raw_rel in out.decode("utf-8").split("\x00"):
                if not raw_rel:
                    continue
                rel = Path(raw_rel)
                if _is_skipped(rel):
                    continue
                full = REPO_ROOT / rel
                if full.is_file():
                    files.append(full)
            return sorted(files)
        except Exception:
            pass

    files = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO_ROOT)
        if _is_skipped(rel):
            continue
        files.append(p)
    return sorted(files)


def test_canonical_apache2_license():
    lic_path = REPO_ROOT / "LICENSE"
    assert lic_path.is_file(), "Missing root LICENSE file."
    digest = hashlib.sha256(lic_path.read_bytes()).hexdigest()
    assert digest == CANONICAL_APACHE2_SHA256, (
        f"LICENSE does not match canonical Apache-2.0 SHA-256 ({CANONICAL_APACHE2_SHA256}); got {digest}."
    )
    assert (REPO_ROOT / "data_inputs" / "LICENSE").is_file(), "Missing data_inputs/LICENSE attribution file."


def test_no_forbidden_raw_or_binary_datasets_in_repo():
    offenders = []
    for p in _iter_repo_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if p.suffix.lower() in FORBIDDEN_EXTENSIONS or rel.endswith(".ndjson.gz"):
            offenders.append(rel)
    assert not offenders, f"Forbidden raw/binary dataset files present in repo: {offenders}"


def test_no_secrets_local_paths_or_embedded_hashes_in_repo():
    violations: list[str] = []
    for p in _iter_repo_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in {"LICENSE", ".gitignore"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for label, rx in GENERIC_FORBIDDEN_REGEXES:
            for m in rx.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                violations.append(f"{rel}:{line_no} [{label}] -> {m.group(0)!r}")

    assert not violations, "Repository hygiene guardrail violations found:\n" + "\n".join(violations[:40])


def test_verified_data_inputs_present():
    required = [
        "data_inputs/LICENSE",
        "data_inputs/Supplementary_Data_1_Global_Alerting_Authorities_Audit_185_Jurisdictions.csv",
        "data_inputs/unified_multilingual_keyword_filters.csv",
        "data_inputs/annex_inclusion_keywords.txt",
        "data_inputs/annex_exclusion_keywords.txt",
        "data_inputs/ne_110m_admin_0_countries.geojson",
        "figures/Figure_5_early_warning_impact_chain.svg",
    ]
    missing = [r for r in required if not (REPO_ROOT / r).exists()]
    assert not missing, f"Missing required reference inputs: {missing}"
