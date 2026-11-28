"""Central configuration, canonical constants, and path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Canonical daily evaluation calendar (2020-12-15 .. 2025-12-31, 1,843 days)
# Starts 17 days before 2021-01-01 to accommodate up to 15-day pre-onset lead windows.
CALENDAR_START = "2020-12-15"
CALENDAR_END = "2025-12-31"
DATES = pd.date_range(CALENDAR_START, CALENDAR_END, freq="D")
D_TOTAL = len(DATES)  # 1843
DATE_TO_IDX = {str(d.date()): i for i, d in enumerate(DATES)}

# Canonical evaluation benchmarks
BENCHMARKS: tuple[tuple[str, str, str], ...] = (
    ("2021_2025", "2021-01-01", "2025-12-31"),
    ("2024_2025", "2024-04-16", "2025-12-31"),
)

# Threshold grids matching the manuscript benchmark
RETURN_PERIODS: tuple[float, ...] = (1.01, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)
# Natively published GeoGLOWS v2 RAPID-era return periods (tier index, return period yr);
# omits extrapolated 1.01-yr and interpolated 20-yr levels for operational GeoGLOWS v2 forecasts.
GEOGLOWS_OPER_RP_TIERS: tuple[tuple[int, float], ...] = (
    (2, 2.0),
    (3, 5.0),
    (4, 10.0),
    (6, 25.0),
    (7, 50.0),
    (8, 100.0),
)
FFM_THRESHOLDS: tuple[float, ...] = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05)
# Maximum pre-onset matching lead for Google FFM (0-24h forward forecast horizon; outlook = 86,400 s).
FFM_MAX_PRE_DAYS: int = 1
CAP_SEVERITIES: tuple[tuple[int, str], ...] = (
    (4, "Extreme"),
    (3, ">=Severe"),
    (2, ">=Moderate"),
    (1, "All"),
)
CAP_CONTINUOUS_THRESHOLDS: tuple[int, ...] = (64, 61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 3, 2, 1)
LEAD_WINDOWS_DAYS: tuple[int, ...] = (0, 1, 3, 5, 7, 10, 15)

# Legacy 19-country non-CAP exclusion sensitivity set (166-country cohort boundary check)
LEGACY_EXCLUDED_19: frozenset[str] = frozenset(
    {
        "JPN",
        "PER",
        "MYS",
        "NPL",
        "VNM",
        "BGD",
        "PAK",
        "THA",
        "BOL",
        "GTM",
        "URY",
        "RWA",
        "ZAF",
        "LKA",
        "DOM",
        "CHL",
        "UGA",
        "AGO",
        "COL",
    }
)

# ISO-3166-1 alpha-3 normalization applied consistently across GADM, CAP, IDMC, and Montandon
ISO_UNIFY: dict[str, str] = {
    "XXK": "XKX",
    "XKO": "XKX",
    "KOS": "XKX",
    "AZO": "PRT",
    "PRI": "USA",
}

DEFAULT_ZENODO_RECORD_ID = os.environ.get("CAP_GAP_ZENODO_RECORD_ID", "24583418")
DEFAULT_ZENODO_DOI = f"10.5281/zenodo.{DEFAULT_ZENODO_RECORD_ID}"


def date_to_idx(dt_val: object) -> int:
    """Map a date-like value to its integer offset from CALENDAR_START (2020-12-15)."""
    if isinstance(dt_val, bytes):
        dt_val = dt_val.decode("utf-8")
    elif hasattr(dt_val, "item") and not isinstance(dt_val, pd.Timestamp):
        dt_val = dt_val.item()
    return int((pd.Timestamp(dt_val).normalize() - DATES[0]).days)


def default_repo_root() -> Path:
    """Resolve repository root cleanly from either editable install or installed package."""
    env_root = os.environ.get("CAP_GAP_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # src/cap_alerting_coverage/config.py -> parents[2] is repo root in editable checkouts
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data_inputs").is_dir():
        return candidate
    return Path.cwd().resolve()


def default_data_inputs_dir(repo_root: Path) -> Path:
    """Locate curated reference data_inputs/ across repo root, wheel package data, or CWD."""
    for cand in (
        repo_root / "data_inputs",
        Path(__file__).resolve().parent / "_data_inputs",
        Path.cwd().resolve() / "data_inputs",
    ):
        if (cand / "Supplementary_Data_1_Global_Alerting_Authorities_Audit_185_Jurisdictions.csv").is_file():
            return cand
    return repo_root / "data_inputs"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for inputs, Zenodo staged archive, working caches, results, and figures."""

    repo_root: Path
    data_inputs_dir: Path
    zenodo_dir: Path | None
    work_dir: Path
    results_dir: Path
    figures_dir: Path

    @classmethod
    def from_env(
        cls,
        repo_root: str | Path | None = None,
        zenodo_dir: str | Path | None = None,
        work_dir: str | Path | None = None,
        results_dir: str | Path | None = None,
        figures_dir: str | Path | None = None,
        data_inputs_dir: str | Path | None = None,
    ) -> "ProjectPaths":
        root = Path(repo_root).expanduser().resolve() if repo_root else default_repo_root()
        z_raw = zenodo_dir or os.environ.get("CAP_GAP_ZENODO_DIR")
        z_path = Path(z_raw).expanduser().resolve() if z_raw else None
        d_raw = data_inputs_dir or os.environ.get("CAP_GAP_DATA_INPUTS_DIR")
        d_in = Path(d_raw).expanduser().resolve() if d_raw else default_data_inputs_dir(root)
        w_dir = Path(work_dir or os.environ.get("CAP_GAP_WORK_DIR") or (root / "work_data")).expanduser().resolve()
        r_dir = (
            Path(results_dir or os.environ.get("CAP_GAP_RESULTS_DIR") or (root / "out_results")).expanduser().resolve()
        )
        f_dir = (
            Path(figures_dir or os.environ.get("CAP_GAP_FIGURES_DIR") or (root / "out_figures")).expanduser().resolve()
        )
        return cls(
            repo_root=root,
            data_inputs_dir=d_in,
            zenodo_dir=z_path,
            work_dir=w_dir,
            results_dir=r_dir,
            figures_dir=f_dir,
        )

    @property
    def supplementary_data_1_csv(self) -> Path:
        override = os.environ.get("CAP_GAP_SUP1_CSV")
        if override:
            return Path(override).expanduser().resolve()
        return self.data_inputs_dir / "Supplementary_Data_1_Global_Alerting_Authorities_Audit_185_Jurisdictions.csv"

    @property
    def countries_geojson(self) -> Path:
        override = os.environ.get("CAP_GAP_GEOJSON_PATH")
        if override:
            return Path(override).expanduser().resolve()
        return self.data_inputs_dir / "ne_110m_admin_0_countries.geojson"

    def resolve_data_file(self, filename: str, *sub_candidates: str) -> Path:
        """Locate a Tier-1/Tier-2 input file across Zenodo subdirectories or work_dir."""
        search_roots: list[Path] = []
        if self.zenodo_dir is not None:
            for sub in sub_candidates:
                search_roots.append(self.zenodo_dir / sub)
            search_roots.append(self.zenodo_dir)
        search_roots.append(self.work_dir)
        search_roots.append(self.results_dir)
        search_roots.append(self.data_inputs_dir)
        for root in search_roots:
            cand = root / filename
            if cand.exists():
                return cand
        checked = ", ".join(str(r / filename) for r in search_roots)
        raise FileNotFoundError(f"Required dataset '{filename}' not found. Checked: {checked}")
