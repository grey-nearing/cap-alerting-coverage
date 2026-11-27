"""Integration tests against a real staged Zenodo archive (``CAP_GAP_ZENODO_DIR``).

Skipped automatically in offline CI when ``CAP_GAP_ZENODO_DIR`` is unset.
Run explicitly via::

    CAP_GAP_ZENODO_DIR=/path/to/zenodo_archive pytest -m zenodo -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cap_alerting_coverage.cli import main as cli_main
from cap_alerting_coverage.schemas import (
    validate_cap_gid0_matrix,
    validate_cap_gid1_matrix,
    validate_deidentified_validation_table,
    validate_event_catalog,
    validate_ffm_gid1_matrix,
    validate_public_summary_table,
    validate_tier_cube,
)
from cap_alerting_coverage.validation_audit import load_deidentified_validation_table
from cap_alerting_coverage.zenodo_fetch import verify_zenodo_checksums

ZENODO_ENV = os.environ.get("CAP_GAP_ZENODO_DIR", "").strip()
pytestmark = [
    pytest.mark.zenodo,
    pytest.mark.skipif(not ZENODO_ENV, reason="CAP_GAP_ZENODO_DIR is not set"),
]


def _zenodo_root() -> Path:
    root = Path(ZENODO_ENV).expanduser().resolve()
    assert root.is_dir(), f"CAP_GAP_ZENODO_DIR does not exist: {root}"
    return root


def test_zenodo_checksums_and_schemas():
    root = _zenodo_root()
    if (root / "SHA256SUMS.txt").exists():
        verified = verify_zenodo_checksums(root)
        assert len(verified) == 54

    # Validate event catalogs (strictly non-redistribution & schema compliant)
    for fname in (
        "catalog_without_gs_all_185_2021_2025.parquet",
        "catalog_with_gs_all_185_2021_2025.parquet",
    ):
        p = root / "01_event_catalogs" / fname
        if p.exists():
            validate_event_catalog(pd.read_parquet(p), dataset_name=fname)

    # Validate CAP & FFM derived matrices
    mat_dir = root / "02_admin1_exceedance_matrices"
    if (mat_dir / "cap_gid1_daily_2020_2025.parquet").exists():
        validate_cap_gid1_matrix(pd.read_parquet(mat_dir / "cap_gid1_daily_2020_2025.parquet"))
    if (mat_dir / "cap_gid0_daily_2020_2025.parquet").exists():
        validate_cap_gid0_matrix(pd.read_parquet(mat_dir / "cap_gid0_daily_2020_2025.parquet"))
    if (mat_dir / "ffm_gid1_daily_2021_2025.parquet").exists():
        validate_ffm_gid1_matrix(pd.read_parquet(mat_dir / "ffm_gid1_daily_2021_2025.parquet"))

    # Validate riverine exceedance tier cubes
    dhm_p = mat_dir / "dhm_10d_gid1_multilead_2021_2025.npz"
    if dhm_p.exists():
        dhm = dict(np.load(dhm_p, allow_pickle=True))
        gid1 = dhm["gid_1"]
        validate_tier_cube(dhm, expected_ndim=3, gid1_list=gid1, dataset_name="DHM_10D")
        for npz_name, ndim, exp_days in (
            ("geoglows_v2_retro_gid1_daily_2021_2025.npz", 2, 1843),
            ("geoglows_v2_oper_gid1_multilead_2024_2025.npz", 4, None),
            ("glofas_v5_gid1_daily_2021_2025.npz", 2, 1843),
        ):
            p = mat_dir / npz_name
            if p.exists():
                validate_tier_cube(
                    dict(np.load(p, allow_pickle=True)),
                    expected_ndim=ndim,
                    gid1_list=gid1,
                    dataset_name=npz_name,
                    expected_days=exp_days,
                )

    # Validate all public summary/audit CSVs in 03_summary_results/ (blocks raw Montandon descriptions, etc.)
    res_dir = root / "03_summary_results"
    if res_dir.is_dir():
        for csv_p in sorted(res_dir.glob("*.csv")):
            validate_public_summary_table(pd.read_csv(csv_p, nrows=5), dataset_name=csv_p.name)

    # Validate de-identified N=400 audit table
    val_p = root / "04_validation_deidentified" / "validation_n400_deidentified.csv"
    if val_p.exists():
        validate_deidentified_validation_table(load_deidentified_validation_table(val_p), dataset_name=val_p.name)


def test_zenodo_full_tier1_reproduction_and_parity(tmp_path: Path):
    root = _zenodo_root()
    out_res = tmp_path / "results"
    out_fig = tmp_path / "figures"

    rc = cli_main(
        [
            "reproduce-tier1",
            "--zenodo-dir",
            str(root),
            "--results-dir",
            str(out_res),
            "--figures-dir",
            str(out_fig),
            "--no-figures",
        ]
    )
    assert rc == 0

    if (root / "03_summary_results").is_dir():
        rc_par = cli_main(
            [
                "verify-parity",
                "--actual-dir",
                str(out_res),
                "--reference-dir",
                str(root),
                "--atol",
                "1e-3",
            ]
        )
        assert rc_par == 0

    # Verify country-level paired Wilcoxon, DEFF, and clustered 95% CIs (Methods Section 6)
    cl_df = pd.read_csv(out_res / "country_clustered_inference.csv")
    hyb_a = cl_df[(cl_df["catalog"] == "Without_GS") & (cl_df["system"] == "CAP_Perm_U_DHM_Oper_U_FFM050")].iloc[0]
    hyb_b = cl_df[(cl_df["catalog"] == "With_GS") & (cl_df["system"] == "CAP_Perm_U_DHM_Oper_U_FFM050")].iloc[0]
    assert int(hyb_a["n_countries"]) == 139 and int(hyb_a["n_improved"]) == 126
    assert abs(float(hyb_a["median_pp_gain"]) - 50.0) < 0.05
    assert abs(float(hyb_a["wilcoxon_p"]) - 1.994e-22) < 1e-23
    assert abs(float(hyb_a["baseline_deff"]) - 116.1) < 0.15
    assert abs(float(hyb_a["baseline_ci_low_pct"]) - 24.82) < 0.05
    assert abs(float(hyb_a["baseline_ci_high_pct"]) - 51.05) < 0.05
    assert abs(float(hyb_a["candidate_ci_low_pct"]) - 70.95) < 0.05
    assert abs(float(hyb_a["candidate_ci_high_pct"]) - 85.23) < 0.05
    assert abs(float(hyb_a["gap_closed_ci_low_pct"]) - 58.35) < 0.05
    assert abs(float(hyb_a["gap_closed_ci_high_pct"]) - 72.33) < 0.05

    assert int(hyb_b["n_countries"]) == 149 and int(hyb_b["n_improved"]) == 134
    assert abs(float(hyb_b["median_pp_gain"]) - 48.0) < 0.05
    assert abs(float(hyb_b["wilcoxon_p"]) - 9.785e-24) < 1e-24
    assert abs(float(hyb_b["baseline_deff"]) - 288.4) < 0.15
