"""End-to-end synthetic Tier-1 integration tests (scoring, tables, figures, validation, CLI)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from cap_alerting_coverage.cli import main as cli_main
from cap_alerting_coverage.config import ProjectPaths
from cap_alerting_coverage.manuscript_outputs import generate_manuscript_tables_and_figures
from cap_alerting_coverage.schemas import (
    validate_cap_gid1_matrix,
    validate_deidentified_validation_table,
    validate_event_catalog,
    validate_ffm_gid1_matrix,
)
from cap_alerting_coverage.scoring import run_full_scoring
from cap_alerting_coverage.validation_audit import (
    compute_confusion_matrix_table,
    compute_spatial_precision_table,
    counts_from_deidentified_table,
    load_deidentified_validation_table,
    run_validation_audit,
)
from cap_alerting_coverage.zenodo_fetch import (
    download_zenodo_archive,
    safe_extract_zip,
    verify_zenodo_checksums,
)
from tests.fixtures.synthetic_zenodo import build_synthetic_zenodo_archive


def test_synthetic_tier1_pipeline_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zenodo_dir = build_synthetic_zenodo_archive(tmp_path / "zenodo")
    verified = verify_zenodo_checksums(zenodo_dir)
    assert len(verified) >= 12

    # Schema-validate synthetic Zenodo artifacts
    validate_event_catalog(
        pd.read_parquet(zenodo_dir / "01_event_catalogs" / "catalog_without_gs_all_185_2021_2025.parquet")
    )
    validate_event_catalog(
        pd.read_parquet(zenodo_dir / "01_event_catalogs" / "catalog_with_gs_all_185_2021_2025.parquet")
    )
    validate_cap_gid1_matrix(
        pd.read_parquet(zenodo_dir / "02_admin1_exceedance_matrices" / "cap_gid1_daily_2020_2025.parquet")
    )
    validate_ffm_gid1_matrix(
        pd.read_parquet(zenodo_dir / "02_admin1_exceedance_matrices" / "ffm_gid1_daily_2021_2025.parquet")
    )

    val_df = load_deidentified_validation_table(
        zenodo_dir / "04_validation_deidentified" / "validation_n400_deidentified.csv"
    )
    validate_deidentified_validation_table(val_df)
    assert len(val_df) == 400
    cm = compute_confusion_matrix_table(val_df)
    sp = compute_spatial_precision_table(val_df)
    assert len(cm) == 4 * 6 * 7  # 4 label versions x 6 subsets x 7 predictors
    assert len(sp) == 4 * 6
    v4_counts = counts_from_deidentified_table(
        val_df, label_version="v4_reviewed", predictor="filter02_broad_flood_storm"
    )
    assert v4_counts.n_total == 400
    assert v4_counts.n_exclusion_tp == 220
    assert v4_counts.n_inclusion_selected == 235
    assert v4_counts.n_inclusion_tp == 208
    assert v4_counts.n_country_only_tp == 86

    # Point SUP1 override to synthetic 6-jurisdiction subset
    monkeypatch.setenv("CAP_GAP_SUP1_CSV", str(zenodo_dir / "supplementary_data_1_subset.csv"))
    monkeypatch.setenv("CAP_GAP_ZENODO_DIR", str(zenodo_dir))

    results_dir = tmp_path / "out_results"
    figures_dir = tmp_path / "out_figures"
    paths = ProjectPaths.from_env(
        zenodo_dir=zenodo_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    state = run_full_scoring(paths, strict_185=False)
    outputs = generate_manuscript_tables_and_figures(
        paths,
        scoring_state=state,
        strict_185=False,
        render_figures=False,
    )

    # Check key CSV outputs exist and are non-empty
    expected_csvs = [
        "pr_curves_summary.csv",
        "cap_continuous_pr.csv",
        "dhm_ffm_2d_grid.csv",
        "fine_ffm_pr_curves.csv",
        "lead_window_sweep.csv",
        "single_lead_decay.csv",
        "sensitivity_audits.csv",
        "ffm_gs_slices_pr.csv",
        "table1_summary.csv",
        "table2_gauge_vs_catalogs.csv",
        "stream_gauge_nse_summary.csv",
        "stream_gauge_nse_ecdf_quantiles.csv",
        "stream_gauge_precision_iqr.csv",
        "ed_table1_cohort_d_35.csv",
        "ed_table3_sensitivity_matrix.csv",
        "ed_table4_validation_summary.csv",
        "validation_confusion_counts.csv",
        "validation_spatial_precision.csv",
        "ed_table5_full_benchmark.csv",
        "ed_table6a_annual_onboarding.csv",
        "ed_table6b_late_onboarders.csv",
        "country_coverage_and_gains_150.csv",
    ]
    for fname in expected_csvs:
        p = results_dir / fname
        assert p.exists(), f"Missing output CSV: {fname}"
        assert p.stat().st_size > 0, f"Empty output CSV: {fname}"

    assert outputs["table1"] is not None
    # Verify strict Table 1 & Cohort D consistency on synthetic fixture
    t1 = pd.read_csv(results_dir / "table1_summary.csv")
    assert {"2021_2025", "2024_2025"} <= set(t1["benchmark"])
    assert len(pd.read_csv(results_dir / "ed_table1_cohort_d_35.csv")) == 2  # JPN, BGD

    # Operational GeoGLOWS v2 (RAPID routing) uses only natively published return periods (2..100 yr)
    pr_df = pd.read_csv(results_dir / "pr_curves_summary.csv")
    geo_oper_thrs = set(pr_df[pr_df["system"].str.contains("GeoGLOWS_Oper", na=False)]["thr_label"])
    assert geo_oper_thrs == {"RP=2", "RP=5", "RP=10", "RP=25", "RP=50", "RP=100"}

    # Test CLI roundtrip: reproduce-tier1 -> verify-parity -> build-figures -> audit-validation
    results_dir_2 = tmp_path / "out_results_cli"
    figures_dir_2 = tmp_path / "out_figures_cli"
    rc = cli_main(
        [
            "reproduce-tier1",
            "--zenodo-dir",
            str(zenodo_dir),
            "--results-dir",
            str(results_dir_2),
            "--figures-dir",
            str(figures_dir_2),
            "--verify-checksums",
            "--allow-subset",
            "--no-figures",
        ]
    )
    assert rc == 0

    rc_par = cli_main(
        [
            "verify-parity",
            "--actual-dir",
            str(results_dir_2),
            "--reference-dir",
            str(results_dir),
            "--atol",
            "1e-6",
        ]
    )
    assert rc_par == 0

    # Verify build-figures also works standalone from results_dir without --zenodo-dir
    monkeypatch.delenv("CAP_GAP_ZENODO_DIR", raising=False)
    rc_fig = cli_main(
        [
            "build-figures",
            "--results-dir",
            str(results_dir_2),
            "--figures-dir",
            str(figures_dir_2),
        ]
    )
    assert rc_fig == 0
    expected_stems = (
        "fig-1",
        "fig-2",
        "fig-3",
        "fig-4",
        "fig-5",
        "ed-fig-1",
        "ed-fig-2",
        "ed-fig-3",
        "ed-fig-4",
        "ed-fig-5",
        "supp-fig-1",
    )
    for stem in expected_stems:
        for ext in ("pdf", "svg", "png"):
            fig_p = figures_dir_2 / f"{stem}.{ext}"
            assert fig_p.exists(), f"Missing rendered figure: {fig_p.name}"
            assert fig_p.stat().st_size > 2_000, f"Unexpectedly small figure: {fig_p.name}"

    # Verify verify-parity fails when a reference table is missing unless --allow-missing is passed
    partial_ref = tmp_path / "partial_ref"
    partial_ref.mkdir()
    (partial_ref / "table1_summary.csv").write_text(
        (results_dir / "table1_summary.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert cli_main(["verify-parity", "--actual-dir", str(results_dir_2), "--reference-dir", str(partial_ref)]) == 1
    assert (
        cli_main(
            [
                "verify-parity",
                "--actual-dir",
                str(results_dir_2),
                "--reference-dir",
                str(partial_ref),
                "--allow-missing",
            ]
        )
        == 0
    )

    # Verify --verify-checksums fails loudly when SHA256SUMS.txt is missing
    (zenodo_dir / "SHA256SUMS.txt").unlink()
    assert (
        cli_main(
            [
                "reproduce-tier1",
                "--zenodo-dir",
                str(zenodo_dir),
                "--results-dir",
                str(tmp_path / "out_fail"),
                "--verify-checksums",
                "--allow-subset",
                "--no-figures",
            ]
        )
        == 1
    )

    val_out = tmp_path / "val_cli" / "ed_table4.csv"
    rc_val = cli_main(
        [
            "audit-validation",
            "--input-csv",
            str(zenodo_dir / "04_validation_deidentified" / "validation_n400_deidentified.csv"),
            "--output-csv",
            str(val_out),
        ]
    )
    assert rc_val == 0
    assert val_out.exists()
    assert (val_out.parent / "validation_confusion_counts.csv").exists()


def test_validation_privacy_guard_rejects_raw_text_and_urls():
    bad_df = pd.DataFrame([{"sample_id": 1, "headline": "Flood warning in town"}])
    with pytest.raises(ValueError, match="Privacy/governance violation"):
        validate_deidentified_validation_table(bad_df)

    bad_url = pd.DataFrame(
        [{"sample_id": 1, "is_true_flood_alert": "https://example.com/alert.xml", "matches_inclusion_filter": True}]
    )
    with pytest.raises(ValueError, match="URL"):
        validate_deidentified_validation_table(bad_url)

    bad_long = pd.DataFrame([{"sample_id": 1, "is_true_flood_alert": "x" * 100, "matches_inclusion_filter": True}])
    with pytest.raises(ValueError, match="64-char"):
        validate_deidentified_validation_table(bad_long)


def test_validation_audit_requires_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CAP_GAP_ZENODO_DIR", raising=False)
    with pytest.raises(FileNotFoundError):
        run_validation_audit(deidentified_csv=None, output_csv=tmp_path / "ed4.csv")
    with pytest.raises(FileNotFoundError):
        run_validation_audit(deidentified_csv=tmp_path / "does_not_exist.csv")
    assert cli_main(["audit-validation"]) == 1
    assert cli_main(["audit-validation", "--input-csv", str(tmp_path / "missing.csv")]) == 1


def test_zenodo_fetch_zip_extraction_zip_slip_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1. Placeholder / omitted record_id raises friendly ValueError (CLI exits 1)
    monkeypatch.delenv("CAP_GAP_ZENODO_RECORD_ID", raising=False)
    with pytest.raises(ValueError, match="explicit published Zenodo --record-id"):
        download_zenodo_archive(tmp_path / "dl_none", record_id=None)
    assert cli_main(["fetch-zenodo", "--dest-dir", str(tmp_path / "dl_cli")]) == 1

    # 2. Zip-slip and symlink rejection in safe_extract_zip
    bad_zip = tmp_path / "evil_slip.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError, match="unsafe zip member"):
        safe_extract_zip(bad_zip, tmp_path / "extract_bad")

    sym_zip = tmp_path / "evil_symlink.zip"
    with zipfile.ZipFile(sym_zip, "w") as zf:
        zi = zipfile.ZipInfo("01_event_catalogs/link.txt")
        zi.external_attr = 0o120777 << 16
        zf.writestr(zi, "/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        safe_extract_zip(sym_zip, tmp_path / "extract_sym")

    # 3. Mocked Zenodo record with modular ZIP + SHA256SUMS.txt succeeds & verifies
    inner_payload = b"hello,zenodo\n1,2\n"
    inner_sha256 = hashlib.sha256(inner_payload).hexdigest()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("03_summary_results/sample.csv", inner_payload)
    zip_bytes = zip_buf.getvalue()
    manifest_bytes = f"{inner_sha256}  03_summary_results/sample.csv\n".encode("utf-8")

    class _FakeResp:
        def __init__(self, status_code: int = 200, json_data: dict | None = None, content: bytes = b""):
            self.status_code = status_code
            self._json = json_data
            self._content = content

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict:
            assert self._json is not None
            return self._json

        def iter_content(self, chunk_size: int = 1024):
            yield self._content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_get(url: str, **kwargs):
        if url.endswith("/99999999"):
            return _FakeResp(status_code=404)
        if url.endswith("/12345678"):
            return _FakeResp(
                status_code=200,
                json_data={
                    "files": [
                        {
                            "key": "03_summary_results.zip",
                            "checksum": f"md5:{hashlib.md5(zip_bytes).hexdigest()}",
                            "links": {"self": "https://zenodo.example/dl/03_summary_results.zip"},
                        },
                        {
                            "key": "SHA256SUMS.txt",
                            "checksum": f"md5:{hashlib.md5(manifest_bytes).hexdigest()}",
                            "links": {"self": "https://zenodo.example/dl/SHA256SUMS.txt"},
                        },
                    ]
                },
            )
        if url.endswith("03_summary_results.zip"):
            return _FakeResp(content=zip_bytes)
        if url.endswith("SHA256SUMS.txt"):
            return _FakeResp(content=manifest_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("cap_alerting_coverage.zenodo_fetch.requests.get", _fake_get)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        download_zenodo_archive(tmp_path / "dl_404", record_id="99999999")

    dl_ok = download_zenodo_archive(tmp_path / "dl_ok", record_id="12345678", verify=True)
    assert (dl_ok / "03_summary_results" / "sample.csv").read_bytes() == inner_payload
