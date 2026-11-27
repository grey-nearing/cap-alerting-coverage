"""Command-line interface for Tier-1 reproduction, parity verification, and figure building."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cap_alerting_coverage import __version__
from cap_alerting_coverage.config import ProjectPaths
from cap_alerting_coverage.manuscript_outputs import (
    generate_manuscript_tables_and_figures,
    render_figures_from_results,
)
from cap_alerting_coverage.scoring import run_full_scoring
from cap_alerting_coverage.validation_audit import run_validation_audit
from cap_alerting_coverage.zenodo_fetch import download_zenodo_archive, verify_zenodo_checksums

PARITY_TABLES: tuple[str, ...] = (
    "table1_summary.csv",
    "table2_gauge_vs_catalogs.csv",
    "stream_gauge_nse_summary.csv",
    "stream_gauge_nse_ecdf_quantiles.csv",
    "stream_gauge_precision_iqr.csv",
    "ed_table1_cohort_d_35.csv",
    "ed_table3_sensitivity_matrix.csv",
    "ed_table5_full_benchmark.csv",
    "ed_table6a_annual_onboarding.csv",
    "ed_table6b_late_onboarders.csv",
    "country_coverage_and_gains_150.csv",
    "pr_curves_summary.csv",
    "cap_continuous_pr.csv",
    "dhm_ffm_2d_grid.csv",
    "fine_ffm_pr_curves.csv",
    "lead_window_sweep.csv",
    "single_lead_decay.csv",
    "sensitivity_audits.csv",
    "ffm_gs_slices_pr.csv",
)

VALIDATION_PARITY_TABLES: tuple[str, ...] = (
    "validation_confusion_counts.csv",
    "validation_spatial_precision.csv",
)

PASS_THROUGH_TAGS: dict[str, str] = {
    "stream_gauge_nse_summary.csv": " [PASS-THROUGH]",
    "stream_gauge_nse_ecdf_quantiles.csv": " [PASS-THROUGH]",
    "stream_gauge_precision_iqr.csv": " [PASS-THROUGH]",
    "table2_gauge_vs_catalogs.csv": " [PARTIAL PASS-THROUGH]",
}


def compare_csv_tables(
    actual_path: Path,
    reference_path: Path,
    atol: float = 1e-3,
) -> tuple[bool, str]:
    """Compare two CSV tables cell-by-cell with numeric tolerance ``atol``."""
    df_a = pd.read_csv(actual_path)
    df_r = pd.read_csv(reference_path)
    if list(df_a.columns) != list(df_r.columns):
        return False, f"column mismatch: actual={list(df_a.columns)} vs ref={list(df_r.columns)}"
    if len(df_a) != len(df_r):
        return False, f"row count mismatch: actual={len(df_a)} vs ref={len(df_r)}"

    max_diff = 0.0
    for col in df_a.columns:
        sa = df_a[col]
        sr = df_r[col]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sr):
            va = sa.to_numpy(dtype=np.float64)
            vr = sr.to_numpy(dtype=np.float64)
            nan_a, nan_r = np.isnan(va), np.isnan(vr)
            if not np.array_equal(nan_a, nan_r):
                return False, f"NaN mask mismatch in column '{col}'"
            finite = ~nan_a
            if np.any(finite):
                diff = float(np.max(np.abs(va[finite] - vr[finite])))
                max_diff = max(max_diff, diff)
                if diff > atol:
                    return False, f"numeric diff {diff:.6g} > atol={atol} in column '{col}'"
        else:
            fa = sa.fillna("").astype(str).to_numpy()
            fr = sr.fillna("").astype(str).to_numpy()
            if not np.array_equal(fa, fr):
                return False, f"string value mismatch in column '{col}'"
    return True, f"PASS ({len(df_a)} rows, max |diff|={max_diff:.2e})"


def cmd_reproduce_tier1(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_env(
        zenodo_dir=args.zenodo_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        data_inputs_dir=args.data_inputs_dir,
    )
    if args.verify_checksums:
        if paths.zenodo_dir is None:
            print(
                "[FAIL] --verify-checksums requires --zenodo-dir or CAP_GAP_ZENODO_DIR.",
                file=sys.stderr,
            )
            return 1
        try:
            verified = verify_zenodo_checksums(paths.zenodo_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[FAIL] Checksum verification failed: {exc}", file=sys.stderr)
            return 1
        print(f"Verified {len(verified)} files in SHA256SUMS.txt.")

    state = run_full_scoring(paths, strict_185=not args.allow_subset)
    generate_manuscript_tables_and_figures(
        paths,
        scoring_state=state,
        strict_185=not args.allow_subset,
        render_figures=not args.no_figures,
    )
    print(f"Tier-1 reproduction complete. Tables -> {paths.results_dir} | Figures -> {paths.figures_dir}")
    return 0


def cmd_verify_parity(args: argparse.Namespace) -> int:
    actual_dir = Path(args.actual_dir).expanduser().resolve()
    ref_input = Path(args.reference_dir).expanduser().resolve()
    ref_dir = ref_input / "03_summary_results" if (ref_input / "03_summary_results").is_dir() else ref_input

    all_ok = True
    checked = 0
    for fname in PARITY_TABLES:
        ref_p = ref_dir / fname
        act_p = actual_dir / fname
        tag = PASS_THROUGH_TAGS.get(fname, "")
        if not ref_p.exists():
            if not args.allow_missing:
                print(f"[FAIL] {fname}: missing in reference directory {ref_dir}")
                all_ok = False
            continue
        checked += 1
        if not act_p.exists():
            print(f"[FAIL] {fname}: missing in {actual_dir}")
            all_ok = False
            continue
        ok, msg = compare_csv_tables(act_p, ref_p, atol=args.atol)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {fname}{tag}: {msg}")
        if not ok:
            all_ok = False

    # Also verify de-identified validation confusion/spatial tables when 04_validation_deidentified/ is present
    val_ref_dir = ref_input / "04_validation_deidentified"
    if not val_ref_dir.is_dir() and ref_dir.name == "03_summary_results":
        val_ref_dir = ref_dir.parent / "04_validation_deidentified"
    if val_ref_dir.is_dir():
        for fname in VALIDATION_PARITY_TABLES:
            ref_p = val_ref_dir / fname
            act_p = actual_dir / fname
            if not ref_p.exists():
                continue
            checked += 1
            if not act_p.exists():
                print(f"[FAIL] {fname}: missing in {actual_dir}")
                all_ok = False
                continue
            ok, msg = compare_csv_tables(act_p, ref_p, atol=args.atol)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {fname} [VALIDATION]: {msg}")
            if not ok:
                all_ok = False

    if checked == 0:
        print(f"[FAIL] No reference parity tables found in {ref_dir}", file=sys.stderr)
        return 1
    return 0 if all_ok else 1


def cmd_build_figures(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_env(
        zenodo_dir=args.zenodo_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        data_inputs_dir=args.data_inputs_dir,
    )
    try:
        written = render_figures_from_results(paths)
    except FileNotFoundError as exc:
        print(
            f"[FAIL] {exc} Pass --zenodo-dir <path> when --results-dir does not contain "
            "all auxiliary inputs (such as cap_country_hazard_counts_2021_2025.csv or "
            "ed_table4_validation_summary.csv).",
            file=sys.stderr,
        )
        return 1
    print(f"Rendered {len(written)} figures into {paths.figures_dir}")
    return 0


def cmd_verify_zenodo(args: argparse.Namespace) -> int:
    try:
        verified = verify_zenodo_checksums(args.zenodo_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"All {len(verified)} files in {args.zenodo_dir} passed SHA-256 verification.")
    return 0


def cmd_fetch_zenodo(args: argparse.Namespace) -> int:
    try:
        dest = download_zenodo_archive(
            dest_dir=args.dest_dir,
            record_id=args.record_id,
            verify=not args.skip_verify,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"Downloaded Zenodo archive into {dest}")
    return 0


def cmd_audit_validation(args: argparse.Namespace) -> int:
    in_csv = args.input_csv
    if not in_csv:
        z_raw = args.zenodo_dir or os.environ.get("CAP_GAP_ZENODO_DIR")
        if z_raw:
            cand = (
                Path(z_raw).expanduser().resolve() / "04_validation_deidentified" / "validation_n400_deidentified.csv"
            )
            if cand.exists():
                in_csv = str(cand)
    if not in_csv:
        print(
            "[FAIL] Missing input CSV. Pass --input-csv <path/to/validation_n400_deidentified.csv> "
            "or --zenodo-dir <path> (or set CAP_GAP_ZENODO_DIR).",
            file=sys.stderr,
        )
        return 1
    try:
        df = run_validation_audit(
            deidentified_csv=in_csv,
            output_csv=args.output_csv,
            label_version=args.label_version,
            predictor=args.predictor,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(df.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible CAP flood alerting coverage & global flood model benchmark CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rep = sub.add_parser("reproduce-tier1", help="Run full Tier-1 scoring, tables, and figures from Zenodo archive.")
    p_rep.add_argument("--zenodo-dir", default=None, help="Path to local Zenodo archive (or set CAP_GAP_ZENODO_DIR).")
    p_rep.add_argument("--results-dir", default=None, help="Output directory for generated summary CSV tables.")
    p_rep.add_argument("--figures-dir", default=None, help="Output directory for generated 300-DPI PNG figures.")
    p_rep.add_argument("--data-inputs-dir", default=None, help="Override path to curated data_inputs/ reference files.")
    p_rep.add_argument("--no-figures", action="store_true", help="Skip matplotlib figure rendering.")
    p_rep.add_argument(
        "--verify-checksums", action="store_true", help="Require and verify SHA256SUMS.txt before scoring."
    )
    p_rep.add_argument(
        "--allow-subset", action="store_true", help="Allow subset jurisdiction tables (e.g., synthetic fixtures)."
    )
    p_rep.set_defaults(func=cmd_reproduce_tier1)

    p_par = sub.add_parser("verify-parity", help="Compare generated CSV outputs against reference 03_summary_results/.")
    p_par.add_argument("--actual-dir", required=True, help="Directory of reproduced CSV outputs.")
    p_par.add_argument("--reference-dir", required=True, help="Zenodo root or 03_summary_results/ directory.")
    p_par.add_argument("--atol", type=float, default=1e-3, help="Absolute numeric tolerance (default 1e-3).")
    p_par.add_argument("--allow-missing", action="store_true", help="Skip reference tables absent from reference-dir.")
    p_par.set_defaults(func=cmd_verify_parity)

    p_fig = sub.add_parser("build-figures", help="Re-render all figures directly from precomputed CSV tables.")
    p_fig.add_argument(
        "--zenodo-dir",
        default=None,
        help="Zenodo archive directory (needed if results-dir lacks cap_country_hazard_counts_2021_2025.csv or ed_table4_validation_summary.csv).",
    )
    p_fig.add_argument("--results-dir", required=True, help="Directory containing summary CSVs.")
    p_fig.add_argument("--figures-dir", required=True, help="Output directory for figures.")
    p_fig.add_argument("--data-inputs-dir", default=None, help="Override path to curated data_inputs/ reference files.")
    p_fig.set_defaults(func=cmd_build_figures)

    p_vz = sub.add_parser("verify-zenodo", help="Verify SHA256SUMS.txt integrity of a local Zenodo archive.")
    p_vz.add_argument("--zenodo-dir", required=True, help="Path to Zenodo archive root.")
    p_vz.set_defaults(func=cmd_verify_zenodo)

    p_fz = sub.add_parser("fetch-zenodo", help="Download a published Zenodo archive record.")
    p_fz.add_argument("--dest-dir", required=True, help="Destination directory for downloaded Zenodo files.")
    p_fz.add_argument(
        "--record-id",
        default=None,
        help="Published Zenodo numeric record ID (or set CAP_GAP_ZENODO_RECORD_ID once minted).",
    )
    p_fz.add_argument("--skip-verify", action="store_true", help="Skip SHA256SUMS.txt verification after download.")
    p_fz.set_defaults(func=cmd_fetch_zenodo)

    p_val = sub.add_parser("audit-validation", help="Tabulate de-identified N=400 CAP keyword validation audit.")
    p_val.add_argument(
        "--input-csv",
        default=None,
        help="Path to de-identified N=400 CSV (defaults to <zenodo-dir>/04_validation_deidentified/validation_n400_deidentified.csv).",
    )
    p_val.add_argument(
        "--zenodo-dir",
        default=None,
        help="Zenodo archive root used to locate validation_n400_deidentified.csv when --input-csv is omitted.",
    )
    p_val.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path for summary table (for 30-column N=400 inputs, also writes sibling validation_confusion_counts.csv and validation_spatial_precision.csv).",
    )
    p_val.add_argument("--label-version", default="v4_reviewed", help="Label version for 30-column N=400 schema.")
    p_val.add_argument("--predictor", default="filter02_broad_flood_storm", help="Predictor flag for 30-column schema.")
    p_val.set_defaults(func=cmd_audit_validation)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
