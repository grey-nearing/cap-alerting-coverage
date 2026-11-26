"""Global CAP Flood Alerting Coverage & Global Flood Model Benchmark."""

__version__ = "1.0.0"

from cap_alerting_coverage.config import (
    BENCHMARKS,
    CAP_CONTINUOUS_THRESHOLDS,
    CAP_SEVERITIES,
    D_TOTAL,
    DATES,
    FFM_THRESHOLDS,
    ISO_UNIFY,
    LEAD_WINDOWS_DAYS,
    LEGACY_EXCLUDED_19,
    RETURN_PERIODS,
    ProjectPaths,
    date_to_idx,
)

__all__ = [
    "__version__",
    "BENCHMARKS",
    "CAP_CONTINUOUS_THRESHOLDS",
    "CAP_SEVERITIES",
    "D_TOTAL",
    "DATES",
    "FFM_THRESHOLDS",
    "ISO_UNIFY",
    "LEAD_WINDOWS_DAYS",
    "LEGACY_EXCLUDED_19",
    "RETURN_PERIODS",
    "ProjectPaths",
    "date_to_idx",
]
