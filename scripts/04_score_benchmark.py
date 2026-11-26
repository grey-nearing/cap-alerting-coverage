#!/usr/bin/env python3
"""Tier 1 Step 04: Run full scoring engine, tables, and figures from staged Zenodo or working data."""

import sys

from cap_alerting_coverage.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce-tier1", *sys.argv[1:]]))
