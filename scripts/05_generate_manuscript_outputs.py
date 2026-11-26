#!/usr/bin/env python3
"""Tier 1 Step 05: Re-render all manuscript and Extended Data figures from summary CSV tables."""

import sys

from cap_alerting_coverage.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-figures", *sys.argv[1:]]))
