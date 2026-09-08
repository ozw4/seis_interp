#!/usr/bin/env python3
"""Thin entry point for a single frozen C3 first-results action."""

from seis_interp.pipelines.c3_first_results import main

if __name__ == "__main__":
    raise SystemExit(main())
