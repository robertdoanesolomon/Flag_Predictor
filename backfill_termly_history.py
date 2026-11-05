"""Fill in data/termly/outlook_history.csv (and docs/termly/history.json) for days the page did not record:
each day rebuilt with today's method from the river record as it stood then, without the weather forecasts
(EC46 weighting) the live page used. Days the page itself recorded are never replaced.

    python backfill_termly_history.py 2026-09-01 2026-11-04
"""
from __future__ import annotations

import sys

import pandas as pd

import generate_termly as G

if __name__ == '__main__':
    start, end = sys.argv[1], sys.argv[2]
    for d in pd.date_range(start, end):
        sys.argv = ['generate_termly.py', '--no-fetch', '--rebuild', '--date', f'{d:%Y-%m-%d}']
        G.main()
