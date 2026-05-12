#!/usr/bin/env python
"""
Fetch rainfall forecast from Open-Meteo Ensemble API for a location.

Uses the same API and logic as the pipeline; only the set of stations (lat/lon)
differs by location:

  - isis:   13 stations (RAINFALL_STATION_COORDINATES)
  - godstow: 11 stations (same set minus Bicester, Grimsbury)
  - wallingford: 19 stations (13 base + 6 Wallingford-specific)

Usage (from project root):

  python get_rainfall_forecast.py --location isis
  python get_rainfall_forecast.py --location godstow --output data/forecast_rain_godstow.csv
  python get_rainfall_forecast.py --location wallingford --n-members 20
  python get_rainfall_forecast.py --location isis --model ecmwf_ifs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root and src on path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flag_predictor.data.api import get_rainfall_forecast_ensemble  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo ensemble rainfall forecast for a location."
    )
    parser.add_argument(
        "--location",
        choices=["isis", "godstow", "wallingford"],
        default="isis",
        help="Location: isis (13 stations), godstow (11), wallingford (19). Default: isis",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=50,
        help="Number of ensemble members to fetch (default 50).",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=10,
        help="Forecast horizon in days (default 10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ecmwf_aifs025",
        help="Open-Meteo ensemble model: ecmwf_aifs025 (default), ecmwf_ifs, etc.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save CSV (e.g. data/forecast_rain_isis.csv).",
    )
    args = parser.parse_args()

    if args.output is not None and not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output

    print(f"Fetching rainfall forecast for {args.location.upper()}...")
    print(f"  Stations: isis=13, godstow=11, wallingford=19")
    print(f"  model={args.model}, n_members={args.n_members}, forecast_days={args.forecast_days}")

    df = get_rainfall_forecast_ensemble(
        location=args.location,
        ensemble_model=args.model,
        n_members=args.n_members,
        forecast_days=args.forecast_days,
    )

    if df.empty:
        print("No data returned.")
        sys.exit(1)

    print(f"\n✓ Shape: {df.shape}")
    print(f"  Index: {df.index[0]} to {df.index[-1]} (hourly)")
    print(f"  Columns (sample): {list(df.columns[:4])} ...")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output)
        print(f"  Saved: {args.output}")
    else:
        print("\nFirst 3 rows (first 3 columns):")
        print(df.iloc[:3, :3].to_string())


if __name__ == "__main__":
    main()
