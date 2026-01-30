#!/usr/bin/env python
"""
Generate per-location forecast figures matching the
`all_locations_visualization.ipynb` notebook and save them to a folder.

This script reproduces the **spaghetti + rainfall** figure for each
configured location (Isis, Godstow, Wallingford).

Usage (from project root):

    python generate_all_location_figures.py --output-dir figures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# ---------------------------------------------------------------------------
# Project / import setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure `src` is on the path (mirrors the notebook setup, but from project root)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from flag_predictor import get_location_config  # type: ignore  # noqa: E402
from flag_predictor.config import (  # type: ignore  # noqa: E402
    FLAG_COLORS,
    LOCATIONS,
    RAINFALL_STATION_NAMES,
    WALLINGFORD_RAINFALL_STATION_NAMES,
    get_flag_thresholds,
)
from flag_predictor.data.api import (  # type: ignore  # noqa: E402
    get_rainfall_forecast_ensemble,
)
from flag_predictor.models import load_model  # type: ignore  # noqa: E402
from flag_predictor.pipeline import prepare_training_data  # type: ignore  # noqa: E402
from flag_predictor.prediction import predict_ensemble  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Helper utilities (ported from the notebook)
# ---------------------------------------------------------------------------

def get_location_station_names(location: str) -> list[str]:
    """
    Get the correct rainfall station names for a given location.

    - ISIS: All default stations (13 stations)
    - Godstow: Default stations EXCEPT Bicester and Grimsbury (11 stations)
    - Wallingford: All default stations PLUS Wallingford-specific stations (19 stations)
    """
    location_lower = location.lower()

    if location_lower == "wallingford":
        return list(RAINFALL_STATION_NAMES) + list(WALLINGFORD_RAINFALL_STATION_NAMES)
    elif location_lower == "godstow":
        return [s for s in RAINFALL_STATION_NAMES if s not in {"Bicester", "Grimsbury"}]
    else:
        return list(RAINFALL_STATION_NAMES)


def _ensure_timezone_naive(index: pd.Index) -> pd.Index:
    """Make a DatetimeIndex timezone-naive if needed."""
    if hasattr(index, "tz") and index.tz is not None:
        return index.tz_localize(None)
    return index


def generate_spaghetti_figure(
    location: str,
    output_dir: Path,
    n_members: int = 50,
    project_root: Path | None = None,
) -> None:
    """
    Generate the spaghetti + rainfall figure for a single location and save it.

    This closely follows STEP 5 in `all_locations_visualization.ipynb`.
    """
    project_root = project_root or PROJECT_ROOT

    print(f"\n{'=' * 80}")
    print(f"GENERATING FIGURE FOR: {location.upper()}")
    print(f"{'=' * 80}")

    config = get_location_config(location)
    flag_thresholds = get_flag_thresholds(location)

    # ------------------------------------------------------------------
    # STEP 1: Historical data (matches notebook call)
    # ------------------------------------------------------------------
    merged_df, X, y_multi = prepare_training_data(
        location=location,
        project_root=project_root,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # STEP 2: Load model (use location-specific latest files)
    # ------------------------------------------------------------------
    models_dir = project_root / "models"

    # Use the 2026-01 experiment models for all locations, matching the
    # notebook `USE_EXPERIMENT_MODEL = True` behaviour.
    model_path = models_dir / f"multihorizon_model_experiment_2026_01_{location}.pth"
    scaler_path = models_dir / f"scaler_experiment_2026_01_{location}.pkl"
    config_path = models_dir / f"config_experiment_2026_01_{location}.pkl"

    try:
        model, scaler, model_config = load_model(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            config_path=str(config_path),
        )
    except FileNotFoundError as exc:
        # If the experiment artefacts for this location are not present in the
        # repo (e.g. not committed to GitHub), skip this location instead of
        # failing the whole workflow.
        print(f"\n[WARNING] Skipping location '{location}' – {exc}")
        return

    feature_columns = model_config["feature_columns"]
    sequence_length = model_config["sequence_length"]
    horizons = model_config["horizons"]

    print("\nModel configuration:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden sizes: {model_config['hidden_sizes']}")
    print(f"  Features: {len(feature_columns)}")

    # ------------------------------------------------------------------
    # STEP 3: Fetch ensemble rainfall forecast (location-specific)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"STEP 3: Fetching location-specific rainfall forecast for {location.upper()}")
    print(f"{'=' * 70}")

    rainfall_forecast = get_rainfall_forecast_ensemble(
        location=location,
        n_members=n_members,
    )

    print(f"\n✓ Rainfall forecast: {rainfall_forecast.shape}")
    print(f"  Time range: {rainfall_forecast.index[0]} to {rainfall_forecast.index[-1]}")
    print(f"  Stations loaded: {len(get_location_station_names(location))}")

    # ------------------------------------------------------------------
    # STEP 4: Ensemble prediction
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"STEP 4: Running ensemble prediction for {location.upper()}")
    print(f"{'=' * 70}")

    station_names_list = get_location_station_names(location)

    ensemble_predictions = predict_ensemble(
        model=model,
        scaler=scaler,
        historical_df=merged_df,
        rainfall_ensemble_df=rainfall_forecast,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizons=horizons,
        station_names=station_names_list,
        n_members=n_members,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # STEP 5: Visualisation (spaghetti + rainfall bars)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"STEP 5: Generating visualizations for {location.upper()}")
    print(f"{'=' * 70}")

    # Make data timezone-naive for plotting
    plot_df = ensemble_predictions.copy()
    plot_df.index = _ensure_timezone_naive(plot_df.index)

    # Calculate ensemble statistics (same as notebook)
    plot_stats = pd.DataFrame(
        {
            "mean": plot_df.mean(axis=1),
            "std": plot_df.std(axis=1),
            "min": plot_df.min(axis=1),
            "max": plot_df.max(axis=1),
            "p05": plot_df.quantile(0.05, axis=1),
            "p10": plot_df.quantile(0.10, axis=1),
            "p25": plot_df.quantile(0.25, axis=1),
            "p75": plot_df.quantile(0.75, axis=1),
            "p90": plot_df.quantile(0.90, axis=1),
            "p95": plot_df.quantile(0.95, axis=1),
        }
    )

    n_members_used = len(ensemble_predictions.columns)

    # Create figure with dual y-axis
    fig, ax = plt.subplots(figsize=(20, 12))
    ax_rain = ax.twinx()  # Secondary axis for rainfall

    # Add flag boundaries as horizontal filled regions (skip for Wallingford - white background)
    if location.lower() != "wallingford":
        ax.axhspan(
            -4,
            flag_thresholds["light_blue"][0],
            color=FLAG_COLORS["green"],
            alpha=0.08,
            zorder=0,
        )
        ax.axhspan(
            flag_thresholds["light_blue"][0],
            flag_thresholds["dark_blue"][0],
            color=FLAG_COLORS["light_blue"],
            alpha=0.08,
            zorder=0,
        )
        ax.axhspan(
            flag_thresholds["dark_blue"][0],
            flag_thresholds["amber"][0],
            color=FLAG_COLORS["dark_blue"],
            alpha=0.08,
            zorder=0,
        )
        ax.axhspan(
            flag_thresholds["amber"][0],
            flag_thresholds["red"][0],
            color=FLAG_COLORS["amber"],
            alpha=0.08,
            zorder=0,
        )
        ax.axhspan(
            flag_thresholds["red"][0],
            4,
            color=FLAG_COLORS["red"],
            alpha=0.08,
            zorder=0,
        )

    # ============= RAINFALL DATA PREPARATION =============
    print("Preparing rainfall data...")

    # 1. Historical rainfall (last 8 days) - aggregate across all stations
    hist_merged = merged_df.copy()
    hist_merged.index = _ensure_timezone_naive(hist_merged.index)

    last_8_days = hist_merged.iloc[-24 * 8 :].copy()

    # Get location-specific station names (excludes Bicester and Grimsbury for godstow,
    # includes Wallingford-specific for wallingford)
    station_names_list = get_location_station_names(location)

    # Filter to only location-specific stations
    rainfall_cols = [col for col in last_8_days.columns if col in station_names_list]
    historical_rainfall_hourly = last_8_days[rainfall_cols].mean(axis=1)  # Average across stations
    historical_rainfall_daily = historical_rainfall_hourly.resample("1D").sum()  # Daily totals

    # 2. Forecast rainfall - calculate ensemble statistics
    rainfall_forecast_naive = rainfall_forecast.copy()
    rainfall_forecast_naive.index = _ensure_timezone_naive(rainfall_forecast_naive.index)

    # Calculate average rainfall per ensemble member (mean across all stations)
    member_totals = pd.DataFrame(index=rainfall_forecast_naive.index)
    for member_idx in range(n_members_used):
        member_cols = [f"{station}_member_{member_idx}" for station in station_names_list]
        existing_cols = [col for col in member_cols if col in rainfall_forecast_naive.columns]
        if existing_cols:
            member_totals[f"member_{member_idx}"] = rainfall_forecast_naive[existing_cols].mean(
                axis=1
            )

    # Resample to daily for cleaner visualization
    member_totals_daily = member_totals.resample("1D").sum()

    # Calculate rainfall ensemble statistics
    forecast_rain_mean = member_totals_daily.mean(axis=1)
    forecast_rain_p10 = member_totals_daily.quantile(0.10, axis=1)
    forecast_rain_p90 = member_totals_daily.quantile(0.90, axis=1)

    # Error bars: distance from mean to percentiles
    error_lower = forecast_rain_mean - forecast_rain_p10
    error_upper = forecast_rain_p90 - forecast_rain_mean

    print(
        f"  Historical rainfall: {len(historical_rainfall_daily)} daily bars "
        f"(avg across stations)"
    )
    print(
        f"  Forecast rainfall: {len(forecast_rain_mean)} daily bars "
        f"with ensemble spread (P10-P90)"
    )

    # ============= PLOT RAINFALL BARS =============
    bar_width = 0.8  # Width in days

    # Historical rainfall bars (gray)
    ax_rain.bar(
        historical_rainfall_daily.index,
        historical_rainfall_daily.values,
        width=bar_width,
        color="gray",
        alpha=0.4,
        label="Historical Rainfall",
        zorder=1,
    )

    # Forecast rainfall bars with error bars showing ensemble spread
    ax_rain.bar(
        forecast_rain_mean.index,
        forecast_rain_mean.values,
        width=bar_width,
        color="cornflowerblue",
        alpha=0.5,
        yerr=[error_lower.values, error_upper.values],
        error_kw={
            "elinewidth": 1.5,
            "capsize": 3,
            "capthick": 1,
            "alpha": 0.7,
            "color": "navy",
        },
        label="Forecast Rainfall (mean ± 10th-90th percentile)",
        zorder=2,
    )

    # ============= PLOT RIVER DIFFERENTIAL =============
    # Determine forecast start time (where forecasts begin)
    forecast_start_time = plot_df.index[0]

    # Extend historical data to forecast start time if there's a gap
    # This ensures the historical line connects seamlessly with forecasts
    historical_to_plot = last_8_days.copy()
    if historical_to_plot.index[-1] < forecast_start_time:
        # Fill gap by forward-filling the last known value
        gap_times = pd.date_range(
            start=historical_to_plot.index[-1] + pd.Timedelta(hours=1),
            end=forecast_start_time,
            freq="1h",
        )
        gap_data = pd.Series(
            [historical_to_plot["differential"].iloc[-1]] * len(gap_times),
            index=gap_times,
        )
        historical_to_plot = pd.concat(
            [
                historical_to_plot,
                pd.DataFrame({"differential": gap_data}),
            ]
        )
    elif historical_to_plot.index[-1] > forecast_start_time:
        # Trim historical data to end at forecast start
        historical_to_plot = historical_to_plot[historical_to_plot.index <= forecast_start_time]

    # Plot historical differential as a solid black line
    ax.plot(
        historical_to_plot.index,
        historical_to_plot["differential"].values,
        color="black",
        linewidth=3,
        label="Historical Differential",
        zorder=100,
        alpha=0.9,
    )

    # Current time marker (where forecasts start)
    current_time = forecast_start_time
    ax.axvline(
        x=current_time,
        color="red",
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="Now",
        zorder=101,
    )

    # Plot ALL ensemble member river forecasts (spaghetti plot)
    print(f"Plotting {n_members_used} ensemble member trajectories...")

    for idx, col in enumerate(plot_df.columns):
        if idx == 0:
            ax.plot(
                plot_df.index,
                plot_df[col].values,
                color="steelblue",
                linewidth=1.2,
                alpha=0.5,
                label=f"Ensemble Predictions (n={n_members_used})",
                zorder=50,
            )
        else:
            ax.plot(
                plot_df.index,
                plot_df[col].values,
                color="steelblue",
                linewidth=1.2,
                alpha=0.5,
                zorder=50,
            )

    # Overlay the ensemble MEAN as a bold line
    ax.plot(
        plot_df.index,
        plot_stats["mean"].values,
        color="darkviolet",
        linewidth=3,
        label="Ensemble Mean",
        zorder=102,
        alpha=1,
    )

    # ============= FORMATTING =============
    # X-axis: "Wed 4 May" style (weekday, day, month)
    def _short_date(x, pos=None):
        d = mdates.num2date(x)
        return d.strftime("%a ") + str(d.day) + " " + d.strftime("%b")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_short_date))
    ax.set_xlabel("Date", fontsize=20, fontweight="bold")
    ax.set_ylabel("Height Differential (m)", fontsize=20, fontweight="bold", color="black")
    ax_rain.set_ylabel(
        "Rainfall (mm/day, avg across stations)",
        fontsize=20,
        fontweight="bold",
        color="cornflowerblue",
    )

    ax.tick_params(axis="both", labelsize=16, labelcolor="black")
    ax_rain.tick_params(axis="y", labelsize=16, labelcolor="cornflowerblue")

    # Set y-limits
    ax.set_ylim(-0.1, max(1.1, plot_stats["max"].max() + 0.1))
    max_rain = (
        max(historical_rainfall_daily.max(), forecast_rain_p90.max())
        if len(forecast_rain_p90) > 0
        else historical_rainfall_daily.max()
    )
    ax_rain.set_ylim(0, max_rain * 1.3)

    ax.set_title(
        f"{location.upper()}",
        fontsize=32,
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=14,
        loc="upper left",
        framealpha=0.9,
    )

    # Add day markers
    current_time = forecast_start_time
    for i in range(1, 11):
        day_time = current_time + pd.Timedelta(days=i)
        if day_time <= plot_df.index[-1]:
            ax.axvline(x=day_time, color="gray", linestyle=":", alpha=0.3, zorder=0)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save instead of showing
    output_path = output_dir / f"spaghetti_rain_{location.lower()}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Spaghetti + rainfall figure saved to: {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the three per-location figures from "
            "notebooks/all_locations_visualization.ipynb and save them."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory to save figures (relative to project root or absolute).",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=50,
        help="Number of ensemble members to use for the forecast.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Global plot style (matches notebook)
    plt.rcParams["figure.figsize"] = (18, 10)
    plt.rcParams["font.size"] = 11

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    locations_to_process = list(LOCATIONS.keys())
    print(f"Processing all locations: {locations_to_process}")
    print(f"Total locations: {len(locations_to_process)}")

    for location in locations_to_process:
        generate_spaghetti_figure(
            location=location,
            output_dir=output_dir,
            n_members=args.n_members,
            project_root=PROJECT_ROOT,
        )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

