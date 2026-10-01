"""
Days-before forecasts for a whole year: September LSTM vs the physics hybrid.

For every day D, take the forecast issued at 00z on D-N and keep its hours
+24N..+24N+23 (i.e. day D as forecast N days before; default N=1). Stitching those together
gives one "what we said yesterday" line for the year, plotted against the
observed differential. Rainfall is what actually fell.

Usage:
    python plot_day_before.py [--year 2025] [--days-before 3] [isis godstow wallingford]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

from plot_redesign_month import MODELS, flag_bands, predictors  # noqa: E402
from flag_predictor.evaluation import future_rain_frame, load_merged, mean_station_rain  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / 'figures' / 'eval'


def start_hour(diff: pd.Series, t: pd.Timestamp):
    """00z on the day, or the latest observed hour in the 6 h before it."""
    window = diff.loc[t - pd.Timedelta(hours=6):t].dropna()
    return window.index[-1] if len(window) else None


def day_before_series(location: str, year: int, days_before: int = 1) -> tuple[pd.Series, dict]:
    merged = load_merged(location, PROJECT_ROOT)
    tz = merged.index.tz
    diff = merged['differential']
    preds = predictors(location)
    out = {label: [] for label in preds}
    days = pd.date_range(pd.Timestamp(f'{year}-01-01', tz=tz), pd.Timestamp(f'{year}-12-31', tz=tz), freq='D')
    for day in days:
        t0 = start_hour(diff, day - pd.Timedelta(days=days_before))
        if t0 is None:
            continue
        hist, fut = merged.loc[:t0], future_rain_frame(merged, t0)
        target = pd.date_range(day, periods=24, freq='1h')
        for label, predict in preds.items():
            values = pd.Series(predict(hist, fut), index=pd.date_range(t0, periods=241, freq='1h'))  # target = day D
            out[label].append(values.reindex(target))
        if day.day == 1:
            print(f"  {location} {day:%b}", flush=True)
    end = pd.Timestamp(f'{year + 1}-01-01', tz=tz)
    obs = diff.loc[days[0]:end]
    return obs, {label: pd.concat(parts) for label, parts in out.items()}, mean_station_rain(merged).loc[days[0]:end]


def plot(location: str, year: int, obs: pd.Series, lines: dict, rain: pd.Series, days_before: int = 1) -> Path:
    when = 'the day before' if days_before == 1 else f'{days_before} days before'
    lead = f'{24 * days_before}–{24 * days_before + 24} h ahead'
    fig, axes = plt.subplots(3, 1, figsize=(22, 13), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0.08})
    top = float(np.nanmax([obs.max()] + [s.max() for s in lines.values()])) + 0.05
    bottom = min(-0.1, float(np.nanmin([obs.min()] + [s.min() for s in lines.values()])) - 0.03)
    for ax, (label, color) in zip(axes[:2], MODELS):
        s = lines[label]
        err = (s - obs.reindex(s.index)).abs()
        flag_bands(ax, location, top)
        ax.plot(obs.index, obs.values, color='black', lw=1.4, label='Observed differential', zorder=5)
        ax.plot(s.index, s.values, color=color, lw=1.2, alpha=0.9, zorder=6,
                label=f'{label}: forecast made {when} (MAE {np.nanmean(err):.3f} m)')
        ax.set_ylim(bottom, top)
        ax.set_ylabel('Differential (m)')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', framealpha=0.92)
    axes[0].set_title(f'{location.title()} {year}: each day as forecast at 00z {when} '
                      f'({lead}), rainfall = what actually fell', fontsize=14)
    daily = rain.resample('D').sum()
    axes[2].bar(daily.index, daily.values, width=1.0, color='0.55')
    axes[2].set_ylabel('Rain (mm/day,\nmean of gauges)')
    axes[2].grid(alpha=0.25, axis='y')
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[2].set_xlim(obs.index[0], obs.index[-1])
    prefix = 'day_before' if days_before == 1 else f'{days_before}_days_before'
    out = FIGURES_DIR / f'{prefix}_{year}_{location}_sept_vs_hybrid.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('locations', nargs='*', default=['isis', 'godstow', 'wallingford'])
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--days-before', type=int, default=1, choices=range(1, 10))
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for location in args.locations:
        obs, lines, rain = day_before_series(location, args.year, args.days_before)
        for label, s in lines.items():
            err = (s - obs.reindex(s.index)).abs()
            print(f"  {location} {label}: {args.days_before}-day-before MAE {np.nanmean(err):.4f} m over {err.notna().sum()} h")
        print(f"saved {plot(location, args.year, obs, lines, rain, args.days_before)}", flush=True)


if __name__ == '__main__':
    main()
