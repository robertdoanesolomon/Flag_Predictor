"""
Month comparison plots: September LSTM vs the physics hybrid.

Generalises the two February 2025 figures (plot_feb2025_clairvoyant_isis.py and
plot_feb20_uncertain_rainfall_isis.py) to any month, with the physics-hybrid
ensemble alongside the September LSTM (clamps on, as on the old page).

1. Every 00z start in the month, 10-day forecasts with the rain that actually
   fell (one figure per location).
2. Isis from 00z on --scenario-start with 36 synthetic rain scenarios
   (0–1.2x the actual rain, random noise, +/-1 day shifts), as in the
   original script.

Usage:
    python plot_redesign_month.py                       # February 2025, 20 Feb scenarios
    python plot_redesign_month.py --month 2025-11 --scenario-start 2025-11-10
    python plot_redesign_month.py --month 2025-11 isis   # one location, no scenarios
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

from evaluate_candidates import september  # noqa: E402
from flag_predictor.config import FLAG_COLORS, get_flag_thresholds  # noqa: E402
from flag_predictor.evaluation import future_rain_frame, load_merged, mean_station_rain  # noqa: E402
from flag_predictor.models.candidates import ensemble_predictor  # noqa: E402
import plot_feb20_uncertain_rainfall_isis as feb20  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / 'figures' / 'eval'
HYBRID = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]
MODELS = [
    ('September 2026 (live)', '#2ca02c'),
    ('Physics hybrid (new)', '#d62728'),
]


def predictors(location: str):
    return {
        MODELS[0][0]: september(location, 'full', True),
        MODELS[1][0]: ensemble_predictor(HYBRID, location),
    }


def as_series(values: np.ndarray, t0: pd.Timestamp) -> pd.Series:
    return pd.Series(values, index=pd.date_range(t0, periods=len(values), freq='1h'))


def flag_bands(ax, location: str, top: float) -> None:
    if location == 'wallingford':
        return
    for key, (lo, hi) in get_flag_thresholds(location).items():
        if hi <= lo:
            continue
        lo = max(lo, -0.2)
        hi = min(hi, top)
        ax.axhspan(lo, hi, color=FLAG_COLORS[key], alpha=0.07, zorder=0)


def month_daily(location: str, month: str) -> Path:
    merged = load_merged(location, PROJECT_ROOT)
    tz = merged.index.tz
    start = pd.Timestamp(f'{month}-01', tz=tz)
    end = start + pd.offsets.MonthBegin(1)
    t0s = [t for t in pd.date_range(start, periods=(end - start).days, freq='D')
           if np.isfinite(merged['differential'].get(t, np.nan))]
    preds = predictors(location)
    actual = merged['differential']
    runs = {label: {} for label in preds}
    for t0 in t0s:
        hist, fut = merged.loc[:t0], future_rain_frame(merged, t0)
        for label, predict in preds.items():
            runs[label][t0] = as_series(predict(hist, fut), t0)
        print(f"  {location} {t0:%d %b}", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0.08})
    obs = actual.loc[start:end]
    top = float(np.nanmax([obs.max()] + [s.max() for r in runs.values() for s in r.values()])) + 0.05
    for ax, (label, color) in zip(axes[:2], MODELS):
        flag_bands(ax, location, top)
        errs = []
        for t0, s in runs[label].items():
            ax.plot(s.loc[:end].index, s.loc[:end].values, color=color, lw=0.9, alpha=0.45)
            a = actual.reindex(s.index)
            errs.append(float(np.nanmean(np.abs(s.values - a.values)[1:])))
        ax.plot(obs.index, obs.values, color='black', lw=2.4, label='Observed differential', zorder=10)
        ax.plot([], [], color=color, lw=1.5,
                label=f'{label}: {len(runs[label])} × 00z forecasts, mean 10-day MAE {np.mean(errs):.3f} m')
        ax.set_ylabel('Differential (m)')
        ax.set_ylim(min(-0.05, float(obs.min()) - 0.05), top)
        ax.legend(loc='upper left', framealpha=0.92)
        ax.grid(alpha=0.25)
    axes[0].set_title(f'{location.title()}: forecasts from every 00z in {start:%B %Y}, '
                      'rainfall = what actually fell', fontsize=13)
    rain = mean_station_rain(merged).loc[start:end].resample('D').sum()
    axes[2].bar(rain.index, rain.values, width=0.9, color='0.6', edgecolor='0.45', lw=0.3)
    axes[2].set_ylabel('Rain (mm/day,\nmean of gauges)')
    axes[2].grid(alpha=0.25, axis='y')
    axes[2].set_xlim(start, end)
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right')
    out = FIGURES_DIR / (f'{start:%b%Y}'.lower() + f'_clairvoyant_{location}_sept_vs_hybrid.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def uncertain_scenarios(start_date: str) -> Path:
    location = 'isis'
    merged = load_merged(location, PROJECT_ROOT)
    t0 = pd.Timestamp(start_date, tz=merged.index.tz)
    hist = merged.loc[:t0]
    base = future_rain_frame(merged, t0)
    scenarios = feb20.build_scenarios(base)
    preds = predictors(location)
    runs = {label: [as_series(p(hist, rain), t0) for _, rain in scenarios] for label, p in preds.items()}
    clair = {label: as_series(p(hist, base), t0) for label, p in preds.items()}
    end = t0 + pd.Timedelta(hours=240)
    obs = merged['differential'].loc[t0:end]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1.2], 'hspace': 0.08})
    top = max(1.2, float(np.nanmax([s.max() for r in runs.values() for s in r])) + 0.05)
    bottom = min(0.0, float(np.nanmin([obs.min()] + [s.min() for r in runs.values() for s in r])) - 0.05)
    for ax, (label, color) in zip(axes[:2], MODELS):
        flag_bands(ax, location, top)
        for s in runs[label]:
            ax.plot(s.index, s.values, color=color, lw=0.9, alpha=0.35)
        ax.plot(obs.index, obs.values, color='black', lw=2.6, label='Observed differential', zorder=10)
        ax.plot(clair[label].index, clair[label].values, color=color, lw=2.4, ls='--',
                label=f'{label}, actual rain', zorder=9)
        ax.plot([], [], color=color, lw=1.5, alpha=0.8,
                label=f'{label}, synthetic rain scenarios (n={len(scenarios)})')
        spread = pd.DataFrame({i: s for i, s in enumerate(runs[label])})
        ax.set_ylabel('Differential (m)')
        ax.set_ylim(bottom, top)
        ax.legend(loc='upper left', framealpha=0.92)
        ax.grid(alpha=0.25)
        ax.text(0.99, 0.03, f'spread at +240h: {spread.iloc[-1].min():.2f}–{spread.iloc[-1].max():.2f} m',
                transform=ax.transAxes, ha='right', fontsize=10)
    axes[0].set_title(
        f'Isis from 00z {t0:%-d %b %Y} (differential {obs.iloc[0]:.3f} m): '
        f'{len(scenarios)} synthetic rain scenarios, 0–{feb20.MAX_RAIN_SCALE}× actual, noise and ±1 day shifts',
        fontsize=12)
    daily = pd.DataFrame({name: rain.clip(lower=0).mean(axis=1).resample('D').sum()
                          for name, rain in scenarios})
    actual_daily = base.clip(lower=0).mean(axis=1).resample('D').sum()
    axes[2].fill_between(daily.index, daily.quantile(0.1, axis=1), daily.quantile(0.9, axis=1),
                         color='#1f77b4', alpha=0.25, label='Scenario P10–P90')
    axes[2].bar(actual_daily.index, actual_daily.values, width=0.85, color='0.55',
                edgecolor='0.4', lw=0.3, label='Actual daily rain')
    axes[2].set_ylabel('Rain (mm/day,\nmean of gauges)')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(alpha=0.25, axis='y')
    axes[2].set_xlim(t0, end)
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right')
    out = FIGURES_DIR / f'{t0:%b}{t0.day}_uncertain_rainfall_isis_sept_vs_hybrid.png'.lower()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('locations', nargs='*', default=['isis', 'godstow', 'wallingford'])
    parser.add_argument('--month', default='2025-02', help='YYYY-MM')
    parser.add_argument('--scenario-start', default=None,
                        help='YYYY-MM-DD for the Isis rain-scenario test (default: 20 Feb 2025 '
                             'for February 2025, otherwise skipped)')
    args = parser.parse_args()
    scenario_start = args.scenario_start or (feb20.T0 if args.month == '2025-02' else None)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for location in args.locations:
        print(f"saved {month_daily(location, args.month)}", flush=True)
    if scenario_start and 'isis' in args.locations:
        print(f"saved {uncertain_scenarios(scenario_start)}", flush=True)


if __name__ == '__main__':
    main()
