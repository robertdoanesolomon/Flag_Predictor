"""
Live forecast right now: live September model vs the physics hybrid.

The September model runs exactly as on the website (production history from
prepare_training_data with API data, its own ensemble-rain request, clamps).
The physics hybrid gets a continuous hourly history: rain from recent EA gauge
files (download them first, see below), Farmoor flow from the local record
topped up with the API, and the API differential. Both start at the same hour
and see the same 50 ECMWF AIFS ensemble members.

Setup (recent gauge rain, kept apart from the training CSVs):
    python src/flag_predictor/data/download_rainfall.py --min-date <~60 days ago> \
        --output-dir data/recent_rainfall

Usage:
    python forecast_now_redesign.py [isis godstow wallingford]

Writes figures/eval/now_{location}_sept_vs_hybrid.png and the member
trajectories to figures/eval/now_{location}_members.csv.
"""

from __future__ import annotations

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
warnings.filterwarnings('ignore')

from flag_predictor.config import FLAG_COLORS, get_flag_thresholds  # noqa: E402
from flag_predictor.data.api import get_rainfall_forecast_ensemble  # noqa: E402
from flag_predictor.data.loader import load_historical_flow, load_historical_rainfall  # noqa: E402
from flag_predictor.evaluation import load_merged, mean_station_rain  # noqa: E402
from flag_predictor.models.candidates import ensemble_predictor  # noqa: E402
from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.pipeline import prepare_training_data  # noqa: E402
from flag_predictor.prediction.forecast import predict_ensemble  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models'
OUT_DIR = PROJECT_ROOT / 'figures' / 'eval'
HYBRID = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]
N_MEMBERS = 50
HORIZON = 240
FLAG_ORDER = ['green', 'light_blue', 'dark_blue', 'amber', 'red']


def station_names_for(location: str) -> list[str]:
    """Same gauge lists as generate_all_location_figures.get_location_station_names."""
    from flag_predictor.config import RAINFALL_STATION_NAMES, WALLINGFORD_RAINFALL_STATION_NAMES
    if location == 'wallingford':
        return list(RAINFALL_STATION_NAMES) + list(WALLINGFORD_RAINFALL_STATION_NAMES)
    if location == 'godstow':
        return [s for s in RAINFALL_STATION_NAMES if s not in {'Bicester', 'Grimsbury'}]
    return list(RAINFALL_STATION_NAMES)


def run_september(location: str, merged_df: pd.DataFrame, rain_forecast: pd.DataFrame) -> pd.DataFrame:
    name = f'experiment_2026_09_{location}'
    model, scaler, cfg = load_model(
        model_path=MODELS_DIR / f'multihorizon_model_{name}.pth',
        scaler_path=MODELS_DIR / f'scaler_{name}.pkl',
        config_path=MODELS_DIR / f'config_{name}.pkl',
    )
    flow = (None, None, None)
    if cfg.get('uses_predicted_flow'):
        flow = load_model(
            model_path=MODELS_DIR / 'multihorizon_model_experiment_2026_09_farmoor.pth',
            scaler_path=MODELS_DIR / 'scaler_experiment_2026_09_farmoor.pkl',
            config_path=MODELS_DIR / 'config_experiment_2026_09_farmoor.pkl',
        )
    return predict_ensemble(
        model=model, scaler=scaler, historical_df=merged_df, rainfall_ensemble_df=rain_forecast,
        feature_columns=cfg['feature_columns'], sequence_length=cfg['sequence_length'],
        horizons=cfg.get('horizons'), station_names=station_names_for(location),
        n_members=N_MEMBERS, predicts_delta=True,
        max_recession_m_per_day=cfg.get('max_recession_m_per_day'), verbose=False,
        model_config=cfg, flow_model=flow[0], flow_scaler=flow[1], flow_config=flow[2],
    )


def live_frame(location: str, merged_df: pd.DataFrame) -> pd.DataFrame:
    """Continuous hourly history: recent gauge rain, flow and the API differential."""
    stations = [c for c in load_merged(location, PROJECT_ROOT).columns
                if c not in ('differential', 'flow_m3s_Farmoor')]
    t0 = merged_df.index[-1]
    raw = load_historical_rainfall(rainfall_dir='data/recent_rainfall/', project_root=PROJECT_ROOT,
                                   location=location, verbose=False)
    if raw.index.tz is None:
        raw = raw.tz_localize('UTC')
    raw = raw.rename(columns=lambda c: (c.split('mm_', 1)[1] if 'mm_' in c else c).split('-')[0])
    raw = raw.loc[:, ~raw.columns.duplicated()]
    start = raw.index.min().ceil('h')
    grid = pd.date_range(start, t0, freq='1h')
    rain = (raw.reindex(columns=stations).astype(float).resample('1h').sum(min_count=1)
            .clip(lower=0, upper=50).reindex(grid))

    flow = load_historical_flow(project_root=PROJECT_ROOT)
    if flow.index.tz is None:
        flow = flow.tz_localize('UTC')
    flow = flow['flow_m3s_Farmoor'].astype(float).resample('1h').mean().reindex(grid)
    api_flow = merged_df['flow_m3s_Farmoor'].reindex(grid)
    flow = api_flow.combine_first(flow).clip(lower=0)

    diff = merged_df['differential'].reindex(grid)
    return pd.concat([diff.rename('differential'), rain, flow.rename('flow_m3s_Farmoor')], axis=1)


def run_hybrid(location: str, frame: pd.DataFrame, rain_forecast: pd.DataFrame) -> pd.DataFrame:
    t0 = frame.index[-1]
    fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
    stations = [c for c in frame.columns if c not in ('differential', 'flow_m3s_Farmoor')]
    forecast = rain_forecast.copy()
    if forecast.index.tz is None:
        forecast.index = forecast.index.tz_localize('UTC')
    predict = ensemble_predictor(HYBRID, location)
    out = {}
    for i in range(N_MEMBERS):
        cols = {f'{s}_member_{i}': s for s in stations if f'{s}_member_{i}' in forecast.columns}
        if not cols:
            continue
        member = forecast[list(cols)].rename(columns=cols).reindex(fut_idx).fillna(0.0)
        out[f'member_{i}'] = predict(frame, member)
    return pd.DataFrame(out, index=pd.date_range(t0, periods=HORIZON + 1, freq='1h'))


def flag_probabilities(members: pd.DataFrame, location: str) -> pd.DataFrame:
    probs = {}
    for key, (lo, hi) in get_flag_thresholds(location).items():
        if hi <= lo:
            continue
        probs[key] = ((members >= lo) & (members < hi)).mean(axis=1)
    return pd.DataFrame(probs)


def plot(location, t0, observed, sept, hybrid, rain_mean) -> Path:
    has_flags = location != 'wallingford'
    rows = 3 if has_flags else 2
    ratios = [3, 1.2, 1.2] if has_flags else [3, 1.2]
    fig, axes = plt.subplots(rows, 2, figsize=(18, 11 if has_flags else 9), sharex=True,
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.1, 'wspace': 0.08})
    hist = observed.loc[t0 - pd.Timedelta(days=5):t0]
    lo = min(float(hist.min()), float(np.nanmin(sept.values)), float(np.nanmin(hybrid.values))) - 0.05
    hi = max(float(hist.max()), float(np.nanmax(sept.values)), float(np.nanmax(hybrid.values))) + 0.05
    thresholds = get_flag_thresholds(location)
    for col, (label, members, color) in enumerate([
        ('September 2026 (live)', sept, '#2ca02c'),
        ('Physics hybrid (new)', hybrid, '#d62728'),
    ]):
        ax = axes[0, col]
        if has_flags:
            for key, (a, b) in thresholds.items():
                if b > a:
                    ax.axhspan(max(a, lo), min(b, hi), color=FLAG_COLORS[key], alpha=0.08, zorder=0)
        for c in members.columns:
            ax.plot(members.index, members[c], color=color, lw=0.6, alpha=0.3)
        ax.plot(members.index, members.median(axis=1), color=color, lw=2.4, label=f'{label}: median of {members.shape[1]} members')
        ax.fill_between(members.index, members.quantile(0.1, axis=1), members.quantile(0.9, axis=1),
                        color=color, alpha=0.12, label='10–90% range')
        ax.plot(hist.index, hist.values, color='black', lw=2.2, label='Observed (last 5 days)')
        ax.axvline(t0, color='0.3', ls=':', lw=1)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.set_title(label, fontsize=12)
        if col == 0:
            ax.set_ylabel('Differential (m)')
        if has_flags:
            axp = axes[1, col]
            probs = flag_probabilities(members, location)
            probs = probs[[k for k in FLAG_ORDER if k in probs.columns]]
            axp.stackplot(probs.index, probs.T.values, colors=[FLAG_COLORS[k] for k in probs.columns],
                          alpha=0.85, labels=[k.replace('_', ' ') for k in probs.columns])
            axp.set_ylim(0, 1)
            axp.axvline(t0, color='0.3', ls=':', lw=1)
            if col == 0:
                axp.set_ylabel('Flag probability')
            axp.legend(loc='upper left', fontsize=8, ncol=5, framealpha=0.8)
        axr = axes[-1, col]
        axr.plot(rain_mean.index, rain_mean.values, color='#4a90d9', lw=0.5, alpha=0.3)
        axr.plot(rain_mean.index, rain_mean.mean(axis=1), color='#1f4e8c', lw=1.8, label='Ensemble mean')
        axr.axvline(t0, color='0.3', ls=':', lw=1)
        axr.grid(alpha=0.25)
        if col == 0:
            axr.set_ylabel('Forecast rain\n(mm/h, mean of gauges)')
        axr.legend(loc='upper right', fontsize=8)
        axr.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        axr.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        plt.setp(axr.xaxis.get_majorticklabels(), rotation=30, ha='right')
    fig.suptitle(f'{location.title()}: 10-day forecast from {t0:%d %b %Y %H:%M} UTC '
                 f'(ECMWF AIFS ensemble rain, {sept.shape[1]} members)', fontsize=14)
    out = OUT_DIR / f'now_{location}_sept_vs_hybrid.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    locations = sys.argv[1:] or ['isis', 'godstow', 'wallingford']
    # All 19 gauges, so the hybrid gets forecast rain at every gauge it trained on.
    rain_all = get_rainfall_forecast_ensemble(location='wallingford', n_members=N_MEMBERS)
    for location in locations:
        print(f'\n=== {location} ===', flush=True)
        merged_df, _, _ = prepare_training_data(location=location, project_root=PROJECT_ROOT, verbose=False)
        t0 = merged_df.index[-1]
        rain_live = get_rainfall_forecast_ensemble(location=location, n_members=N_MEMBERS)
        sept = run_september(location, merged_df, rain_live)
        frame = live_frame(location, merged_df)
        hybrid = run_hybrid(location, frame, rain_all)
        sept.index = pd.DatetimeIndex(sept.index)
        if sept.index.tz is None:
            sept.index = sept.index.tz_localize('UTC')

        stations = [c for c in frame.columns if c not in ('differential', 'flow_m3s_Farmoor')]
        fc = rain_all.copy()
        if fc.index.tz is None:
            fc.index = fc.index.tz_localize('UTC')
        fc = fc.loc[t0:t0 + pd.Timedelta(hours=HORIZON)]
        rain_mean = pd.DataFrame({i: fc[[f'{s}_member_{i}' for s in stations if f'{s}_member_{i}' in fc]].mean(axis=1)
                                  for i in range(N_MEMBERS)})
        past = mean_station_rain(frame).loc[t0 - pd.Timedelta(days=5):t0]
        print(f"t0 {t0}  observed {frame['differential'].iloc[-1]:.3f} m  "
              f"rain last 5 days {past.sum():.1f} mm, forecast mean 10-day {rain_mean.mean(axis=1).sum():.1f} mm", flush=True)
        for label, m in (('September', sept), ('Hybrid', hybrid)):
            med = m.median(axis=1)
            print(f"  {label:9s} median +24h {med.iloc[min(24, len(med) - 1)]:.3f}  +120h {med.iloc[min(120, len(med) - 1)]:.3f}  "
                  f"+240h {med.iloc[-1]:.3f}   range at end {m.iloc[-1].min():.3f}–{m.iloc[-1].max():.3f}")
        both = pd.concat({'september': sept, 'hybrid': hybrid}, axis=1)
        both.to_csv(OUT_DIR / f'now_{location}_members.csv')
        print(f"saved {plot(location, t0, frame['differential'], sept, hybrid, rain_mean)}", flush=True)


if __name__ == '__main__':
    main()
