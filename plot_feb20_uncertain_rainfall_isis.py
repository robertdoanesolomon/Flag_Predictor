"""
Feb 20 2025 stress test: high differential + super-uncertain future rainfall.

Initialises the September 2026 Isis model at 00z on 20 Feb 2025 (differential
~0.20 m, just before the late-month rise) and feeds it many synthetic
future-rainfall scenarios — from bone-dry to 1.2× actual — to show how much
ensemble spread the model produces when rainfall uncertainty is large.
June clairvoyant (actual rain) is overlaid for comparison.

Usage:
    python plot_feb20_uncertain_rainfall_isis.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.config import PHYSICAL_CONSTRAINTS, FIXED_THRESHOLDS, FLAG_COLORS  # noqa: E402
from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.prediction.forecast import predict_single  # noqa: E402

MAX_RECESSION = PHYSICAL_CONSTRAINTS['max_recession_m_per_day']
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'
T0 = '2025-02-20'
N_RANDOM = 20
RNG = np.random.default_rng(42)

# Keep synthetic rain in a plausible band around what actually fell (~470 mm
# catchment total over the 10 days from 20 Feb). Worst case ~1.2× actual.
MAX_RAIN_SCALE = 1.2


def get_merged_df() -> pd.DataFrame:
    cache = PROJECT_ROOT / 'data' / 'backtest_merged_isis.pkl'
    if not cache.exists():
        raise FileNotFoundError(f"Missing {cache}; run backtest_june_vs_may.py isis first.")
    return pd.read_pickle(cache)


def rain_columns(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c != 'differential'
        and not c.startswith(('flow_m3s_', 'level_m_', 'groundwater_mAOD_'))
    ]


def snap(index: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    return index[index.get_indexer([t], method='nearest')[0]]


def baseline_future_rain(merged_df: pd.DataFrame, t0: pd.Timestamp, cols: list) -> pd.DataFrame:
    return merged_df.loc[t0:t0 + pd.Timedelta(hours=241), cols].iloc[1:].copy()


def scaled_rain(base: pd.DataFrame, factor: float) -> pd.DataFrame:
    return (base * factor).clip(lower=0)


def random_rain(base: pd.DataFrame, sigma: float, scale: float) -> pd.DataFrame:
    """Mild log-normal noise around a scaled copy of the actual rain."""
    noise = RNG.lognormal(mean=0, sigma=sigma, size=base.shape)
    return pd.DataFrame(
        (base.values * scale * noise).clip(min=0),
        index=base.index,
        columns=base.columns,
    )


def shifted_rain(base: pd.DataFrame, hours: int) -> pd.DataFrame:
    out = base.copy()
    if hours > 0:
        out.iloc[hours:] = base.iloc[:-hours].values
        out.iloc[:hours] = 0
    else:
        h = -hours
        out.iloc[: len(out) - h] = base.iloc[h:].values
        out.iloc[len(out) - h :] = 0
    return out


def build_scenarios(base: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Dry-ish to slightly-wet scenarios; all within ~0–1.2× actual rain."""
    scenarios: list[tuple[str, pd.DataFrame]] = []

    # Dry → about actual (mm totals roughly 0 – 570 for this event)
    for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, MAX_RAIN_SCALE]:
        scenarios.append((f'scale×{f:g}', scaled_rain(base, f)))

    # Moderate random scatter around 0.5×–1.1× actual pattern
    for i in range(N_RANDOM):
        scale = RNG.uniform(0.4, MAX_RAIN_SCALE)
        sigma = RNG.uniform(0.12, 0.35)
        scenarios.append((f'random_{i+1}', random_rain(base, sigma, scale)))

    # Small timing shifts only (±1 day)
    for h in [-24, 12, 24, 36]:
        scenarios.append((f'shift_{h:+d}h', shifted_rain(base, h)))

    return scenarios


def daily_catchment(rain: pd.DataFrame) -> pd.Series:
    return rain.sum(axis=1).resample('D').sum()


def main():
    merged_df = get_merged_df()
    rain_cols = rain_columns(merged_df)
    tz = merged_df.index.tz
    t0 = snap(merged_df.index, pd.Timestamp(T0, tz=tz))

    diff_at_t0 = merged_df.loc[t0, 'differential']
    print(f"t0={t0}  differential={diff_at_t0:.4f} m")

    base = baseline_future_rain(merged_df, t0, rain_cols)
    scenarios = build_scenarios(base)
    print(f"Running {len(scenarios)} rainfall scenarios through September 2026 model...")

    sept_model, sept_scaler, sept_config = load_model(
        model_path=MODELS_DIR / 'multihorizon_model_experiment_2026_09_isis.pth',
        scaler_path=MODELS_DIR / 'scaler_experiment_2026_09_isis.pkl',
        config_path=MODELS_DIR / 'config_experiment_2026_09_isis.pkl',
    )
    flow_model = flow_scaler = flow_config = None
    farmoor = MODELS_DIR / 'multihorizon_model_experiment_2026_09_farmoor.pth'
    if sept_config.get('uses_predicted_flow') and farmoor.exists():
        flow_model, flow_scaler, flow_config = load_model(
            model_path=farmoor,
            scaler_path=MODELS_DIR / 'scaler_experiment_2026_09_farmoor.pkl',
            config_path=MODELS_DIR / 'config_experiment_2026_09_farmoor.pkl',
        )
    history = merged_df.loc[:t0]

    def _run(model, scaler, config, rain_df):
        return predict_single(
            model=model,
            scaler=scaler,
            historical_df=history,
            rainfall_forecast_df=rain_df,
            feature_columns=config['feature_columns'],
            sequence_length=config['sequence_length'],
            horizons=config.get('horizons'),
            predicts_delta=True,
            max_recession_m_per_day=MAX_RECESSION,
            verbose=False,
            model_config=config,
            flow_model=flow_model if config.get('uses_predicted_flow') else None,
            flow_scaler=flow_scaler if config.get('uses_predicted_flow') else None,
            flow_config=flow_config if config.get('uses_predicted_flow') else None,
        )

    forecasts = {}
    rain_totals = []
    for label, rain_df in scenarios:
        pred = _run(sept_model, sept_scaler, sept_config, rain_df)
        forecasts[label] = pred
        rain_totals.append(rain_df.sum().sum())

    # Overlay June clairvoyant-scale envelope cheaply: actual-rain + 0 and 1.2×
    june_model, june_scaler, june_config = load_model(
        model_path=MODELS_DIR / 'multihorizon_model_experiment_2026_06_isis.pth',
        scaler_path=MODELS_DIR / 'scaler_experiment_2026_06_isis.pkl',
        config_path=MODELS_DIR / 'config_experiment_2026_06_isis.pkl',
    )

    # Clairvoyant (actual rain) reference
    actual_rain_pred = _run(sept_model, sept_scaler, sept_config, base)
    june_actual = _run(june_model, june_scaler, june_config, base)

    # Actual differential after t0
    window_end = t0 + pd.Timedelta(hours=240)
    actual_diff = merged_df['differential'].loc[t0:window_end]
    actual_daily_rain = daily_catchment(
        merged_df[rain_cols].loc[t0:window_end]
    )

    # Stats
    ens = pd.DataFrame(forecasts)
    spread_end = ens.iloc[-1].max() - ens.iloc[-1].min()
    print(f"10-day catchment rain across scenarios: {min(rain_totals):.0f} – {max(rain_totals):.0f} mm")
    print(f"Differential spread at +240h: {spread_end:.4f} m  ({ens.iloc[-1].min():.3f} – {ens.iloc[-1].max():.3f})")
    print(f"Mean hourly std: {ens.std(axis=1).mean():.4f} m")

    # --- Plot ---
    fig, (ax_diff, ax_rain) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.07},
    )

    # Flag bands (Isis)
    for key, (lo, hi) in FIXED_THRESHOLDS.items():
        if hi == float('inf'):
            hi = 1.2
        if lo == -float('inf'):
            lo = 0
        ax_diff.axhspan(lo, hi, color=FLAG_COLORS[key], alpha=0.06, zorder=0)

    ax_diff.plot(actual_diff.index, actual_diff.values, color='black', lw=2.8,
                 label='Actual differential', zorder=10)
    ax_diff.axhline(diff_at_t0, color='black', ls=':', lw=1, alpha=0.5)

    for pred in forecasts.values():
        ax_diff.plot(pred.index, pred.values, color='#2ca02c', lw=0.9, alpha=0.35, zorder=5)
    ax_diff.plot([], [], color='#2ca02c', lw=1.5, alpha=0.8,
                 label=f'Synthetic-rain forecasts (n={len(scenarios)})')
    ax_diff.plot(actual_rain_pred.index, actual_rain_pred.values, color='#2ca02c', lw=2.2,
                 ls='--', label='September clairvoyant (actual rain)', zorder=8)
    ax_diff.plot(june_actual.index, june_actual.values, color='#1f77b4', lw=2.0,
                 ls=':', label='June clairvoyant (actual rain)', zorder=8)

    ax_diff.set_ylabel('Differential (m)')
    ax_diff.set_title(
        f'Isis September 2026 model — 20 Feb 2025 00z init (differential = {diff_at_t0:.3f} m)\n'
        f'Uncertain rainfall scenarios (0 – {MAX_RAIN_SCALE:.1f}× actual, n={len(scenarios)}): '
        f'dry → slightly wet + mild random + ±1 day shifts',
        fontsize=12,
    )
    ax_diff.legend(loc='upper left', framealpha=0.92)
    ax_diff.grid(True, alpha=0.25)
    ax_diff.set_xlim(t0, window_end)

    # Rainfall panel: envelope of scenario daily totals + actual
    scenario_daily = pd.DataFrame(
        {label: daily_catchment(rain) for label, rain in scenarios}
    )
    rain_idx = scenario_daily.index
    p10 = scenario_daily.quantile(0.10, axis=1)
    p90 = scenario_daily.quantile(0.90, axis=1)
    p50 = scenario_daily.quantile(0.50, axis=1)

    ax_rain.fill_between(rain_idx, p10, p90, color='#1f77b4', alpha=0.25, label='Scenario P10–P90')
    ax_rain.plot(rain_idx, p50, color='#1f77b4', lw=1.2, alpha=0.7, label='Scenario median')
    ax_rain.bar(actual_daily_rain.index, actual_daily_rain.values, width=0.85,
                color='0.55', edgecolor='0.4', linewidth=0.3, label='Actual daily rain', zorder=5)

    ax_rain.set_ylabel('Catchment rain (mm/day)')
    ax_rain.set_xlabel('Date (UTC)')
    ax_rain.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax_rain.grid(True, alpha=0.25, axis='y')
    ax_rain.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax_rain.xaxis.get_majorticklabels(), rotation=30, ha='right')

    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / 'feb20_uncertain_rainfall_isis_september2026.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
