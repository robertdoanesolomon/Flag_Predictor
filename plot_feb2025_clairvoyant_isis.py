"""
Clairvoyant-rainfall forecast comparison for Isis, February 2025.

For each 00z initialisation in Feb 2025 (28 days), runs a 10-day forecast
using the rainfall that actually fell, for both the May 2026 (old) and
June 2026 (new) models. Plots all trajectories against the observed
differential and catchment rainfall.

Usage:
    python plot_feb2025_clairvoyant_isis.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.config import PHYSICAL_CONSTRAINTS  # noqa: E402
from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.prediction.forecast import predict_single  # noqa: E402

MAX_RECESSION = PHYSICAL_CONSTRAINTS['max_recession_m_per_day']
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'


def get_merged_df() -> pd.DataFrame:
    cache = PROJECT_ROOT / 'data' / 'backtest_merged_isis.pkl'
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing cached merged data: {cache}\n"
            "Run backtest_june_vs_may.py isis first to build it."
        )
    return pd.read_pickle(cache)


def rain_columns(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c != 'differential'
        and not c.startswith(('flow_m3s_', 'level_m_', 'groundwater_mAOD_'))
    ]


def run_clairvoyant(model, scaler, config, merged_df, t0, predicts_delta, clamp):
    cols = rain_columns(merged_df)
    history = merged_df.loc[:t0]
    future_rain = merged_df.loc[t0:t0 + pd.Timedelta(hours=241), cols].iloc[1:]
    return predict_single(
        model=model,
        scaler=scaler,
        historical_df=history,
        rainfall_forecast_df=future_rain,
        feature_columns=config['feature_columns'],
        sequence_length=config['sequence_length'],
        horizons=config['horizons'],
        predicts_delta=predicts_delta,
        max_recession_m_per_day=MAX_RECESSION if clamp else None,
        verbose=False,
    )


def snap_to_index(index: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    pos = index.get_indexer([t], method='nearest')[0]
    return index[pos]


def main():
    merged_df = get_merged_df()
    rain_cols = rain_columns(merged_df)
    tz = merged_df.index.tz

    # 28 midnight (00z) initialisations: 1 Feb – 28 Feb 2025
    raw_t0s = pd.date_range('2025-02-01', '2025-02-28', freq='D', tz=tz)
    t0s = [snap_to_index(merged_df.index, t) for t in raw_t0s]
    t0s = sorted(set(t0s))
    print(f"Running {len(t0s)} clairvoyant forecasts per model...")

    model_specs = [
        ('May 2026 (old)', 'experiment_2026_01_isis', False, False),
        ('June 2026 (new)', 'experiment_2026_06_isis', True, True),
    ]
    models = {}
    for label, name, delta, clamp in model_specs:
        model, scaler, config = load_model(
            model_path=MODELS_DIR / f'multihorizon_model_{name}.pth',
            scaler_path=MODELS_DIR / f'scaler_{name}.pkl',
            config_path=MODELS_DIR / f'config_{name}.pkl',
        )
        forecasts = {}
        for i, t0 in enumerate(t0s, 1):
            print(f"  [{label}] {i}/{len(t0s)}  t0={t0:%Y-%m-%d %H:%M}", flush=True)
            forecasts[t0] = run_clairvoyant(
                model, scaler, config, merged_df, t0, delta, clamp
            )
        models[label] = {'forecasts': forecasts}

    # February window for the main series
    month_start = pd.Timestamp('2025-02-01', tz=tz)
    month_end = pd.Timestamp('2025-03-01', tz=tz)
    actual = merged_df['differential'].loc[month_start:month_end]
    catchment_rain = merged_df[rain_cols].sum(axis=1).loc[month_start:month_end]
    daily_rain = catchment_rain.resample('D').sum()

    fig, (ax_diff, ax_rain) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06},
    )

    ax_diff.plot(
        actual.index, actual.values,
        color='black', linewidth=2.5, label='Actual differential', zorder=10,
    )

    panel_styles = [
        ('May 2026 (old)', '#d62728'),
        ('June 2026 (new)', '#1f77b4'),
    ]
    for label, color in panel_styles:
        forecasts = models[label]['forecasts']
        for pred in forecasts.values():
            pred_feb = pred.loc[month_start:month_end]
            ax_diff.plot(
                pred_feb.index, pred_feb.values,
                color=color, linewidth=0.9, alpha=0.4, zorder=5,
            )
        ax_diff.plot(
            [], [], color=color, linewidth=1.5, alpha=0.8,
            label=f'{label} ({len(forecasts)} × 00z)',
        )

    ax_diff.set_ylabel('Differential (m)')
    ax_diff.set_title(
        'Isis clairvoyant forecasts, February 2025\n'
        '28 × 00z initialisations per model; rainfall = what actually fell',
        fontsize=13,
    )
    ax_diff.legend(loc='upper right', framealpha=0.9)
    ax_diff.grid(True, alpha=0.25)
    ax_diff.set_xlim(month_start, month_end)

    # --- Rainfall panel ---
    bar_width = 0.9
    ax_rain.bar(
        daily_rain.index, daily_rain.values,
        width=bar_width, color='0.65', edgecolor='0.5', linewidth=0.3,
        label='Daily catchment rainfall (actual)',
    )
    ax_rain.set_ylabel('Rain (mm)')
    ax_rain.set_xlabel('Date (UTC)')
    ax_rain.legend(loc='upper right', framealpha=0.9)
    ax_rain.grid(True, alpha=0.25, axis='y')

    ax_rain.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax_rain.xaxis.get_majorticklabels(), rotation=30, ha='right')

    FIGURES_DIR.mkdir(exist_ok=True)
    out_png = FIGURES_DIR / 'feb2025_clairvoyant_isis_may_vs_june.png'
    out_pdf = FIGURES_DIR / 'feb2025_clairvoyant_isis_may_vs_june.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
