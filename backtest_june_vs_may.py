"""
Backtest: June 2026 model vs May 2026 (experiment_2026_01) model.

Methodology follows the Substack post "How bad is the model?": set the model
up at a time t0 in the past and feed it the rainfall that ACTUALLY fell
("clairvoyant rainfall"). This removes the random weather-forecast error and
isolates the structural error of the river model itself.

For a grid of t0s (weekly, late 2024 -> early 2026, the validation period of
both models) we compare:
- MAE vs lead time
- the "start jump" (error in the first hours of the forecast)
- recession-rate violations (predicted drops faster than 3 inches/day)
- winter-only subsets (where the May model is worst)

Usage:
    python backtest_june_vs_may.py [location]
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.config import PHYSICAL_CONSTRAINTS  # noqa: E402
from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.pipeline import prepare_training_data  # noqa: E402
from flag_predictor.prediction.forecast import predict_single  # noqa: E402

MAX_RECESSION = PHYSICAL_CONSTRAINTS['max_recession_m_per_day']
LEAD_BINS = [(0, 24), (24, 72), (72, 240)]


def get_merged_df(location: str) -> pd.DataFrame:
    """Prepare (or load cached) merged data for a location."""
    cache = PROJECT_ROOT / 'data' / f'backtest_merged_{location}.pkl'
    if cache.exists():
        print(f"Loading cached merged data: {cache}")
        return pd.read_pickle(cache)
    merged_df, _, _ = prepare_training_data(location=location, project_root=PROJECT_ROOT, verbose=True)
    merged_df.to_pickle(cache)
    return merged_df


def run_clairvoyant_forecast(model, scaler, config, merged_df, t0,
                             predicts_delta, clamp):
    """Run one forecast at t0 using the rainfall that actually fell."""
    rain_cols = [
        c for c in merged_df.columns
        if c != 'differential' and not c.startswith(('flow_m3s_', 'level_m_', 'groundwater_mAOD_'))
    ]
    history = merged_df.loc[:t0]
    future_rain = merged_df.loc[t0:t0 + pd.Timedelta(hours=241), rain_cols].iloc[1:]
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


def main():
    location = sys.argv[1] if len(sys.argv) > 1 else 'isis'
    models_dir = PROJECT_ROOT / 'models'

    models = {}
    for label, name, delta, clamp in [
        ('May 2026', f'experiment_2026_01_{location}', False, False),
        ('June 2026', f'experiment_2026_06_{location}', True, True),
    ]:
        model, scaler, config = load_model(
            model_path=models_dir / f'multihorizon_model_{name}.pth',
            scaler_path=models_dir / f'scaler_{name}.pkl',
            config_path=models_dir / f'config_{name}.pkl',
        )
        models[label] = (model, scaler, config, delta, clamp)

    merged_df = get_merged_df(location)
    actual = merged_df['differential']

    # Weekly t0s through the shared validation period; keep only t0s where
    # we have a full 240h of observations afterwards.
    t0s = pd.date_range('2024-11-01', '2026-01-05', freq='7D', tz=merged_df.index.tz)
    t0s = [merged_df.index[merged_df.index.get_indexer([t], method='nearest')[0]] for t in t0s]
    t0s = sorted(set(
        t for t in t0s
        if actual.loc[t:t + pd.Timedelta(hours=240)].count() > 200
    ))
    print(f"\nBacktesting {len(t0s)} forecast start times: {t0s[0]} -> {t0s[-1]}")

    rows = []
    examples = {}
    for t0 in t0s:
        actual_win = actual.reindex(pd.date_range(t0, periods=241, freq='1h'))
        for label, (model, scaler, config, delta, clamp) in models.items():
            try:
                pred = run_clairvoyant_forecast(model, scaler, config, merged_df, t0, delta, clamp)
            except Exception as e:
                print(f"  {t0} {label}: failed ({e})")
                continue
            err = (pred - actual_win).abs()
            lead = np.arange(241)
            row = {
                't0': t0,
                'model': label,
                'winter': t0.month in (11, 12, 1, 2, 3),
                'mae': err.mean(),
                'jump_6h': err.iloc[1:7].mean(),
            }
            for lo, hi in LEAD_BINS:
                row[f'mae_{lo}_{hi}h'] = err.iloc[lo:hi + 1].mean()
            # Recession violations: hourly drop beyond physical max (+10% tolerance)
            drops = -pred.diff().dropna()
            row['recession_violation_frac'] = (drops > 1.1 * MAX_RECESSION / 24).mean()
            rows.append(row)
            examples.setdefault(t0, {})[label] = pred
        examples[t0]['actual'] = actual_win

    results = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / 'figures' / f'backtest_june_vs_may_{location}.csv'
    out_csv.parent.mkdir(exist_ok=True)
    results.to_csv(out_csv, index=False)

    # ---- Summary ----
    print(f"\n{'=' * 80}\nBACKTEST SUMMARY ({location}, clairvoyant rainfall, {len(t0s)} forecasts)\n{'=' * 80}")
    summary = results.groupby('model').agg(
        mae=('mae', 'mean'),
        mae_0_24h=('mae_0_24h', 'mean'),
        mae_24_72h=('mae_24_72h', 'mean'),
        mae_72_240h=('mae_72_240h', 'mean'),
        start_jump_6h=('jump_6h', 'mean'),
        recession_violations=('recession_violation_frac', 'mean'),
    )
    print(summary.round(4).to_string())
    print("\nWinter only:")
    print(results[results.winter].groupby('model')[['mae', 'jump_6h']].mean().round(4).to_string())

    # ---- Figures ----
    colors = {'May 2026': '#d62728', 'June 2026': '#1f77b4'}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) MAE vs lead time
    ax = axes[0, 0]
    for label in models:
        sub = results[results.model == label]
        if sub.empty:
            continue
        per_lead = []
        for t0 in t0s:
            pred = examples[t0].get(label)
            if pred is None:
                continue
            per_lead.append((pred - examples[t0]['actual']).abs().to_numpy())
        per_lead = np.nanmean(np.vstack(per_lead), axis=0)
        ax.plot(np.arange(241), per_lead, label=label, color=colors[label], lw=2)
    ax.set_xlabel('Lead time (hours)')
    ax.set_ylabel('MAE (m)')
    ax.set_title('Error vs lead time (clairvoyant rainfall)')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) MAE distribution
    ax = axes[0, 1]
    data = [results[results.model == m]['mae'].dropna() for m in models]
    bp = ax.boxplot(data, tick_labels=list(models), patch_artist=True)
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.5)
    ax.set_ylabel('MAE over 240h forecast (m)')
    ax.set_title('Forecast MAE distribution across start times')
    ax.grid(alpha=0.3)

    # (c)+(d) example winter trajectories
    winter_t0s = [t for t in t0s if t.month in (12, 1, 2)]
    show = winter_t0s[:2] if len(winter_t0s) >= 2 else t0s[:2]
    for ax, t0 in zip(axes[1], show):
        actual_win = examples[t0]['actual']
        ax.plot(actual_win.index, actual_win.values, 'k-', lw=2.5, label='Actual')
        for label in models:
            pred = examples[t0].get(label)
            if pred is not None:
                ax.plot(pred.index, pred.values, color=colors[label], lw=1.8,
                        label=label, alpha=0.9)
        ax.set_title(f"Forecast from {t0:%Y-%m-%d}")
        ax.set_ylabel('Differential (m)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle(f'June 2026 vs May 2026 flag predictor - {location} (clairvoyant rainfall backtest)',
                 fontsize=14)
    fig.tight_layout()
    out_png = PROJECT_ROOT / 'figures' / f'backtest_june_vs_may_{location}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {out_png}")
    print(f"Per-forecast metrics saved to: {out_csv}")


if __name__ == '__main__':
    main()
