"""
Stage-1 kill switch: Farmoor hourly decoder vs 10-day flow persistence.

Same weekly t0 grid as the May/June/September differential backtest.
Returns (beats_persistence, summary_dict).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.pipeline import prepare_flow_training_data  # noqa: E402
from flag_predictor.prediction.forecast import predict_flow_hourly  # noqa: E402
from flag_predictor.processing.features import rain_station_columns  # noqa: E402


def run_flow_persistence_backtest(project_root=None):
    project_root = Path(project_root or PROJECT_ROOT)
    models_dir = project_root / 'models'
    figures = project_root / 'figures'
    figures.mkdir(exist_ok=True)

    merged = prepare_flow_training_data(project_root=project_root, verbose=True)
    flow_col = 'flow_m3s_Farmoor'
    rain_cols = rain_station_columns(merged)
    actual = merged[flow_col]

    model, scaler, config = load_model(
        model_path=models_dir / 'multihorizon_model_experiment_2026_09_farmoor.pth',
        scaler_path=models_dir / 'scaler_experiment_2026_09_farmoor.pkl',
        config_path=models_dir / 'config_experiment_2026_09_farmoor.pkl',
    )

    t0s = pd.date_range('2024-11-01', '2026-01-05', freq='7D', tz=merged.index.tz)
    t0s = [merged.index[merged.index.get_indexer([t], method='nearest')[0]] for t in t0s]
    t0s = sorted(set(
        t for t in t0s
        if actual.loc[t:t + pd.Timedelta(hours=240)].count() > 200
    ))
    print(f"Flow backtest on {len(t0s)} starts")

    rows = []
    for t0 in t0s:
        hist = merged.loc[:t0]
        future_rain = merged.loc[t0:t0 + pd.Timedelta(hours=241), rain_cols].iloc[1:]
        actual_win = actual.reindex(pd.date_range(t0, periods=241, freq='1h', tz=merged.index.tz))
        persist = pd.Series(float(actual.loc[t0]), index=actual_win.index)
        try:
            pred = predict_flow_hourly(
                model, scaler, config, hist, future_rain, verbose=False
            )
            pred = pred.reindex(actual_win.index[1:]).astype(float)
            # prepend t0
            pred_full = pd.concat([pd.Series({t0: float(actual.loc[t0])}), pred]).sort_index()
            pred_full = pred_full.reindex(actual_win.index).ffill()
        except Exception as exc:
            print(f"  {t0} failed: {exc}")
            continue
        err_m = (pred_full - actual_win).abs()
        err_p = (persist - actual_win).abs()
        rows.append({
            't0': t0,
            'mae_model_24_240h': err_m.iloc[24:].mean(),
            'mae_persist_24_240h': err_p.iloc[24:].mean(),
            'mae_model': err_m.mean(),
            'mae_persist': err_p.mean(),
        })

    results = pd.DataFrame(rows)
    out_csv = figures / 'backtest_flow_vs_persistence.csv'
    results.to_csv(out_csv, index=False)

    model_far = float(results['mae_model_24_240h'].mean())
    persist_far = float(results['mae_persist_24_240h'].mean())
    beat = model_far < persist_far
    summary = {
        'n': int(len(results)),
        'mae_model_24_240h': model_far,
        'mae_persist_24_240h': persist_far,
        'mae_model': float(results['mae_model'].mean()),
        'mae_persist': float(results['mae_persist'].mean()),
        'beats_persistence': beat,
    }
    print(summary)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        ['Persistence', 'September flow decoder'],
        [persist_far, model_far],
        color=['#d62728', '#1f77b4'],
        alpha=0.8,
    )
    ax.set_ylabel('MAE 24–240 h (m³/s)')
    ax.set_title('Farmoor flow: hourly decoder vs persistence (clairvoyant rain)')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    out_png = figures / 'backtest_flow_vs_persistence.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")
    return beat, summary


def main():
    beat, summary = run_flow_persistence_backtest(PROJECT_ROOT)
    print('beats_persistence', beat, summary)


if __name__ == '__main__':
    main()
