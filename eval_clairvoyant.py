"""
Fast clairvoyant-rain backtest for one September (or named) model.

Usage:
    python eval_clairvoyant.py isis experiment_2026_09_isis
    python eval_clairvoyant.py isis experiment_2026_09_isis --physics off
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.config import PHYSICAL_CONSTRAINTS  # noqa: E402
from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.prediction.forecast import predict_single  # noqa: E402

from backtest_may_june_sept import get_merged_df, load_optional_flow, run_clairvoyant_forecast  # noqa: E402

MAX_RECESSION = PHYSICAL_CONSTRAINTS['max_recession_m_per_day']
LEAD_BINS = [(0, 24), (24, 72), (72, 240)]


def score_location(location: str, name: str, physics_mode: str = 'full') -> dict:
    models_dir = PROJECT_ROOT / 'models'
    flow_model, flow_scaler, flow_config = load_optional_flow(models_dir)
    model, scaler, config = load_model(
        model_path=models_dir / f'multihorizon_model_{name}.pth',
        scaler_path=models_dir / f'scaler_{name}.pkl',
        config_path=models_dir / f'config_{name}.pkl',
    )
    config = dict(config)
    config['physics_mode'] = physics_mode
    merged_df = get_merged_df(location)
    actual = merged_df['differential']
    t0s = pd.date_range('2024-11-01', '2026-01-05', freq='7D', tz=merged_df.index.tz)
    t0s = [merged_df.index[merged_df.index.get_indexer([t], method='nearest')[0]] for t in t0s]
    t0s = sorted(set(
        t for t in t0s
        if actual.loc[t:t + pd.Timedelta(hours=240)].count() > 200
    ))
    rows = []
    for t0 in t0s:
        actual_win = actual.reindex(pd.date_range(t0, periods=241, freq='1h'))
        try:
            pred = run_clairvoyant_forecast(
                model, scaler, config, merged_df, t0,
                predicts_delta=True,
                clamp=physics_mode != 'off',
                flow_model=flow_model,
                flow_scaler=flow_scaler,
                flow_config=flow_config,
            )
        except Exception as exc:
            print(f'  fail {t0} {exc}')
            continue
        err = (pred.reindex(actual_win.index) - actual_win).abs()
        rain_cols = [
            c for c in merged_df.columns
            if c != 'differential' and not c.startswith(('flow_m3s_', 'level_m_', 'groundwater_mAOD_'))
        ]
        rain24 = merged_df.loc[t0:t0 + pd.Timedelta(hours=24), rain_cols].sum().sum()
        row = {
            't0': str(t0),
            'winter': t0.month in (11, 12, 1, 2, 3),
            'mae': float(err.mean()),
            'jump_6h': float(err.iloc[1:7].mean()),
            'rain_0_24h': float(rain24),
        }
        for lo, hi in LEAD_BINS:
            row[f'mae_{lo}_{hi}h'] = float(err.iloc[lo:hi + 1].mean())
        drops = -pred.diff().dropna()
        row['recession_violation_frac'] = float((drops > 1.1 * MAX_RECESSION / 24).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    dry = df[df['rain_0_24h'] < 5]
    wet = df[df['rain_0_24h'] >= 5]
    summary = {
        'location': location,
        'name': name,
        'physics': physics_mode,
        'n': int(len(df)),
        'mae': float(df['mae'].mean()),
        'mae_0_24h': float(df['mae_0_24h'].mean()),
        'mae_24_72h': float(df['mae_24_72h'].mean()),
        'mae_72_240h': float(df['mae_72_240h'].mean()),
        'jump_6h': float(df['jump_6h'].mean()),
        'recession_violation_frac': float(df['recession_violation_frac'].mean()),
        'winter_mae': float(df.loc[df.winter, 'mae'].mean()) if df.winter.any() else None,
        'dry24_mae_0_24h': float(dry['mae_0_24h'].mean()) if len(dry) else None,
        'wet24_mae_0_24h': float(wet['mae_0_24h'].mean()) if len(wet) else None,
        'n_dry24': int(len(dry)),
        'n_wet24': int(len(wet)),
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('location')
    p.add_argument('name')
    p.add_argument('--physics', default='full', choices=['full', 'recession', 'off'])
    args = p.parse_args()
    summary = score_location(args.location, args.name, args.physics)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
