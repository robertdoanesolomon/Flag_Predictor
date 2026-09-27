"""
Run candidate forecasters through the shared evaluation (physics-redesign).

Usage:
    python evaluate_candidates.py isis --window test --candidates sept_full,sept_raw
    python evaluate_candidates.py all --window val --candidates persistence

Per-start rows land in figures/eval/{window}_{location}_{candidate}.csv and the
summary table is appended to figures/eval/summary_{window}.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
warnings.filterwarnings('ignore')

from flag_predictor.evaluation import EvalConfig, evaluate, load_merged, summarise  # noqa: E402
from flag_predictor.models.training import load_model  # noqa: E402
from flag_predictor.prediction.forecast import predict_single  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models'
EVAL_DIR = PROJECT_ROOT / 'figures' / 'eval'


def _load(name: str):
    return load_model(
        model_path=MODELS_DIR / f'multihorizon_model_{name}.pth',
        scaler_path=MODELS_DIR / f'scaler_{name}.pkl',
        config_path=MODELS_DIR / f'config_{name}.pkl',
    )


def persistence(location: str):
    def predict(history, future_rain):
        return np.full(241, float(history['differential'].iloc[-1]))
    return predict


def september(location: str, physics: str = 'full', flow_blend: bool = True,
              name: str = None, flow_name: str = 'experiment_2026_09_farmoor'):
    """September-style hourly decoder. physics: 'full' (live), 'recession' or 'off'."""
    model, scaler, config = _load(name or f'experiment_2026_09_{location}')
    config = {**config, 'physics_mode': physics}
    flow_model = flow_scaler = flow_config = None
    if config.get('uses_predicted_flow'):
        flow_model, flow_scaler, flow_config = _load(flow_name)
        flow_config = {**flow_config, 'flow_blend': flow_blend}
    rec = None if physics == 'off' else config.get('max_recession_m_per_day')

    def predict(history, future_rain):
        out = predict_single(
            model=model, scaler=scaler, historical_df=history,
            rainfall_forecast_df=future_rain,
            feature_columns=config['feature_columns'],
            sequence_length=config['sequence_length'],
            horizons=config.get('horizons'), predicts_delta=True,
            max_recession_m_per_day=rec, verbose=False, model_config=config,
            flow_model=flow_model, flow_scaler=flow_scaler, flow_config=flow_config,
        )
        return out.to_numpy()
    return predict


CANDIDATES = {
    'persistence': persistence,
    'sept_full': lambda loc: september(loc, 'full', True),
    'sept_raw': lambda loc: september(loc, 'off', False),
    'sept_off_blend': lambda loc: september(loc, 'off', True),
    'sept_recession': lambda loc: september(loc, 'recession', True),
}


def resolve(cand: str):
    """Built-in candidates, or 'redesign:<name>' for models/redesign_<name>_<loc>.pt."""
    if cand.startswith('redesign:'):
        from flag_predictor.models.candidates import redesign_predictor
        name = cand.split(':', 1)[1]
        return lambda location: redesign_predictor(name, location)
    return CANDIDATES[cand]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('location')
    parser.add_argument('--window', default='test', choices=['test', 'val'])
    parser.add_argument('--candidates', default='persistence,sept_full,sept_raw')
    parser.add_argument('--no-scaling', action='store_true')
    parser.add_argument('--no-perturb', action='store_true')
    args = parser.parse_args()

    locations = ['isis', 'godstow', 'wallingford'] if args.location == 'all' else [args.location]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for location in locations:
        merged = load_merged(location, PROJECT_ROOT)
        for cand in args.candidates.split(','):
            t = time.time()
            predict = resolve(cand)(location)
            cfg = EvalConfig(
                location=location, window=args.window,
                scaling=not args.no_scaling, perturb=not args.no_perturb,
            )
            traj = {}
            res = evaluate(predict, merged, cfg, trajectories=traj)
            stem = f"{args.window}_{location}_{cand.replace(':', '_')}"
            res.to_csv(EVAL_DIR / f'{stem}.csv', index=False)
            np.savez_compressed(
                EVAL_DIR / f'{stem}_traj.npz',
                t0=np.array([t.value for t in traj['t0']]),
                pred=np.vstack(traj['pred']), actual=np.vstack(traj['actual']),
                rain=np.vstack(traj['rain']),
                zero_rain=np.vstack([z if z is not None else np.full(241, np.nan) for z in traj['zero_rain']]),
            )
            s = summarise(res)
            s['location'] = location
            s['candidate'] = cand
            summaries.append(s)
            print(f"{location:12s} {cand:24s} mae={s['mae']:.4f}  ({time.time() - t:.0f}s)", flush=True)

    table = pd.DataFrame(summaries).set_index(['location', 'candidate'])
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(table.round(4).to_string())
    summary_path = EVAL_DIR / f'summary_{args.window}.csv'
    if summary_path.exists():
        old = pd.read_csv(summary_path).set_index(['location', 'candidate'])
        old = old[~old.index.isin(table.index)]
        table = pd.concat([old, table])
    table.to_csv(summary_path)


if __name__ == '__main__':
    main()
