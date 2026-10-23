"""
Backtest: the deployed hybrid (hybrid_c1_ps, 5 copies) against the same set-up with soil moisture
and upstream levels as extra inputs (hybrid_su, train_extra_models.py), on the same starts.

Starts every 6 h in the 2023 validation year and the Nov 2024 - Jan 2026 test window (neither
used for fitting), rain that fell. Reports mean absolute error by lead time, the share of hours
in the right flag band (Isis, Godstow), and the situations where the model struggles most.

    python compare_extra_models.py [--new hybrid_su] [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from build_error_library import predict_hybrid, start_positions  # noqa: E402
from flag_predictor.config import get_flag_thresholds  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, load_location_data  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402
from train_extra_models import data_with_extra  # noqa: E402

LEADS = {'1-24h': (0, 24), '25-72h': (24, 72), '73-240h': (72, 240), 'all': (0, 240)}


def names_for(base: str, seeds: int):
    return [base] + [f'{base}_s{i}' for i in range(1, seeds)]


def band(x: np.ndarray, edges) -> np.ndarray:
    return np.digitize(x, edges)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--new', default='hybrid_su')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--locations', default='isis,godstow,wallingford')
    ap.add_argument('--inputs', default='all', choices=['none', 'all', 'soil', 'soil_rise', 'soil_rise_han'])
    ap.add_argument('--out', default=str(ROOT / 'figures' / 'eval' / 'extra_inputs_comparison.json'))
    args = ap.parse_args()
    torch.set_num_threads(2)
    report = {}
    for loc in args.locations.split(','):
        new_names = [n for n in names_for(args.new, args.seeds) if (ROOT / 'models' / f'redesign_{n}_{loc}.pt').exists()]
        if not new_names:
            print(f'{loc}: no {args.new} models yet')
            continue
        base_data = load_location_data(loc, ROOT)
        starts = start_positions(base_data)
        t_all = np.concatenate([starts['val2023'], starts['test']])
        obs = np.stack([base_data.diff[t + 1:t + HORIZON + 1] for t in t_all])
        old = predict_hybrid(HYBRID_MODELS[:len(new_names)], loc, base_data, t_all)
        new = predict_hybrid(new_names, loc, data_with_extra(loc, args.inputs), t_all)
        d0 = base_data.diff[t_all]
        rain3 = np.array([base_data.rain[t + 1:t + 73].sum() for t in t_all])
        trend = d0 - np.array([base_data.diff[t - 24] for t in t_all])
        test = np.isin(t_all, starts['test'])
        th = get_flag_thresholds(loc)
        edges = sorted({hi for lo, hi in th.values() if np.isfinite(hi) and hi > -1e8})
        mid = (np.percentile(d0, 60), np.percentile(d0, 95))
        subsets = {
            'all': np.ones(len(t_all), bool),
            'test window only': test,
            'heavy rain ahead (15 mm+ in 3 days)': rain3 >= 15,
            'rising fast (top quarter of 24 h rise)': trend >= np.nanpercentile(trend, 75),
            'mid-range and rising': (d0 >= mid[0]) & (d0 < mid[1]) & (trend > 0),
        }
        rep = {'copies': len(new_names), 'starts': int(len(t_all))}
        print(f'\n=== {loc}: {len(t_all)} starts, {len(new_names)} copies each (old vs new, MAE m) ===')
        for sname, m in subsets.items():
            row = {}
            for lname, (a, b) in LEADS.items():
                eo = np.nanmean(np.abs(old[m, a:b] - obs[m, a:b]))
                en = np.nanmean(np.abs(new[m, a:b] - obs[m, a:b]))
                row[lname] = [round(float(eo), 4), round(float(en), 4)]
            if loc in ('isis', 'godstow') and edges:
                ok = np.isfinite(obs[m])
                bo = band(obs[m], edges)
                row['right band'] = [round(float((band(old[m], edges) == bo)[ok].mean()), 4),
                                     round(float((band(new[m], edges) == bo)[ok].mean()), 4)]
            rep[sname] = dict(row, n=int(m.sum()))
            cells = '  '.join(f"{k} {v[0]:.4f}->{v[1]:.4f}" if k != 'right band' else f"band {v[0]:.1%}->{v[1]:.1%}"
                              for k, v in row.items())
            print(f'  {sname:42s} n={int(m.sum()):5d}  {cells}')
        report[loc] = rep
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f'\nwritten {args.out}')


if __name__ == '__main__':
    main()
