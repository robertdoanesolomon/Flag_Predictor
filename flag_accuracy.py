"""
Flag-colour accuracy of saved evaluation trajectories (Isis and Godstow).

Usage:
    python flag_accuracy.py --window test --candidates persistence,sept_full,redesign:hybrid_v1

For every forecast start and lead hour where the differential was observed,
compare the flag colour of the forecast with the flag colour of the
observation. Reports the share of hours with the right flag by lead band, and
how often the forecast is off by two or more colours.
Needs figures/eval/{window}_{location}_{candidate}_traj.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.config import get_flag_thresholds  # noqa: E402

EVAL_DIR = PROJECT_ROOT / 'figures' / 'eval'
BANDS = {'1-24h': (1, 24), '25-72h': (25, 72), '73-240h': (73, 240)}


def flag_index(values: np.ndarray, location: str) -> np.ndarray:
    """Ordinal flag class (0 = green upward); zero-width bands are skipped."""
    thresholds = get_flag_thresholds(location)
    edges = sorted({hi for lo, hi in thresholds.values() if np.isfinite(hi) and hi > lo})
    return np.searchsorted(np.asarray(edges), values, side='right')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='test')
    parser.add_argument('--candidates', required=True)
    args = parser.parse_args()

    rows = []
    for location in ['isis', 'godstow']:
        for cand in args.candidates.split(','):
            path = EVAL_DIR / f"{args.window}_{location}_{cand.replace(':', '_')}_traj.npz"
            if not path.exists():
                continue
            z = np.load(path)
            pred, actual = z['pred'], z['actual']
            ok = np.isfinite(actual)
            fp = flag_index(pred, location)
            fa = flag_index(np.nan_to_num(actual), location)
            hit = (fp == fa) & ok
            off2 = (np.abs(fp - fa) >= 2) & ok
            row = {'location': location, 'candidate': cand,
                   'right_flag': hit[:, 1:].sum() / ok[:, 1:].sum(),
                   'off_by_2plus': off2[:, 1:].sum() / ok[:, 1:].sum()}
            for name, (lo, hi) in BANDS.items():
                row[f'right_{name}'] = hit[:, lo:hi + 1].sum() / ok[:, lo:hi + 1].sum()
            rows.append(row)
    table = pd.DataFrame(rows).set_index(['location', 'candidate'])
    print(table.round(3).to_string())
    table.to_csv(EVAL_DIR / f'flag_accuracy_{args.window}.csv')


if __name__ == '__main__':
    main()
