"""
Plot saved evaluation trajectories for storms and dry spells.

Usage:
    python plot_candidate_examples.py isis --window test --candidates sept_full,redesign:hybrid_v1

Needs figures/eval/{window}_{location}_{candidate}_traj.npz from
evaluate_candidates.py. Picks the three starts with the biggest observed rise
(storms) and the three with the least forecast-period rain (dry spells).
Also plots each candidate's zero-rain run, which should never climb late.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).parent
EVAL_DIR = PROJECT_ROOT / 'figures' / 'eval'
LABELS = {
    'sept_full': 'September (live, clamped)',
    'sept_raw': 'September (raw output)',
    'persistence': 'Persistence',
    'redesign:lstm_v2f': 'LSTM + predicted flow (candidate A)',
}


def label(cand: str) -> str:
    if cand.startswith('ensemble:'):
        n = len(cand.split(':', 1)[1].split('+'))
        return f'Physics hybrid ({n}-seed ensemble)'
    return LABELS.get(cand, cand.replace('redesign:', ''))


COLORS = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']


def load(window: str, location: str, cand: str):
    path = EVAL_DIR / f"{window}_{location}_{cand.replace(':', '_')}_traj.npz"
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('location')
    parser.add_argument('--window', default='test')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cands = args.candidates.split(',')
    runs = {c: load(args.window, args.location, c) for c in cands}
    base = runs[cands[0]]
    t0 = pd.to_datetime(base['t0'], utc=True)
    actual = base['actual']
    rain = base['rain']

    rise = np.nanmax(actual, axis=1) - actual[:, 0]
    storms = list(np.argsort(-rise)[:3])
    dry = [i for i in np.argsort(rain.sum(axis=1)) if i not in storms][:3]

    fig, axes = plt.subplots(4, 3, figsize=(18, 15), sharex=False)
    hours = np.arange(241)
    panels = [(r, c) for r in (0, 1) for c in range(3)]
    picks = [('Storm', i) for i in storms] + [('Dry spell', i) for i in dry]
    for (row, col), (kind, i) in zip(panels, picks):
        ax = axes[row * 2, col]
        ax.plot(hours, actual[i], 'k-', lw=2.2, label='Observed')
        for (cand, run), color in zip(runs.items(), COLORS):
            ax.plot(hours, run['pred'][i], color=color, lw=1.6, label=label(cand))
        ax.set_title(f"{kind}: forecast from {t0[i]:%Y-%m-%d %H:%M}")
        ax.set_ylabel('Differential (m)')
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.bar(np.arange(1, 241), rain[i], color='#4a90d9', alpha=0.35, width=1.0)
        ax2.set_ylim(0, max(4.0, float(np.nanmax(rain[i])) * 3))
        ax2.set_ylabel('Rain (mm/h)', color='#4a90d9')
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc='best', framealpha=0.85)
        # Zero-rain companion panel
        axz = axes[row * 2 + 1, col]
        for (cand, run), color in zip(runs.items(), COLORS):
            axz.plot(hours, run['zero_rain'][i], color=color, lw=1.4, ls='--', label=label(cand))
        axz.set_title('Same start, all future rain removed', fontsize=9)
        axz.set_xlabel('Hours ahead')
        axz.grid(alpha=0.3)
    fig.suptitle(f'{args.location}: example forecasts ({args.window} window, observed rain)', fontsize=14)
    fig.tight_layout()
    out = Path(args.out) if args.out else EVAL_DIR / f'examples_{args.window}_{args.location}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
