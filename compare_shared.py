"""
Shared catchment model against the deployed per-stretch hybrid, on the deployed model's own
backtest starts (every 6 h over 2023 and Nov 2024 - Jan 2026, the rain that fell).

The deployed forecasts come from its error libraries (observed - forecast, stored per start), so
both are scored on exactly the same starts and observations.

    python compare_shared.py hybrid_shared hybrid_shared_s1 ...    # copies are averaged
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flag_predictor.models.shared_catchment import (STRETCHES, JointSampler, batch_forecast,  # noqa: E402
                                                    build_joint_data, load)

BINS = [(1, 24), (25, 72), (73, 168), (169, 240)]


def deployed(loc: str):
    z = np.load(ROOT / 'models' / f'error_library_hybrid_c1_ps_{loc}.npz')
    t0 = pd.to_datetime(z['t0'].astype('int64'), unit='s', utc=True)
    return t0, z['residuals'].astype(float)


def main(names):
    data = build_joint_data(ROOT)
    pos = pd.Series(np.arange(len(data.index)), index=data.index)
    preds = {}
    for n in names:
        for loc in STRETCHES:
            model, scaler, meta = load(n, ROOT / 'models', loc)
            sampler = JointSampler(data, scaler, 'cpu')
            t0, _ = deployed(loc)
            idx = pos.reindex(t0).dropna().astype(int).to_numpy()
            preds.setdefault(loc, []).append((idx, batch_forecast(model, sampler, idx)))
    report = {}
    for j, loc in enumerate(STRETCHES):
        t0, R = deployed(loc)
        idx = pos.reindex(t0).to_numpy()
        keep = np.isfinite(idx)
        t0, R, idx = t0[keep], R[keep], idx[keep].astype(int)
        shared = np.mean([p[1][:, :, j] for p in preds[loc]], axis=0)
        fut = idx[:, None] + np.arange(1, 241)[None, :]
        y = data.diff[fut, j]
        d0 = data.diff[idx, j]
        dep = y - R
        ok = np.isfinite(y) & np.isfinite(d0)[:, None] & data.stretch_ok[idx, j][:, None]
        e_dep = np.where(ok, np.abs(dep - y), np.nan)
        e_sh = np.where(ok, np.abs(shared - y), np.nan)
        test = np.asarray(t0 >= pd.Timestamp('2024-11-01', tz='UTC'))
        res = {}
        for name, m in (('2023', ~test), ('test', test), ('all', np.ones_like(test))):
            row = {'n_starts': int(m.sum()),
                   'deployed': round(float(np.nanmean(e_dep[m])), 4), 'shared': round(float(np.nanmean(e_sh[m])), 4)}
            for a, b in BINS:
                row[f'{a}-{b}h'] = (round(float(np.nanmean(e_dep[m, a - 1:b])), 4),
                                    round(float(np.nanmean(e_sh[m, a - 1:b])), 4))
            res[name] = row
        # Rises: the river up 0.12 m+ within 7 days of the start; error of the 7-day peak height.
        rise_dep, rise_sh = [], []
        for i in range(len(idx)):
            if not ok[i, :168].sum() > 150:
                continue
            obs_pk = np.nanmax(y[i, :168])
            if obs_pk - d0[i] >= 0.12:
                rise_dep.append(np.nanmax(dep[i, :168]) - obs_pk)
                rise_sh.append(np.nanmax(shared[i, :168]) - obs_pk)
        res['rises'] = {'n': len(rise_dep),
                        'deployed_bias': round(float(np.mean(rise_dep)), 4), 'shared_bias': round(float(np.mean(rise_sh)), 4),
                        'deployed_mae': round(float(np.mean(np.abs(rise_dep))), 4),
                        'shared_mae': round(float(np.mean(np.abs(rise_sh))), 4)}
        report[loc] = res
        report.setdefault('_fc', {})[loc] = (t0, d0, y, dep, shared, ok)
    return data, report


def consistency(report):
    """Godstow-minus-Isis 10-day rise: forecast against observed, on starts both have."""
    fc = report['_fc']
    tg, ti = fc['godstow'][0], fc['isis'][0]
    common = tg.intersection(ti)
    gi = pd.Series(np.arange(len(tg)), index=tg).reindex(common).to_numpy()
    ii = pd.Series(np.arange(len(ti)), index=ti).reindex(common).to_numpy()
    out = {}
    for h in (72, 168, 238):
        g, i = fc['godstow'], fc['isis']
        ob = (g[2][gi, h] - g[1][gi]) - (i[2][ii, h] - i[1][ii])
        dp = (g[3][gi, h] - g[1][gi]) - (i[3][ii, h] - i[1][ii])
        sh = (g[4][gi, h] - g[1][gi]) - (i[4][ii, h] - i[1][ii])
        m = np.isfinite(ob) & np.isfinite(dp) & np.isfinite(sh)
        out[f'{h + 1}h'] = {'n': int(m.sum()),
                            'deployed_gap_error_mae': round(float(np.mean(np.abs(dp[m] - ob[m]))), 4),
                            'shared_gap_error_mae': round(float(np.mean(np.abs(sh[m] - ob[m]))), 4),
                            'corr_deployed': round(float(np.corrcoef(dp[m], ob[m])[0, 1]), 3),
                            'corr_shared': round(float(np.corrcoef(sh[m], ob[m])[0, 1]), 3)}
    return out


if __name__ == '__main__':
    names = sys.argv[1:] or ['hybrid_shared']
    data, report = main(names)
    report['consistency_godstow_minus_isis'] = consistency(report)
    fc = report.pop('_fc')
    # November 2025: the first big rise after the dry autumn (starts 8-11 Nov).
    nov = {}
    for loc in STRETCHES:
        t0, d0, y, dep, sh, ok = fc[loc]
        m = np.asarray((t0 >= pd.Timestamp('2025-11-08', tz='UTC')) & (t0 <= pd.Timestamp('2025-11-11', tz='UTC')))
        nov[loc] = {'starts': int(m.sum()),
                    '7d_rise_observed': round(float(np.nanmean(y[m, 167] - d0[m])), 3),
                    '7d_rise_deployed': round(float(np.nanmean(dep[m, 167] - d0[m])), 3),
                    '7d_rise_shared': round(float(np.nanmean(sh[m, 167] - d0[m])), 3)}
    report['nov2025'] = nov
    print(json.dumps(report, indent=1))
    out = ROOT / 'figures' / 'eval' / 'shared_catchment_compare.json'
    out.write_text(json.dumps({'models': names, **report}, indent=1))
