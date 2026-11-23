"""
Verification only: the official-flag forecast against the flag-band forecast, 20 Jan - 28 Feb 2025,
with the rain that was actually forecast (ECMWF IFS Previous Runs, as in verify_ifs_feb2025.py).

For forecasts issued every 6 hours, the deployed hybrid gives the river outcomes (model-error library
spread, no step layer). From the same outcomes:

  band   most likely flag band at each hour (share of outcomes in each band)
  flag   most likely official flag at each hour (official_flag.simulate along the outcomes,
         started from the flag flying at issue; hourly chances as published, not recalibrated)

Both are scored against the official OURCs flag flying at that hour (Grey hours left out), by lead.
Writes figures/eval/flag_vs_band_feb2025.json (scores, and hourly strips for hours 19-24 ahead).

    python compare_flag_vs_band_feb2025.py
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
import torch  # noqa: E402

from flag_predictor.data.ourcs_flags import flag_series  # noqa: E402
from flag_predictor.models.physics_data import SEQ_LEN, load_location_data  # noqa: E402
from flag_predictor.prediction import official_flag as of  # noqa: E402
from flag_predictor.prediction.hybrid_live import ERROR_LIBRARY, HYBRID_MODELS, MODEL_TAG, TRAINING_GAUGES  # noqa: E402
from flag_predictor.prediction.model_error import ErrorLibrary, start_features, total_uncertainty  # noqa: E402
from verify_ifs_feb2025 import RAIN, ifs_rain, run  # noqa: E402

OUT = ROOT / 'figures' / 'eval' / 'flag_vs_band_feb2025.json'
START, END = pd.Timestamp('2025-01-20', tz='UTC'), pd.Timestamp('2025-02-28', tz='UTC')
LEADS = (3, 6, 12, 24, 48, 72)
H = 72


def main() -> None:
    ifs = pd.read_csv(RAIN, parse_dates=['time'])
    ifs['time'] = ifs.time.dt.tz_localize('UTC')
    torch.set_num_threads(2)
    params = of.FlagParams.load(ROOT / 'models' / of.PARAMS_FILE)
    report = {'period': '20 Jan - 28 Feb 2025, forecasts every 6 h, IFS rain as forecast', 'leads': list(LEADS),
              'locations': {}}
    for loc in ('isis', 'godstow'):
        names = of.order(loc)
        hist = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{loc}.csv', parse_dates=['start', 'end'])
        for c in ('start', 'end'):
            hist[c] = pd.to_datetime(hist[c], utc=True)
        hist['flag'] = hist['flag'].replace({'blue': 'dark_blue'})
        data = load_location_data(loc, ROOT)
        idx = data.index.tz_convert('UTC')
        d = np.asarray(data.diff, float)
        actual = flag_series(hist, idx).to_numpy()
        rank = np.array([names.index(f) if f in names else (len(names) - 1 if f == 'black' else -1)
                         for f in actual])                       # Black scored as Red (top band); Grey -1
        enc_ok = np.isfinite(data.enc).all(axis=1)
        ok = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[:len(enc_ok)] == SEQ_LEN
        pos = np.array([i for i in np.flatnonzero((idx >= START) & (idx < END) & ok & np.isfinite(d))
                        if idx[i].hour % 6 == 0 and rank[i] >= 0])
        rains = np.stack([ifs_rain(ifs, TRAINING_GAUGES[loc], idx[i]) for i in pos])
        fc = run(HYBRID_MODELS, loc, data, pos, rains)
        lib = ErrorLibrary.load(ROOT / 'models' / ERROR_LIBRARY.format(tag=MODEL_TAG, location=loc))
        rows, strip = [], {}
        hist = hist.sort_values('start').reset_index(drop=True)
        for j, i in enumerate(pos):
            t0 = idx[i]
            d24 = d[i - 24] if np.isfinite(d[i - 24]) else np.nan
            feats = start_features(d[i], d24, rains[j], float(np.nansum(data.rain[i - 167:i + 1])), t0)[None, :]
            samples = total_uncertainty(np.r_[d[i], fc[j]][None, :], feats, lib, seed=j)[:, :H + 1]
            band_p = np.stack([(of.band_rank(loc, samples) == k).mean(axis=0) for k in range(len(names))])
            k_set = int(hist['start'].searchsorted(t0, side='right')) - 1
            since = (t0 - hist['start'].iloc[k_set]).total_seconds() / 3600 if k_set >= 0 else 24 * 30
            lvl = pd.Series(d[i - 48:i + 1]).ffill().bfill().to_numpy()
            flags = of.simulate(params, loc, int(rank[i]), max(since, 0.0), lvl, samples, t0,
                                rng=np.random.default_rng(j))
            flag_p = np.stack([(flags == k).mean(axis=0) for k in range(len(names))])
            for L in range(1, H + 1):
                if i + L >= len(rank) or rank[i + L] < 0:
                    continue
                o = int(rank[i + L])
                row = {'lead': L, 'valid': idx[i + L], 'obs': o, 'now': int(rank[i]),
                       'band': int(band_p[:, L].argmax()), 'flag': int(flag_p[:, L].argmax()),
                       'brier_band': float(((band_p[:, L] - np.eye(len(names))[o]) ** 2).sum()),
                       'brier_flag': float(((flag_p[:, L] - np.eye(len(names))[o]) ** 2).sum())}
                rows.append(row)
                if 19 <= L <= 24:
                    strip[idx[i + L]] = (o, row['band'], row['flag'])
        df = pd.DataFrame(rows)
        res = {'n_forecasts': int(len(pos)), 'flags': names, 'by_lead': {}}
        changing = df.obs != df.now
        for L in LEADS:
            g = df[df.lead == L]
            c = changing[g.index]
            res['by_lead'][L] = {
                'right_band': round(float((g.band == g.obs).mean()), 3),
                'right_flag': round(float((g.flag == g.obs).mean()), 3),
                'right_no_change': round(float((g.now == g.obs).mean()), 3),
                'brier_band': round(float(g.brier_band.mean()), 3),
                'brier_flag': round(float(g.brier_flag.mean()), 3),
                'n_changed': int(c.sum()),
                'right_band_changed': round(float((g.band == g.obs)[c].mean()), 3) if c.any() else None,
                'right_flag_changed': round(float((g.flag == g.obs)[c].mean()), 3) if c.any() else None,
            }
        ts = sorted(strip)
        res['strip'] = {'times': [t.strftime('%Y-%m-%dT%H:%M') for t in ts],
                        'actual': [strip[t][0] for t in ts], 'band': [strip[t][1] for t in ts],
                        'flag': [strip[t][2] for t in ts],
                        'river': [round(float(v), 3) for v in pd.Series(d, idx).reindex(ts).to_numpy()]}
        res['lines'] = [round(float(v), 3) for v in of.lines(loc)[1:]]
        report['locations'][loc] = res
        print(f'\n=== {loc}: {len(pos)} forecasts')
        print(pd.DataFrame(res['by_lead']).T.to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f'\nwritten {OUT}')


if __name__ == '__main__':
    main()
