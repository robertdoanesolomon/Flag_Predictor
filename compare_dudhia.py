"""
Flag Predictor against the "Isis Level" page's statistical flag forecast (A. Dudhia,
eodg.atm.ox.ac.uk/user/dudhia/rowing/river.html), rebuilt from its description:

  "past occurrences of the last three days' flags in the last 10 years", the forecast "based on what
  followed"; matches "around the same time of year" first (Seasonal), all year (Annual) when fewer
  than 20; Light Blue (from 2021) folded into Blue.

Rebuilt here on the OURCs flag history (from Dec 2018, so at most ~6 years before the held-out days,
not 10): the daily flag is the flag at 12:00 UTC; the pattern is days D-2, D-1, D; seasonal = within
30 days of the same date in earlier years; the forecast for day D+k is the most common flag on day P+k
over the matches P.

Scored against the official flag at 12:00 UTC on days D+1 .. D+5, for every held-out day D (2023, Nov
2024 - Jan 2026) with no Grey in the pattern or the target, on the merged scale Green / Blue (Light +
Dark) / Amber / Red / Black:

  no change     - the flag on day D
  statistical   - the rebuilt statistical forecast
  ours, dry     - the band of our forecast (from 12:00 on day D) at 12:00 on day D+k, with no rain
                  after the start (the same assumption as that page's groundwater forecast): a floor
  ours, rain    - the same with the rain that actually fell: a ceiling
  ours (flag model), dry / rain - the most likely official flag at 12:00 on day D+k from the site's
                  official-flag model run along that forecast (the flag follows the differential with a
                  lag, which the plain band ignores)
Real forecasts, with the real rain forecast, sit between the dry and rain versions.

    python compare_dudhia.py
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

from flag_predictor.evaluation import load_merged  # noqa: E402
from flag_predictor.models.candidates import ensemble_predictor  # noqa: E402
from flag_predictor.prediction import official_flag as of  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402

MERGED = {'green': 'green', 'light_blue': 'blue', 'dark_blue': 'blue', 'blue': 'blue', 'amber': 'amber',
          'red': 'red', 'black': 'black', 'grey': 'grey'}
RANK = {'green': 0, 'blue': 1, 'amber': 2, 'red': 3, 'black': 4}
LEADS = (1, 2, 3, 4, 5)
HELD = [('2023-01-01', '2023-12-21'), ('2024-11-01', '2026-01-10')]


def daily_flags(loc: str) -> pd.Series:
    h = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{loc}.csv')
    h['start'] = pd.to_datetime(h['start'], utc=True)
    h['end'] = pd.to_datetime(h['end'], utc=True)
    days = pd.date_range(h.start.min().normalize() + pd.Timedelta(hours=12), h.end.max(), freq='1D')
    out = {}
    for d in days:
        row = h[(h.start <= d) & (h.end > d)]
        if len(row):
            out[d] = MERGED.get(row.iloc[-1].flag, 'grey')
    return pd.Series(out)


def band_flag(loc: str, d: float) -> str:
    names = of.order(loc)
    k = int(np.searchsorted(of.lines(loc), d, side='right') - 1)
    return MERGED[names[max(0, min(k, len(names) - 1))]]


def statistical(flags: pd.Series, day: pd.Timestamp):
    """{k: (flag, 'seasonal' | 'annual')} for k in LEADS, or None when the pattern has Grey or a gap."""
    pat = tuple(flags.get(day - pd.Timedelta(days=i)) for i in (2, 1, 0))
    if None in pat or 'grey' in pat:
        return None
    past = flags[(flags.index < day - pd.Timedelta(days=5)) & (flags.index >= day - pd.Timedelta(days=3653))]
    vals = past.to_dict()
    one = pd.Timedelta(days=1)
    matches = [p for p in past.index if (vals.get(p - 2 * one), vals.get(p - one), vals.get(p)) == pat]
    doy = day.dayofyear
    seasonal = [p for p in matches if min(abs(p.dayofyear - doy), 365 - abs(p.dayofyear - doy)) <= 30]
    use, how = (seasonal, 'seasonal') if len(seasonal) >= 20 else (matches, 'annual')
    out = {}
    for k in LEADS:
        outcomes = [vals.get(p + k * one) for p in use]
        outcomes = [o for o in outcomes if o and o != 'grey']
        if not outcomes:
            out[k] = (pat[-1], 'none')
            continue
        cnt = pd.Series(outcomes).value_counts()
        best = cnt[cnt == cnt.max()].index
        out[k] = (min(best, key=lambda f: abs(RANK[f] - RANK[pat[-1]])), how)     # ties: nearer today's flag
    return out


def main() -> None:
    torch.set_num_threads(2)
    report = {}
    for loc in ('isis', 'godstow'):
        flags = daily_flags(loc)
        F = load_merged(loc, ROOT)
        F.index = F.index.tz_localize('UTC') if F.index.tz is None else F.index.tz_convert('UTC')
        stations = [c for c in F.columns if c not in ('differential', 'flow_m3s_Farmoor')]
        predict = ensemble_predictor(HYBRID_MODELS, loc)
        params = of.FlagParams.load(ROOT / 'models' / of.PARAMS_FILE)
        names = of.order(loc)
        hh = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{loc}.csv')
        hh['start'] = pd.to_datetime(hh['start'], utc=True)
        hh['end'] = pd.to_datetime(hh['end'], utc=True)

        def flag_model(day, traj):
            row = hh[(hh.start <= day) & (hh.end > day)]
            if not len(row) or row.iloc[-1].flag not in names:
                return None
            f0 = names.index(row.iloc[-1].flag)
            since = (day - row.iloc[-1].start).total_seconds() / 3600
            lh = F['differential'].loc[day - pd.Timedelta(hours=96):day].to_numpy()
            paths = np.repeat(np.r_[F['differential'].get(day), traj[:120]][None, :], 200, axis=0)
            sim = of.simulate(params, loc, f0, since, lh, paths, day, rng=np.random.default_rng(0), horizon=120)
            return {k: MERGED[names[int(np.bincount(sim[:, 24 * k]).argmax())]] for k in LEADS}

        days = [d for d in flags.index if any(pd.Timestamp(a, tz='UTC') <= d < pd.Timestamp(b, tz='UTC') for a, b in HELD)]
        rows = []
        for day in days:
            if not np.isfinite(F['differential'].get(day, np.nan)):
                continue
            today = flags.get(day)
            if today in (None, 'grey'):
                continue
            hist = F.loc[day - pd.Timedelta(days=30):day]
            fut = pd.date_range(day + pd.Timedelta(hours=1), periods=240, freq='1h')
            rain = F[stations].reindex(fut).fillna(0.0)
            wet = pd.Series(np.asarray(predict(hist, rain), float)[1:], fut)
            dry = pd.Series(np.asarray(predict(hist, rain * 0.0), float)[1:], fut)
            stat = statistical(flags, day)
            if stat is None:
                continue
            fm_dry = flag_model(day, dry.to_numpy())
            fm_wet = flag_model(day, wet.to_numpy())
            if fm_dry is None:
                continue
            for k in LEADS:
                tgt_day = day + pd.Timedelta(days=k)
                truth = flags.get(tgt_day)
                w, d_ = wet.get(tgt_day, np.nan), dry.get(tgt_day, np.nan)
                if truth in (None, 'grey') or not (np.isfinite(w) and np.isfinite(d_)):
                    continue
                st, how = stat[k]
                rows.append({'day': day, 'k': k, 'truth': truth, 'today': today,
                             'no_change': today, 'statistical': st, 'how': how,
                             'ours_dry': band_flag(loc, d_), 'ours_rain': band_flag(loc, w),
                             'flagmodel_dry': fm_dry[k], 'flagmodel_rain': fm_wet[k]})
        df = pd.DataFrame(rows)
        rep = {'days': int(df.day.nunique()), 'by_lead': {}, 'changed': {}}
        print(f'\n=== {loc}: {df.day.nunique()} held-out days; statistical used seasonal matches '
              f'{(df.how == "seasonal").mean():.0%} of the time ===')
        M = ('no_change', 'statistical', 'ours_dry', 'ours_rain', 'flagmodel_dry', 'flagmodel_rain')
        print(' days ahead | right flag: ' + ' / '.join(M) + ' | when the flag changed: ' + ' / '.join(M[1:]) + ' (n)')
        for k in LEADS:
            g = df[df.k == k]
            acc = {m: float((g[m] == g.truth).mean()) for m in M}
            ch = g[g.truth != g.today]
            accc = {m: float((ch[m] == ch.truth).mean()) if len(ch) else float('nan') for m in M[1:]}
            rep['by_lead'][k] = {m: round(v, 3) for m, v in acc.items()} | {'n': int(len(g))}
            rep['changed'][k] = {m: round(v, 3) for m, v in accc.items()} | {'n': int(len(ch))}
            print(f'   {k}        | ' + ' / '.join(f'{acc[m]:.0%}' for m in M) + f'  (n={len(g)}) | '
                  + ' / '.join(f'{accc[m]:.0%}' for m in M[1:]) + f' (n={len(ch)})')
        report[loc] = rep
    out = ROOT / 'figures' / 'eval' / 'compare_dudhia.json'
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
