"""
The official OURCs flag: what is flying now, whether it matches the
differential band, and when it is likely to change.

The flags are set by people (the Isis by counting bucks at Iffley weir,
Godstow from the lock differential), so they follow the bands with lags:

* going up, they usually change within an hour or two of the band, in daylight;
* coming down, they usually wait until the river is a few centimetres below
  the line and has stayed there for a few daylight hours (Godstow's Red often
  stays until about 0.6 m);
* changes almost always happen between 08:00 and 20:00 UK time.

An hourly logistic hazard for a change up and for a change down, fitted on the
OURCs flag history against the 15-minute differential (fit_official_flag.py,
models/official_flag.json), is run along every forecast outcome from the flag
flying now. A change up goes to the band the river is in (it can skip a flag);
a change down goes one flag down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LOCAL_TZ = 'Europe/London'
# Flag order and the lower line of each flag (m). Black is not modelled: it depends on
# how high the river is (the differential saturates near 0.95 in big floods).
LINES: Dict[str, Dict[str, float]] = {
    'isis': {'green': -9.0, 'light_blue': 0.215, 'dark_blue': 0.33, 'amber': 0.44, 'red': 0.535},
    'godstow': {'green': -9.0, 'amber': 0.45, 'red': 0.75},
}
RUN_CAP_H = 72
# Black (Isis) is not a differential band: the differential saturates near 0.95 in big
# floods. On the Iffley Lock upstream stage (EA 1501TH, m) the official flag since 2019 has
# gone Black once the level reached about 0.42 m and come back to Red once it fell below
# about 0.22 m; that rule matches the flag in 98% of Red/Black hours (fit_official_flag.py).
BLACK_UP_M = 0.42
BLACK_DOWN_M = 0.22
SINCE_CAP_H = 24 * 30
FEATURES = ['day', 'mid', 'weekend', 'margin', 'margin_above', 'log_run', 'day_log_run', 'log_since', 'trend6',
            'isis', 'isis_margin', 'isis_margin_above']
PARAMS_FILE = 'official_flag.json'
# A forecast that starts longer ago than this (gauges offline) gets no official-flag outlook
# or sync note: both would compare an old differential with the flag flying now. Same 4 h
# as the page's "stale" label. An Iffley reading older than this gets no Black note.
STALE_AFTER_H = 4.0
OURCS_URL = 'https://ourcs.co.uk/information/flags/'


def order(river: str) -> List[str]:
    return list(LINES[river])


def lines(river: str) -> np.ndarray:
    return np.array(list(LINES[river].values()))


def band_rank(river: str, values) -> np.ndarray:
    return np.searchsorted(lines(river)[1:], np.asarray(values, dtype=float), side='right')


def design(margin, run, local_hour, dow, since, trend6, river: str) -> np.ndarray:
    """Hazard inputs. margin (m): for a change up, how far the river is above the next line up;
    for a change down, how far below the flag's own lower line (negative = not there yet).
    run: hours the river has been past that line; since: hours since the flag was set."""
    margin = np.asarray(margin, dtype=float)
    shp = margin.shape

    def b(v):
        return np.broadcast_to(np.asarray(v, dtype=float), shp)

    h = b(local_hour)
    day = ((h >= 8) & (h < 20)).astype(float)
    mid = ((h >= 10) & (h < 17)).astype(float)
    wk = (b(dow) >= 5).astype(float)
    m = np.clip(margin, -0.3, 0.3) / 0.05
    mpos = np.maximum(np.clip(margin, -0.3, 0.3), 0) / 0.05
    lrun = np.log1p(np.minimum(b(run), RUN_CAP_H))
    lsince = np.log1p(np.minimum(b(since), SINCE_CAP_H))
    tr = np.clip(np.nan_to_num(b(trend6)), -0.2, 0.2) / 0.05
    isis = np.full(shp, float(river == 'isis'))
    return np.stack([day, mid, wk, m, mpos, lrun, day * lrun, lsince, tr, isis, isis * m, isis * mpos], axis=-1)


@dataclass
class FlagParams:
    coef: Dict[str, np.ndarray]
    intercept: Dict[str, float]
    notes: Dict

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            'features': FEATURES, 'lines': LINES,
            'coef': {k: [float(x) for x in v] for k, v in self.coef.items()},
            'intercept': {k: float(v) for k, v in self.intercept.items()}, 'notes': self.notes}, indent=1))

    @classmethod
    def load(cls, path: Path) -> 'FlagParams':
        j = json.loads(Path(path).read_text())
        return cls({k: np.asarray(v, float) for k, v in j['coef'].items()},
                   {k: float(v) for k, v in j['intercept'].items()}, j.get('notes', {}))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _run_length(past: np.ndarray) -> float:
    """Hours at the end of a boolean series that are all True."""
    n = 0
    for v in past[::-1]:
        if not v:
            break
        n += 1
    return float(min(n, RUN_CAP_H))


def simulate(params: FlagParams, river: str, flag0: int, since0: float, level_hist: np.ndarray,
             paths: np.ndarray, t0: pd.Timestamp, rng: Optional[np.random.Generator] = None,
             horizon: int = 72) -> np.ndarray:
    """Official flag (rank) along each outcome. paths (S, H+1) hourly differential, index 0 = t0.
    level_hist: hourly observed differential ending at t0 (for how long the river has been past a line)."""
    rng = rng or np.random.default_rng(0)
    L = lines(river)
    top = len(L) - 1
    S, H1 = paths.shape
    H = min(horizon, H1 - 1)
    hist = pd.Series(np.asarray(level_hist, dtype=float)).ffill().to_numpy()
    flag = np.full(S, int(flag0))
    since = np.full(S, float(since0))
    # run lengths at t0 for the current flag, from the observed history
    r_up0 = _run_length(hist > L[min(flag0 + 1, top)]) if flag0 < top else 0.0
    r_dn0 = _run_length(hist < L[flag0]) if flag0 > 0 else 0.0
    run_up = np.full(S, r_up0)
    run_dn = np.full(S, r_dn0)
    local = pd.date_range(pd.Timestamp(t0), periods=H + 1, freq='1h')
    local = (local.tz_localize('UTC') if local.tz is None else local).tz_convert(LOCAL_TZ)
    lh, dw = local.hour.to_numpy(), local.dayofweek.to_numpy()

    def level_at(k):                  # differential k hours after t0 (k may be negative)
        if k >= 0:
            return paths[:, k]
        return np.full(S, hist[len(hist) - 1 + k] if len(hist) - 1 + k >= 0 else hist[0])

    out = np.empty((S, H + 1), dtype=int)
    out[:, 0] = flag
    for h in range(1, H + 1):
        k = h - 1                       # the decision for hour h is taken on hour h-1 (as in the fit)
        d = level_at(k)
        trend = d - level_at(k - 6)
        top_ok = flag < top
        m_up = d - L[np.minimum(flag + 1, top)]
        m_dn = L[flag] - d
        if k > 0:                        # at k = 0 the runs come from the observed history
            run_up = np.where(top_ok & (m_up > 0), np.minimum(run_up + 1, RUN_CAP_H), 0.0)
            run_dn = np.where((flag > 0) & (m_dn > 0), np.minimum(run_dn + 1, RUN_CAP_H), 0.0)
        pu = _sigmoid(design(m_up, run_up, lh[k], dw[k], since, trend, river) @ params.coef['up'] + params.intercept['up'])
        pd_ = _sigmoid(design(m_dn, run_dn, lh[k], dw[k], since, trend, river) @ params.coef['down'] + params.intercept['down'])
        pu = np.where(top_ok, pu, 0.0)
        pd_ = np.where(flag > 0, pd_, 0.0)
        u = rng.random(S)
        up = u < pu
        dn = ~up & (u < pu + pd_)
        band = band_rank(river, d)
        new = np.clip(np.where(up, np.maximum(flag + 1, band), np.where(dn, flag - 1, flag)), 0, top)
        changed = new != flag
        flag = new
        since = np.where(changed, 0.0, since + 1)
        run_up = np.where(changed, 0.0, run_up)
        run_dn = np.where(changed, 0.0, run_dn)
        out[:, h] = flag
    return out


def day_windows(t0: pd.Timestamp, n_hours: int) -> Dict[str, int]:
    from .step_layer import day_windows as dw
    return dw(t0, n_hours)


def outlook(river: str, flags: np.ndarray, flag0: int, t0: pd.Timestamp) -> dict:
    """Chance the official flag has changed by the end of today / tomorrow, where to, and when."""
    names = order(river)
    S, H1 = flags.shape
    win = day_windows(t0, H1 - 1)
    local = pd.date_range(pd.Timestamp(t0), periods=H1, freq='1h')
    local = (local.tz_localize('UTC') if local.tz is None else local).tz_convert(LOCAL_TZ)
    res = {'current_flag': names[flag0], 'windows': {}}
    for name, last in win.items():
        seg = flags[:, 1:last + 1]
        res['windows'][name] = {
            'up': round(float(np.mean((seg > flag0).any(axis=1))), 3) if last >= 1 else 0.0,
            'down': round(float(np.mean((seg < flag0).any(axis=1))), 3) if last >= 1 else 0.0}
    for direction, sign in (('up', 1), ('down', -1)):
        nxt = flag0 + sign
        if not 0 <= nxt < len(names):
            res[direction] = None
            continue
        seg = flags[:, 1:win['tomorrow'] + 1]
        moved = (seg > flag0) if sign > 0 else (seg < flag0)
        hit = moved.any(axis=1) if seg.shape[1] else np.zeros(S, bool)
        when, to = None, names[nxt]
        if hit.any():
            first = moved.argmax(axis=1)[hit]
            from .step_layer import likely_time
            when = {'median_time': likely_time(first + 1, local)}
            dest = seg[hit, :][np.arange(hit.sum()), first]
            vals, cnt = np.unique(dest, return_counts=True)
            to = names[int(vals[np.argmax(cnt)])]
        res[direction] = {'flag': to, 'when': when}
    return res


# ------------------------------------------------------------------ live

_STATUS: Dict[str, Optional[dict]] = {}


def current_status(river: str) -> Optional[dict]:
    if river not in _STATUS:
        from ..data.ourcs_flags import fetch_status
        _STATUS[river] = fetch_status(river)
    return _STATUS[river]


def sync_note(river: str, flag: str, level_hist: np.ndarray) -> dict:
    """Is the official flag where the band is? (and for how long it has not been)."""
    names = order(river)
    hist = pd.Series(np.asarray(level_hist, dtype=float)).ffill().to_numpy()
    if flag not in names or not np.isfinite(hist[-1]):
        return {'in_sync': None}
    b = band_rank(river, hist)
    r = names.index(flag)
    if b[-1] == r:
        return {'in_sync': True, 'band': names[b[-1]]}
    off = 0
    for v in b[::-1]:
        if v == r:
            break
        off += 1
    return {'in_sync': False, 'band': names[int(b[-1])], 'hours_out_of_sync': int(off) if off < len(b) else None,
            'band_above_flag': bool(b[-1] > r)}


def black_indicator(iu: Optional[pd.Series]) -> Optional[dict]:
    """Black-flag state from the Iffley upstream stage (hysteresis: up at BLACK_UP_M, down below BLACK_DOWN_M)."""
    if iu is None or len(iu.dropna()) == 0:
        return None
    v = iu.dropna().sort_index()
    state = False
    for x in v.resample('1h').mean().dropna().to_numpy():
        if not state and x >= BLACK_UP_M:
            state = True
        elif state and x < BLACK_DOWN_M:
            state = False
    return {'level_m': round(float(v.iloc[-1]), 3), 'at': v.index[-1].isoformat(), 'black': bool(state),
            'up_m': BLACK_UP_M, 'down_m': BLACK_DOWN_M,
            'near': bool(not state and v.iloc[-1] >= BLACK_UP_M - 0.08)}


def official_flag_payload(location: str, live, models_dir: Optional[Path] = None, seed: int = 0,
                          now: Optional[pd.Timestamp] = None) -> dict:
    """Preview-page fields: {'official_flag': {...}, 'black': {...}, 'official_outlook': {...}}, each
    only when it can be worked out. ``now`` (default: the clock) decides whether the forecast or
    the Iffley reading is too old to compare with the flag flying now."""
    if location not in LINES:
        return {}
    status = current_status(location)
    if not status:
        return {}
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    out = {'official_flag': {**status, 'url': OURCS_URL}}
    if location == 'isis':
        try:
            from .hybrid_live import iffley_upstream_level
            b = black_indicator(iffley_upstream_level())
            if b and (now - pd.Timestamp(b['at'])).total_seconds() / 3600 <= STALE_AFTER_H:
                out['black'] = b
        except Exception as exc:
            print(f"  [official flag] black indicator skipped ({exc})")
    members = getattr(live, 'members', None)
    if members is None or len(members) == 0:
        return out
    t0 = pd.Timestamp(members.index[0])
    t0 = t0.tz_localize('UTC') if t0.tz is None else t0
    if (now - t0).total_seconds() / 3600 > STALE_AFTER_H:
        out['official_flag']['forecast_stale'] = True
        print(f"  [official flag] forecast starts {t0} (gauges offline?); no sync note or outlook")
        return out
    flag = status['flag']
    names = order(location)
    hist = getattr(live, 'level_hist', None)
    if hist is not None:
        out['official_flag']['sync'] = sync_note(location, flag, hist)
    if flag not in names:            # grey (not monitored) or a flag this model does not know
        return out
    models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[3] / 'models'
    pfile = models_dir / PARAMS_FILE
    paths = live.stepped_total if live.stepped_total is not None else live.total_samples
    if not pfile.exists() or paths is None or hist is None:
        return out
    params = FlagParams.load(pfile)
    set_at = pd.Timestamp(status['set_at']) if status.get('set_at') else None
    since = (t0 - set_at).total_seconds() / 3600 if set_at is not None else 24 * 30
    flags = simulate(params, location, names.index(flag), max(since, 0.0), hist, np.asarray(paths), t0,
                     rng=np.random.default_rng(seed))
    out['official_outlook'] = outlook(location, flags, names.index(flag), t0)
    return out
