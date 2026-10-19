"""
Lock-keeper step layer for the first days of the forecast.

The differential does not rise and fall smoothly: much of each change comes as
a jump when a keeper adjusts the weir gates, mostly between 08:00 and 16:00
UK time, and mostly when the river is heading somewhere quickly. The hybrid is
trained to get the hourly level right on average, so when it is not sure
whether the jump comes this morning or tomorrow morning it spreads the change
over both, and its flag changes land at any time of day.

This layer turns each forecast outcome into held levels and jumps for the
first three days:

* Every hour, the chance of a jump up (down) comes from a logistic hazard. It
  depends on how far the outcome says the river will move over the next 0, 6,
  12 and 24 hours, the time of day, weekends, the past 24 h trend, rain over the
  past day, hours since the last jump, how far ahead we are, and whether this
  stretch or a neighbouring one (Isis / Godstow / Wallingford) jumped in the
  last three hours.
* A jump's size comes from the same pending moves (keepers often overshoot),
  times a factor drawn from past jumps; between jumps the level drifts by a
  fitted share of the pending move.
* The hazard, sizes and drift were fitted against the hybrid's own forecasts on
  the training years (teacher forcing: each forecast hour starts from the level
  the river was really at), so the layer learns when jumps happen *given what the
  smooth forecast says*, including its habit of being early or late.
* Each outcome keeps its own destination: the borrowed model errors are
  smoothed over a day and stay in the target, so the band's level and spread
  still come from the rain ensemble and the model-error library.

The stepped outcomes give the chance of a flag change today / by the end of
tomorrow, shown in the box above the forecast summary.

Fitting: fit_step_layer.py. Parameters: models/step_layer_<location>.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

LOCAL_TZ = 'Europe/London'
LOCATIONS = ('isis', 'godstow', 'wallingford')
JUMP_THRESHOLD = {'isis': 0.03, 'godstow': 0.03, 'wallingford': 0.04}
NO_FLAGS = {'wallingford'}     # no flags flown: stepped forecast, but no flag-change outlook
LEADS = (0, 6, 12, 24)         # pending move to the outcome's level this many hours after the current hour
SINCE_CAP_H = 72
DAY_START_H, DAY_END_H = 8, 16
TRIGGER_DECAY_H = 3.0          # a jump seen in the last 3 h matters for the next few hours only
RES_SMOOTH_H = 25              # borrowed model errors are smoothed over a day before they set the target
NOISE_H = 5                    # ... and their hour-to-hour wobble (vs a 5 h running median) is added back after
HELD_H = 3                     # a flag change counts once the new flag has held for 3 hours
STEP_HORIZON_H = 96            # stepped for 96 h ...
BLEND_START_H = 72             # ... fading back into the outcome from 72 h to 96 h


# ---------------------------------------------------------------- features

def tod_features(local_hour, dow) -> np.ndarray:
    h = np.asarray(local_hour, dtype=float)
    cols = []
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * h / 24), np.cos(2 * np.pi * k * h / 24)]
    weekend = (np.asarray(dow) >= 5).astype(float)
    day = ((h >= DAY_START_H) & (h < DAY_END_H)).astype(float)
    cols += [weekend, weekend * day]
    return np.stack(cols, axis=-1)


def hazard_design(pending, trend_up, trend_down, rain24, since, local_hour, dow, lead, trigger, own) -> np.ndarray:
    """Hazard inputs for one direction.

    pending   : (..., len(LEADS)) moves still to come in this direction (m, >= 0)
    trend_up/down: rise / fall over the past 24 h (m, >= 0)
    rain24    : mean-gauge rain over the past 24 h (mm)
    since     : hours since this location's last jump in this direction
    lead      : forecast hour (1, 2, ...)
    trigger   : 1 if a neighbouring stretch jumped this way in the 3 h before the forecast start
    own       : 1 if this stretch jumped this way in the 3 h before the forecast start
    """
    pending = np.asarray(pending, dtype=float)
    shape = pending.shape[:-1]

    def b(v):
        return np.broadcast_to(np.asarray(v, dtype=float), shape)

    lh = b(local_hour)
    day = ((lh >= DAY_START_H) & (lh < DAY_END_H)).astype(float)
    lp = np.log1p(pending / 0.02)
    decay = np.exp(-b(lead) / TRIGGER_DECAY_H)
    extra = np.stack([
        b(trend_up) / 0.1, b(trend_down) / 0.1,
        np.log1p(np.clip(b(rain24), 0, None)),
        np.log1p(np.minimum(b(since), SINCE_CAP_H)),
        day * lp[..., 2], day * lp[..., 0],
        np.log1p(b(lead)) / np.log(73.0),
        b(trigger) * decay, b(own) * decay,
    ], axis=-1)
    return np.concatenate([tod_features(lh, b(dow)), pending / 0.1, lp, extra], axis=-1)


def hourly_jumps(level: np.ndarray, thr: float) -> np.ndarray:
    """Signed jump in hour t (from hour t-1 to t) on an hourly series, 0 if none.

    A jump is a change of at least thr in one hour that persists: the median of
    hours t..t+2 differs from the median of hours t-3..t-1 by >= 0.7 thr in the
    same direction. The last two hours of a series can only use what exists,
    as at the start of a live forecast.
    """
    s = pd.Series(np.asarray(level, dtype=float))
    d1 = s.diff().to_numpy()
    after = s[::-1].rolling(3, min_periods=1).median()[::-1].to_numpy()
    before = s.rolling(3, min_periods=2).median().shift(1).to_numpy()
    pj = after - before
    with np.errstate(invalid='ignore'):
        ok = (np.isfinite(d1) & (np.abs(d1) >= thr) & np.isfinite(pj) & (np.abs(pj) >= 0.7 * thr)
              & (np.sign(pj) == np.sign(d1)))
    return np.where(ok, pj, 0.0)


def recent_jump(level_hist: np.ndarray, thr: float, sign: int, hours: int = 3) -> int:
    """1 if the hourly series (ending at the forecast start) jumped this way in its last `hours` hours."""
    v = np.asarray(level_hist, dtype=float)[-(hours + 9):]
    if len(v) < 6 or not np.isfinite(v[-1]):
        return 0
    return int(np.any(sign * hourly_jumps(v, thr)[-hours:] > 0))


def issue_triggers(location: str, level_hist: np.ndarray,
                   neighbour_hists: Optional[Mapping[str, np.ndarray]]) -> tuple:
    """({+1, -1}: neighbour jumped) and ({+1, -1}: this stretch jumped), in the 3 h to the start."""
    trig, own = {}, {}
    for sign in (1, -1):
        own[sign] = recent_jump(level_hist, JUMP_THRESHOLD[location], sign)
        seen = 0
        for other, hist in (neighbour_hists or {}).items():
            if other != location and hist is not None:
                seen = max(seen, recent_jump(hist, JUMP_THRESHOLD.get(other, 0.03), sign))
        trig[sign] = seen
    return trig, own


def hourly_wobble(res: np.ndarray, window: int = NOISE_H) -> np.ndarray:
    """Hour-scale part of each borrowed error: the error minus its centred running median."""
    med = pd.DataFrame(np.asarray(res, dtype=float).T).rolling(window, center=True, min_periods=1).median()
    return np.asarray(res, dtype=float) - med.to_numpy().T


def smooth_errors(res: np.ndarray, window: int = RES_SMOOTH_H) -> np.ndarray:
    """Centred moving average along lead time: keeps a borrowed error's level, drops its step timing."""
    res = np.asarray(res, dtype=float)
    half = window // 2
    pad = np.pad(res, ((0, 0), (half, half)), mode='edge')
    c = np.cumsum(np.pad(pad, ((0, 0), (1, 0))), axis=1)
    return (c[:, window:] - c[:, :-window]) / window


# ---------------------------------------------------------------- parameters

@dataclass
class StepParams:
    location: str
    jump_threshold: float
    coef: Dict[str, np.ndarray]          # 'up' / 'down' hazard coefficients
    intercept: Dict[str, float]
    size_w: Dict[str, np.ndarray]        # jump size = size_b + size_w . pending moves, times a factor
    size_b: Dict[str, float]
    size_q: Dict[str, np.ndarray]        # quantiles (0..1 in 21 steps) of that factor in past jumps
    drift: Dict[str, np.ndarray]         # 'day' / 'night': hourly drift = drift . signed pending moves
    fitted_on: str = ''
    notes: Dict = field(default_factory=dict)

    def to_json(self) -> dict:
        lst = lambda a: [float(x) for x in np.ravel(a)]  # noqa: E731
        return {
            'location': self.location, 'jump_threshold': self.jump_threshold, 'leads_h': list(LEADS),
            'coef': {k: lst(v) for k, v in self.coef.items()},
            'intercept': {k: float(v) for k, v in self.intercept.items()},
            'size_w': {k: lst(v) for k, v in self.size_w.items()},
            'size_b': {k: float(v) for k, v in self.size_b.items()},
            'size_q': {k: lst(v) for k, v in self.size_q.items()},
            'drift': {k: lst(v) for k, v in self.drift.items()},
            'fitted_on': self.fitted_on, 'notes': self.notes,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> 'StepParams':
        j = json.loads(Path(path).read_text())
        arr = lambda d: {k: np.asarray(v, dtype=float) for k, v in d.items()}  # noqa: E731
        return cls(j['location'], float(j['jump_threshold']), arr(j['coef']),
                   {k: float(v) for k, v in j['intercept'].items()}, arr(j['size_w']),
                   {k: float(v) for k, v in j['size_b'].items()}, arr(j['size_q']), arr(j['drift']),
                   j.get('fitted_on', ''), j.get('notes', {}))


def params_path(models_dir: Path, location: str) -> Path:
    return Path(models_dir) / f'step_layer_{location}.json'


# ---------------------------------------------------------------- simulation

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def hours_since_last(jumps: np.ndarray, sign: int) -> float:
    idx = np.where(sign * np.asarray(jumps) > 0)[0]
    return float(SINCE_CAP_H) if len(idx) == 0 else float(min(len(jumps) - 1 - idx[-1], SINCE_CAP_H))


def simulate(params: StepParams, level_hist: np.ndarray, targets: np.ndarray, rain_hist: np.ndarray,
             rain_future: np.ndarray, t0: pd.Timestamp, trigger: Optional[Mapping[int, int]] = None,
             own: Optional[Mapping[int, int]] = None, rng: Optional[np.random.Generator] = None,
             final: Optional[np.ndarray] = None, horizon: int = STEP_HORIZON_H,
             blend_start: int = BLEND_START_H) -> np.ndarray:
    """Stepped version of each outcome.

    level_hist : hourly observed differential ending at t0 (>= 25 values; gaps filled forward)
    targets    : (S, H+1) smooth outcomes the steps follow, index 0 = t0
    rain_hist  : mean-gauge rain (mm/h) for the 24 h ending at t0
    rain_future: (H,) or (S, H) mean-gauge rain for hours 1..H
    trigger/own: {+1: 0/1, -1: 0/1} jumps seen in the 3 h before t0 (issue_triggers)
    final      : (S, H+1) what the result fades back into after `blend_start` (default: targets)
    Returns (S, H+1): stepped to `horizon` hours, fading into `final` from
    `blend_start`, identical to `final` after `horizon`.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(targets, dtype=float)
    fin = y if final is None else np.asarray(final, dtype=float)
    trigger = trigger or {1: 0, -1: 0}
    own = own or {1: 0, -1: 0}
    S, H1 = y.shape
    H = min(horizon, H1 - 1)
    hist = pd.Series(np.asarray(level_hist, dtype=float)).ffill().bfill().to_numpy()
    if len(hist) < 25 or not np.isfinite(hist[-1]):
        return fin.copy()
    jumps_hist = hourly_jumps(hist, params.jump_threshold)
    since = {1: np.full(S, hours_since_last(jumps_hist, +1) + 1),
             -1: np.full(S, hours_since_last(jumps_hist, -1) + 1)}

    rh = np.nan_to_num(np.asarray(rain_hist, dtype=float)[-24:], nan=0.0).clip(min=0)
    rh = np.pad(rh, (24 - len(rh), 0))
    rf = np.nan_to_num(np.asarray(rain_future, dtype=float), nan=0.0).clip(min=0)
    rf = np.broadcast_to(rf if rf.ndim == 2 else rf[None, :], (S, rf.shape[-1]))[:, :H]
    if rf.shape[1] < H:
        rf = np.pad(rf, ((0, 0), (0, H - rf.shape[1])))
    rain = np.concatenate([np.broadcast_to(rh[None, :], (S, 24)), rf], axis=1)   # hour h-1 at index 23+h
    crain = np.concatenate([np.zeros((S, 1)), np.cumsum(rain, axis=1)], axis=1)

    times = pd.date_range(pd.Timestamp(t0), periods=H + 1, freq='1h')
    times = times.tz_localize('UTC') if times.tz is None else times
    local = times.tz_convert(LOCAL_TZ)
    lhour, ldow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    q = np.linspace(0, 1, len(params.size_q['up']))
    lead_idx = np.array(LEADS)

    path = np.empty((S, 25 + H))           # 24 h of history, t0, then the forecast hours
    path[:, :25] = hist[-25:][None, :]
    for h in range(1, H + 1):
        prev = path[:, 23 + h]                                   # level at hour h-1
        pending = y[:, np.minimum(h + lead_idx, H1 - 1)] - prev[:, None]
        trend = prev - path[:, h - 1]                            # change over the 24 h to hour h-1
        rain24 = crain[:, 23 + h] - crain[:, h - 1]              # hours h-24..h-1
        p, size = {}, {}
        for d, sign in (('up', 1), ('down', -1)):
            pend = np.maximum(sign * pending, 0)
            x = hazard_design(pend, np.maximum(trend, 0), np.maximum(-trend, 0), rain24, since[sign],
                              lhour[h], ldow[h], h, trigger[sign], own[sign])
            p[d] = _sigmoid(x @ params.coef[d] + params.intercept[d])
            base = np.maximum(pend @ params.size_w[d] + params.size_b[d], 0.005)
            size[d] = base * np.interp(rng.random(S), q, params.size_q[d])
        u = rng.random(S)
        up = u < p['up']
        dn = ~up & (u < p['up'] + p['down'])
        is_day = DAY_START_H <= lhour[h] < DAY_END_H
        drift = pending @ params.drift['day' if is_day else 'night']
        path[:, 24 + h] = prev + np.where(up, size['up'], 0.0) - np.where(dn, size['down'], 0.0) \
            + np.where(up | dn, 0.0, drift)
        since[1] = np.where(up, 1.0, np.minimum(since[1] + 1, SINCE_CAP_H))
        since[-1] = np.where(dn, 1.0, np.minimum(since[-1] + 1, SINCE_CAP_H))

    out = fin.copy()
    z = path[:, 24:]
    w = np.clip((H - np.arange(H + 1)) / max(H - blend_start, 1), 0, 1)
    out[:, :H + 1] = w * z + (1 - w) * fin[:, :H + 1]
    out[:, 0] = fin[:, 0]
    return out


@dataclass
class SteppedForecast:
    members: np.ndarray                  # (M, H+1) stepped rain members
    total: Optional[np.ndarray]          # (S, H+1) stepped outcomes (members + model error), or None
    outlook: Optional[dict]              # flag_outlook(...) of the outcomes
    triggers: dict


def apply_live(location: str, level_hist: np.ndarray, rain_hist: np.ndarray, members: np.ndarray,
               member_rain: np.ndarray, total: Optional[np.ndarray], t0: pd.Timestamp,
               thresholds: Dict[str, List[float]], models_dir: Path,
               neighbour_hists: Optional[Mapping[str, np.ndarray]] = None,
               seed: int = 0) -> Optional[SteppedForecast]:
    """Step the live forecast. Returns None when the location has no fitted step layer.

    members     : (M, H+1) rain-member trajectories, index 0 = t0
    member_rain : (M, H) each member's mean-gauge forecast rain (mm/h)
    total       : (M * n, H+1) member-major outcomes from model_error.total_uncertainty, or None
    neighbour_hists: other stretches' hourly differentials ending at t0, for the jump triggers
    """
    path = params_path(models_dir, location)
    if not path.exists():
        return None
    params = StepParams.load(path)
    rng = np.random.default_rng(seed)
    members = np.asarray(members, dtype=float)
    member_rain = np.asarray(member_rain, dtype=float)
    trig, own = issue_triggers(location, level_hist, neighbour_hists)
    stepped_members = simulate(params, level_hist, members, rain_hist, member_rain, t0, trig, own, rng)
    stepped_total = None
    if total is not None:
        total = np.asarray(total, dtype=float)
        n = max(total.shape[0] // members.shape[0], 1)
        base = np.repeat(members, n, axis=0)
        res = total[:, 1:] - base[:, 1:]
        targets = base.copy()
        targets[:, 1:] += smooth_errors(res)
        stepped_total = simulate(params, level_hist, targets, rain_hist, np.repeat(member_rain, n, axis=0), t0,
                                 trig, own, rng, final=total)
        # Put back the hour-to-hour wobble of the borrowed errors where the outcome is stepped
        # (the river's hourly readings wobble too, which matters near a flag boundary).
        H = stepped_total.shape[1] - 1
        w = np.clip((STEP_HORIZON_H - np.arange(1, H + 1)) / (STEP_HORIZON_H - BLEND_START_H), 0, 1)
        stepped_total[:, 1:] += hourly_wobble(res) * w
    current = pd.Series(np.asarray(level_hist, dtype=float)).ffill().iloc[-1]
    outlook = None if location in NO_FLAGS else flag_outlook(
        stepped_total if stepped_total is not None else stepped_members, thresholds, t0, float(current))
    return SteppedForecast(stepped_members, stepped_total, outlook,
                           {'neighbour_up': trig[1], 'neighbour_down': trig[-1], 'own_up': own[1], 'own_down': own[-1]})


# ---------------------------------------------------------------- flag outlook

def ordered_flags(thresholds: Dict[str, List[float]]) -> List[tuple]:
    """[(name, lower, upper)] in rising order, dropping zero-width placeholder bands."""
    rows = [(k, float(v[0]), float(v[1])) for k, v in thresholds.items() if float(v[1]) > float(v[0])]
    return sorted(rows, key=lambda r: r[1])


def flag_index(values: np.ndarray, flags: List[tuple]) -> np.ndarray:
    uppers = np.array([f[2] for f in flags[:-1]])
    return np.searchsorted(uppers, np.asarray(values, dtype=float), side='right')


def day_windows(t0: pd.Timestamp, n_hours: int) -> Dict[str, int]:
    """Last forecast hour (inclusive) that is still today / tomorrow in UK time."""
    t0 = pd.Timestamp(t0)
    t0 = t0.tz_localize('UTC') if t0.tz is None else t0
    local = pd.date_range(t0, periods=n_hours + 1, freq='1h').tz_convert(LOCAL_TZ)
    day0 = local[0].normalize()
    dd = np.array([(d - day0).days for d in local.normalize()])
    return {'today': int(np.max(np.where(dd == 0)[0])), 'tomorrow': int(np.max(np.where(dd <= 1)[0]))}


PARTS_OF_DAY = ((6, 9), (9, 12), (12, 15), (15, 18), (18, 22))


def likely_time(first: np.ndarray, local: pd.DatetimeIndex) -> Optional[str]:
    """Most common (day, part of day) among the first-change hours; returns its median time (ISO, UTC).

    A plain median can land between two clusters (this afternoon / tomorrow morning) at night."""
    if len(first) == 0:
        return None
    t = local[np.asarray(first, dtype=int)]
    h = t.hour.to_numpy()
    part = np.full(len(h), 5)                     # 5 = overnight (22:00-06:00)
    for i, (a, b) in enumerate(PARTS_OF_DAY):
        part[(h >= a) & (h < b)] = i
    day = (t - pd.Timedelta(hours=6)).normalize()    # overnight counts with the evening before
    key = pd.Series(list(zip(day, part)))
    best = key.value_counts().idxmax()
    sel = np.asarray(first)[(key == best).to_numpy()]
    return local[int(np.median(sel))].tz_convert('UTC').isoformat()


def held(moved: np.ndarray, hours: int = HELD_H) -> np.ndarray:
    """(..., H) True where the change has started and holds for `hours` hours (or to the end of the forecast)."""
    ok = moved.copy()
    for k in range(1, hours):
        ok[..., :-k] &= moved[..., k:]
    return ok


def flag_outlook(stepped: np.ndarray, thresholds: Dict[str, List[float]], t0: pd.Timestamp,
                 current: float, labels: Optional[Dict[str, str]] = None, hold: int = HELD_H) -> Optional[dict]:
    """Chance that the flag changes (up or down) today and by the end of tomorrow (UK time).

    A change counts when an outcome moves into another flag and stays there for
    `hold` hours, starting within the window. `stepped` (S, H+1) outcomes, index 0 = t0.
    """
    flags = ordered_flags(thresholds)
    if len(flags) < 2 or not np.isfinite(current):
        return None
    labels = labels or {}
    S, H1 = stepped.shape
    win = day_windows(t0, H1 - 1)
    c0 = int(flag_index(np.array([current]), flags)[0])
    cat = flag_index(stepped[:, 1:], flags)                 # (S, H) hours 1..H
    local = pd.date_range(pd.Timestamp(t0), periods=H1, freq='1h')
    local = (local.tz_localize('UTC') if local.tz is None else local).tz_convert(LOCAL_TZ)
    changed = {1: held(cat > c0, hold), -1: held(cat < c0, hold)}
    out = {'current_flag': flags[c0][0], 'held_hours': hold, 'windows': {}}
    for name, last in win.items():
        out['windows'][name] = {
            d: (round(float(np.mean(changed[sign][:, :last].any(axis=1))), 3) if last >= 1 else 0.0)
            for d, sign in (('up', 1), ('down', -1))}
    for direction, sign in (('up', 1), ('down', -1)):
        nxt = c0 + sign
        if not 0 <= nxt < len(flags):
            out[direction] = None
            continue
        seg = changed[sign][:, :win['tomorrow']]
        any_hit = seg.any(axis=1) if seg.shape[1] else np.zeros(S, bool)
        when = None
        if any_hit.any():
            first = seg.argmax(axis=1)[any_hit] + 1
            when = {'median_time': likely_time(first, local)}
        out[direction] = {'flag': flags[nxt][0], 'label': labels.get(flags[nxt][0], flags[nxt][0]), 'when': when}
    return out
