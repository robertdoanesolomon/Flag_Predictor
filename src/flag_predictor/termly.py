"""
Termly outlook: how many days at each flag this term and next, and the chance of Torpids at a
cancellation-level river, from how wet the catchment already is.

Everything runs on Farmoor's daily mean flow (EA gauge 1100TH, record from June 1992):

  flags      the flag follows Farmoor flow closely. On the 2-day mean flow, the Isis lines are
             23 / 34.5 / 39.5 m3/s (Green | Blue | Amber | Red; Light and Dark Blue merged, Black
             counted as Red) and Godstow's 28 / 41.5 m3/s (Green | Amber | Red). Fitted on every day
             since 2017: the OURCs noon flag in term, the differential's flag band when no flag flies
             (vacations, Grey). Right on 91% (Isis) / 92% (Godstow) of days.
  Torpids    since 1993, every Torpids week (Wed-Sat, actual dates) averaging over 42 m3/s at Farmoor
             was cancelled (2000, 2002, 2007, 2014, 2020, 2024, 2026) or lost days (1994, 1995); the one
             exception is 2025, raced in full at 48 m3/s. Below 42 every one raced in full.
  outlook    an ensemble of one flow path per past year (33 members): each past year's weather from
             today's date on, started from today's catchment state. In normal scores (each day's flow
             ranked among all years' flows at that time of year; see Outlook), every path is that
             year's departure from normal with the part explained by its own starting state swapped
             for today's: z(t) = c(t) . x_now + [z_y(t) - c(t) . x_y], with x the mean score over the
             last 7 and 90 days and c(t) regressed on the past years for each day ahead (start dates
             within 15 days pooled). The 90-day mean carries the groundwater. With c = 0 it is plain
             climatology: every past year equally likely, whatever the river is doing now.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from typing import Optional
import numpy as np
import pandas as pd

FLOW_FILE = Path('data') / 'termly' / 'farmoor_daily.csv'
LINES = {'isis': (23.0, 34.5, 39.5), 'godstow': (28.0, 41.5)}
NAMES = {'isis': ('green', 'blue', 'amber', 'red'), 'godstow': ('green', 'amber', 'red')}
TORPIDS_FLOW = 42.0
PREDICTORS = (7, 90)          # mean normal score of flow over the last 7 and 90 days
POOL_DAYS = 15                # start dates within this many days of today's date pooled in the fit
POOL_STEP = 3
FIRST_YEAR = 1993             # first academic year (Sep - Aug) with a full record before and after
HORIZON = 300                 # days ahead
MIN_FLOW = 0.3                # m3/s: floor before taking logs (summer lows read near zero)
TERMS = ('Michaelmas', 'Hilary', 'Trinity')
PERIODS = ('Michaelmas', 'Christmas Vac', 'Hilary', 'Easter Vac', 'Trinity')
# The differential's flag bands (m), standing in for the flag when OURCs flies none (vacations, Grey).
BANDS = {'isis': ((0.215, 0.33, 0.44, 0.535), ('green', 'blue', 'blue', 'amber', 'red')),
         'godstow': ((0.45, 0.75), ('green', 'amber', 'red'))}
# How much to trust today's catchment state against an average year, by how many days ahead the
# period starts: the forecast is a mix of the conditioned members (weight w) and the plain
# climatology members (1 - w). Chosen on the leave-one-year-out hindcasts (verify_termly.py), where
# the best mix fell from about 0.9 during a term to about 0 for terms more than four months off.
BLEND = ((0, 0.9), (92, 0.8), (123, 0.5), (10 ** 6, 0.2))


# ---------------------------------------------------------------- data

def load_flow(root: Path) -> pd.Series:
    """Farmoor daily mean flow (m3/s), every day, short gaps filled (linear in log flow)."""
    q = pd.read_csv(Path(root) / FLOW_FILE, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    q = q[~q.index.duplicated(keep='last')].sort_index().asfreq('D')
    lq = np.log(q.clip(lower=MIN_FLOW)).interpolate(limit_area='inside')
    return np.exp(lq).rename('flow')


def daily_means(flow_15min: pd.Series, min_readings: int = 72) -> pd.Series:
    """Daily (UTC) means of 15-minute flow, for days with at least ``min_readings`` of 96."""
    s = flow_15min.dropna()
    if s.index.tz is not None:
        s.index = s.index.tz_convert('UTC').tz_localize(None)
    g = s.groupby(s.index.floor('D'))
    out = g.mean()[g.count() >= min_readings]
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return out[out.index < today]


def update_flow(root: Path, verbose: bool = True) -> pd.Series:
    """Add the latest days to the stored record (EA hydrology + flood-monitoring feeds); return it."""
    from flag_predictor.data.api import fetch_flow_data
    path = Path(root) / FLOW_FILE
    old = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    df = fetch_flow_data(verbose=verbose, min_days_back=60)
    if not df.empty and 'flow_m3s_Farmoor' in df:
        new = daily_means(df['flow_m3s_Farmoor'].astype(float))
        if len(new):
            merged = pd.concat([old[old.index < new.index.min()], new]).sort_index()
            merged.index.name = 'date'
            merged.round(3).rename('flow_m3s').to_csv(path)
            if verbose:
                print(f'  Farmoor daily flow: {len(new)} recent days, record to {merged.index.max():%Y-%m-%d}')
    return load_flow(root)


def ourcs_flags(root: Path, loc: str) -> pd.Series:
    """The OURCs flag at 12:00 UTC each day (merged scale; Grey and missing days left out)."""
    merge = {'green': 'green', 'light_blue': 'blue', 'dark_blue': 'blue', 'blue': 'blue', 'amber': 'amber',
             'red': 'red', 'black': 'red'}
    if loc == 'godstow':
        merge['blue'] = merge['light_blue'] = merge['dark_blue'] = 'amber'
    h = pd.read_csv(Path(root) / 'data' / 'flags' / f'history_{loc}.csv')
    h['start'] = pd.to_datetime(h['start'], utc=True)
    h['end'] = pd.to_datetime(h['end'], utc=True).fillna(pd.Timestamp.now(tz='UTC'))
    starts = h.start.dt.tz_convert(None).to_numpy()
    ends = h.end.dt.tz_convert(None).to_numpy()
    flags = h.flag.to_numpy()
    days = pd.date_range(pd.Timestamp(starts.min()).normalize(), pd.Timestamp(ends.max()), freq='D')
    out = {}
    for d in days:
        noon = (d + pd.Timedelta(hours=12)).to_datetime64()
        k = np.flatnonzero((starts <= noon) & (ends > noon))
        if len(k) and flags[k[-1]] in merge:
            out[d] = merge[flags[k[-1]]]
    return pd.Series(out, dtype=object)


def differential_noon(root: Path, loc: str) -> pd.Series:
    """The differential (m) around noon each day: the stored record plus the recent readings."""
    s = pd.read_csv(Path(root) / 'data' / 'termly' / f'differential_noon_{loc}.csv', index_col=0,
                    parse_dates=True).iloc[:, 0]
    recent = Path(root) / 'data' / 'flags' / f'differential_after_csv_{loc}.csv.gz'
    if recent.exists():
        a = pd.read_csv(recent, parse_dates=['timestamp']).set_index('timestamp')['differential']
        a.index = pd.to_datetime(a.index, utc=True).tz_convert(None)
        a = a[(a.index.hour >= 11) & (a.index.hour <= 13)]
        s = pd.concat([s, a.groupby(a.index.normalize()).median()])
        s = s[~s.index.duplicated(keep='last')]
    return s.sort_index()


def daily_flags(root: Path, loc: str) -> pd.Series:
    """The flag each day: the OURCs flag when one is flying, otherwise (vacations, Grey) the
    differential's flag band at noon."""
    f = ourcs_flags(root, loc)
    d = differential_noon(root, loc).dropna()
    lines, names = BANDS[loc]
    band = pd.Series(np.array(names)[np.searchsorted(lines, d.to_numpy(), side='right')], d.index)
    return f.combine_first(band).sort_index()


# ---------------------------------------------------------------- flags from flow

def flag_ranks(q: np.ndarray, loc: str) -> np.ndarray:
    """Flag index (into NAMES[loc]) for each day of daily flow ``q`` (..., days), on the 2-day mean."""
    q = np.asarray(q, float)
    q2 = np.concatenate([q[..., :1], 0.5 * (q[..., 1:] + q[..., :-1])], axis=-1)
    return np.searchsorted(LINES[loc], q2.ravel(), side='right').reshape(q2.shape)


def reconstructed_flags(q: pd.Series, loc: str) -> pd.Series:
    r = flag_ranks(q.to_numpy(), loc)
    return pd.Series(np.array(NAMES[loc])[r], q.index)


# ---------------------------------------------------------------- terms

def _sunday_on_or_after(d: dt.date) -> pd.Timestamp:
    t = pd.Timestamp(d)
    return t + pd.Timedelta(days=(6 - t.dayofweek) % 7)


def term_start(term: str, acad: int) -> pd.Timestamp:
    """Sunday of 1st week of Full Term, academic year ``acad`` (Michaelmas ``acad`` - Trinity ``acad``+1).
    Michaelmas: first Sunday on or after 8 Oct; Hilary: on or after 13 Jan; Trinity: on or after 21 Apr,
    a week later when that is Easter Day."""
    if term == 'Michaelmas':
        return _sunday_on_or_after(dt.date(acad, 10, 8))
    if term == 'Hilary':
        return _sunday_on_or_after(dt.date(acad + 1, 1, 13))
    from dateutil.easter import easter
    s = _sunday_on_or_after(dt.date(acad + 1, 4, 21))
    return s + pd.Timedelta(days=7) if s.date() == easter(acad + 1) else s


def term_window(term: str, acad: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A term: Sunday of 0th week to Saturday of 8th week (63 days). A vacation: the days between."""
    if term == 'Christmas Vac':
        return term_window('Michaelmas', acad)[1] + pd.Timedelta(days=1), term_window('Hilary', acad)[0] - pd.Timedelta(days=1)
    if term == 'Easter Vac':
        return term_window('Hilary', acad)[1] + pd.Timedelta(days=1), term_window('Trinity', acad)[0] - pd.Timedelta(days=1)
    s = term_start(term, acad)
    return s - pd.Timedelta(days=7), s + pd.Timedelta(days=55)


# First day (Wednesday) of Torpids. It moves between 6th and 7th week of Hilary (it ends on the Saturday that
# is not the Women's Head of the River). From OURCs (2024-2027) and the dates in A. Dudhia's bumps pages
# (eodg.atm.ox.ac.uk/user/dudhia/rowing/bumps/); 2009 unclear there (7th week assumed). Not held in 2021.
TORPIDS_START = {
    1993: '1993-02-24', 1994: '1994-02-23', 1995: '1995-02-22', 1996: '1996-02-28', 1997: '1997-03-05',
    1998: '1998-03-04', 1999: '1999-03-03', 2000: '2000-03-01', 2001: '2001-02-28', 2002: '2002-02-27',
    2003: '2003-03-05', 2004: '2004-03-03', 2005: '2005-02-23', 2006: '2006-03-01', 2007: '2007-02-28',
    2008: '2008-02-20', 2009: '2009-03-04', 2010: '2010-03-03', 2011: '2011-03-02', 2012: '2012-02-22',
    2013: '2013-02-27', 2014: '2014-03-05', 2015: '2015-03-04', 2016: '2016-02-24', 2017: '2017-03-01',
    2018: '2018-02-28', 2019: '2019-02-27', 2020: '2020-02-26', 2022: '2022-03-02', 2023: '2023-02-22',
    2024: '2024-02-28', 2025: '2025-02-26', 2026: '2026-03-04', 2027: '2027-02-24',
}


def race_days(event: str, acad: int) -> pd.DatetimeIndex:
    """Torpids: Wednesday - Saturday (published dates, else 7th week of Hilary); Eights: 5th week of Trinity."""
    if event == 'torpids' and acad + 1 in TORPIDS_START:
        s = pd.Timestamp(TORPIDS_START[acad + 1])
    elif event == 'torpids':
        s = term_start('Hilary', acad) + pd.Timedelta(days=45)
    else:
        s = term_start('Trinity', acad) + pd.Timedelta(days=31)
    return pd.date_range(s, periods=4, freq='D')


def academic_year(t: pd.Timestamp) -> int:
    return t.year if t.month >= 9 else t.year - 1


# ---------------------------------------------------------------- the ensemble

def _same_date(t0: pd.Timestamp, years_back: int) -> pd.Timestamp:
    try:
        return t0.replace(year=t0.year - years_back)
    except ValueError:                         # 29 Feb
        return t0.replace(year=t0.year - years_back, day=28)


@dataclass
class Ensemble:
    t0: pd.Timestamp                # the issue day (flow known up to and including this day)
    days: pd.DatetimeIndex          # t0 + 1 ... t0 + HORIZON
    members: np.ndarray             # (n, HORIZON) daily flow, m3/s: conditioned on today's state
    climatology: np.ndarray         # (n, HORIZON) the same years without the conditioning
    years: list                     # the academic year each member's weather comes from
    state: dict                     # today's normal scores, mean over the last 7 and 90 days
    year_w: Optional[np.ndarray] = None   # weight of each year from the weather forecasts (sums to 1); None = equal


class Outlook:
    """The ensemble for any issue day, from the record with academic year ``exclude`` left out
    entirely (seasonal distribution, regression and members), for honest hindcasts.

    Works in normal scores: each day's flow is ranked among every year's flows within 15 days of
    that date and turned into a standard normal value (0 = the median for the time of year, -1 = drier
    than 84% of years, ...). Regressing and adding back the weather in this space, then mapping back
    through that date's distribution, keeps every member inside what the river has done at that time
    of year, and lets a very dry catchment wet up as fast as the past years did."""

    def __init__(self, q: pd.Series, exclude: int | None = None, predictors=PREDICTORS, window: int = 15):
        self.q = q
        self.exclude = exclude
        self.predictors = tuple(predictors)
        lq = np.log(q.clip(lower=MIN_FLOW))
        v = lq.to_numpy()
        acad = np.where(lq.index.month >= 9, lq.index.year, lq.index.year - 1)
        doy = np.minimum(lq.index.dayofyear, 365)
        keep = np.isfinite(v) & ((acad != exclude) if exclude is not None else True)
        self.samples = {}
        z = np.full(len(v), np.nan)
        for d in range(1, 366):
            near = np.abs(((doy - d + 182) % 365) - 182) <= window
            s_ = np.sort(v[keep & near])
            self.samples[d] = s_
            m = (doy == d) & np.isfinite(v)
            rank = 0.5 * (np.searchsorted(s_, v[m], 'left') + np.searchsorted(s_, v[m], 'right'))
            z[m] = _ndtri(np.clip((rank + 0.5) / (len(s_) + 1), 1e-3, 1 - 1e-3))
        self.z = pd.Series(z, lq.index)
        self.x = pd.DataFrame({w: self.z.rolling(w, min_periods=int(0.8 * w)).mean() for w in self.predictors})
        self._t_first = lq.index[0]
        self._a = z
        self._x = self.x.to_numpy()

    def normal_on(self, days: pd.DatetimeIndex) -> np.ndarray:
        """Median flow (m3/s) for each date's time of year."""
        return np.array([np.exp(np.median(self.samples[d])) for d in np.minimum(days.dayofyear, 365)])

    def to_flow(self, zz: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        out = np.empty_like(zz, dtype=float)
        p = _ndtr(zz)
        for j, d in enumerate(np.minimum(days.dayofyear, 365)):
            s_ = self.samples[d]
            out[:, j] = np.interp(p[:, j], (np.arange(len(s_)) + 1) / (len(s_) + 1), s_)
        return np.exp(out)

    def _pos(self, t: pd.Timestamp) -> int:
        return (t - self._t_first).days

    def _state(self, t: pd.Timestamp) -> np.ndarray:
        i = self._pos(t)
        return self._x[i] if 0 <= i < len(self._x) else np.full(len(self.predictors), np.nan)

    def _path(self, t: pd.Timestamp) -> np.ndarray | None:
        i = self._pos(t) + 1
        if i < 0 or i + HORIZON > len(self._a):
            return None
        a = self._a[i:i + HORIZON]
        if np.isnan(a).mean() > 0.05:
            return None
        return np.where(np.isnan(a), 0.0, a)

    def z_trend(self, before: pd.Timestamp | None = None) -> float:
        """Linear trend of the academic-year mean normal score (per year), from full years before ``before``."""
        z = self.z if before is None else self.z[self.z.index < before]
        acad = np.where(z.index.month >= 9, z.index.year, z.index.year - 1)
        g = z.groupby(acad).agg(['mean', 'count'])
        g = g[(g['count'] > 330) & (g.index >= FIRST_YEAR)]
        return float(np.polyfit(g.index, g['mean'], 1)[0]) if len(g) > 5 else 0.0

    def ensemble(self, t0: pd.Timestamp, trend: float = 0.0) -> Ensemble:
        """``trend``: normal-score change per year; each member is moved by trend x (this year - its year)."""
        t0 = pd.Timestamp(t0).normalize()
        acad0 = academic_year(t0)
        x_now = self._state(t0)
        if not np.isfinite(x_now).all():
            raise ValueError(f'no catchment state on {t0:%Y-%m-%d}')
        X, R, own, yrs = [], [], [], []
        for y in range(FIRST_YEAR, acad0 + 40):
            if y == acad0 or y == self.exclude:
                continue
            base = _same_date(t0, acad0 - y)
            for dd in range(-POOL_DAYS, POOL_DAYS + 1, POOL_STEP):
                t = base + pd.Timedelta(days=dd)
                xv = self._state(t)
                path = self._path(t)
                if path is None or not np.isfinite(xv).all():
                    continue
                X.append(xv)
                R.append(path)
                own.append(dd == 0)
                yrs.append(y)
        X, R, own = np.array(X), np.array(R), np.array(own)
        C = np.linalg.lstsq(X, R, rcond=None)[0]                       # (n_predictors, HORIZON)
        cond = x_now @ C + (R - X @ C)[own]
        clim = R[own]
        if trend:
            shift = trend * (acad0 - np.array([y for y, o in zip(yrs, own) if o]))[:, None]
            cond, clim = cond + shift, clim + shift
        days = pd.date_range(t0 + pd.Timedelta(days=1), periods=HORIZON)
        return Ensemble(t0=t0, days=days, members=self.to_flow(cond, days), climatology=self.to_flow(clim, days),
                        years=[y for y, o in zip(yrs, own) if o],
                        state={f'{w}d': round(float(v), 3) for w, v in zip(self.predictors, x_now)})


def _ndtr(x):
    from scipy.special import ndtr
    return ndtr(x)


def _ndtri(p):
    from scipy.special import ndtri
    return ndtri(p)


# ---------------------------------------------------------------- what the ensemble says

def blend_weight(lead_days: float) -> float:
    for limit, w in BLEND:
        if lead_days <= limit:
            return w
    return BLEND[-1][1]


def mix(ens: 'Ensemble', lead_days: float) -> tuple[np.ndarray, np.ndarray]:
    """(samples, weights): the conditioned and climatology members together, weighted for this lead."""
    w = blend_weight(lead_days)
    n, m = len(ens.members), len(ens.climatology)
    yw = np.full(n, 1 / n)
    if ens.year_w is not None:      # the forecast weights cover three weeks: fade them out for periods further off
        a = float(np.clip(1 - (lead_days - 21) / 49, 0, 1))
        yw = a * ens.year_w + (1 - a) * yw
    return (np.concatenate([ens.members, ens.climatology]), np.r_[w * yw, (1 - w) * yw])


def wquantile(x: np.ndarray, w: np.ndarray, p: float) -> float:
    """Weighted percentile ``p`` (0-100) of samples ``x``."""
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(p / 100 * w.sum(), c, x))


def wcrps(x: np.ndarray, w: np.ndarray, obs: float) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float(np.sum(w * np.abs(x - obs)) - 0.5 * np.sum(w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :])))


def days_at_flags(flow: np.ndarray, days: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp,
                  loc: str, prev_flow: float | None = None) -> np.ndarray:
    """(n_members, n_flags): days at each flag between start and end (inclusive) for each member."""
    sel = (days >= start) & (days <= end)
    f = np.atleast_2d(flow)
    if prev_flow is not None:                  # the 2-day mean on the first day needs the day before
        f = np.concatenate([np.full((len(f), 1), prev_flow), f], axis=1)
        r = flag_ranks(f, loc)[:, 1:]
    else:
        r = flag_ranks(f, loc)
    r = r[:, sel]
    return np.stack([(r == k).sum(1) for k in range(len(NAMES[loc]))], axis=1)


def race_flow(flow: np.ndarray, days: pd.DatetimeIndex, race: pd.DatetimeIndex) -> np.ndarray:
    sel = days.isin(race)
    return np.atleast_2d(flow)[:, sel].mean(1)


def crps(ens: np.ndarray, obs: float) -> float:
    ens = np.asarray(ens, float)
    return float(np.mean(np.abs(ens - obs)) - 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :])))


# ---------------------------------------------------------------- seasonal rainfall forecast

OBS_RAIN_FILE = Path('data') / 'termly' / 'cru_monthly_rain.csv'   # CRU TS 4.10 (build_cru_rain.py)
SEAS5_FILE = Path('data') / 'seasonal' / 'seas5_upper_thames.csv'


def seasonal_rain(root: Path, today: pd.Timestamp | None = None) -> list[dict]:
    """ECMWF SEAS5 (latest monthly run from the C3S archive, 25 members): rain over the catchment above
    Farmoor for each month still to come, corrected for the model's bias by quantile mapping. Each member's
    month total is ranked among the model's own forecasts for that start month and lead in every other year
    since 1993, and replaced by the same percentile of observed (CRU TS) rainfall for that month."""
    today = pd.Timestamp.utcnow().tz_localize(None) if today is None else pd.Timestamp(today)
    s = pd.read_csv(Path(root) / SEAS5_FILE)
    if 'centre' in s:
        s = s[s.centre == 'ecmwf']
    era = pd.read_csv(Path(root) / OBS_RAIN_FILE, index_col=0, parse_dates=True).iloc[:, 0]
    era = era[era.index.year >= FIRST_YEAR]
    latest = max(i for i in s.init.unique() if pd.Period(i, 'M') <= today.to_period('M'))
    init = pd.Period(latest, 'M')
    out = []
    for lead, g in s[s.init == latest].groupby('lead'):
        valid = init + int(lead) - 1
        if valid.start_time <= today:
            continue
        clim = s[(s.init.str[5:] == latest[5:]) & (s.lead == lead) & (s.init.str[:4] != latest[:4])].rain_mm.to_numpy()
        obs = era[era.index.month == valid.month].to_numpy()
        pct = np.array([(clim < v).mean() + 0.5 * (clim == v).mean() for v in g.rain_mm])
        mapped = np.quantile(obs, np.clip(pct, 0, 1))
        lo, hi = np.quantile(obs, [1 / 3, 2 / 3])
        out.append({'month': str(valid), 'lead': int(lead), 'run': latest,
                    'median': round(float(np.median(mapped))), 'p10': round(float(np.percentile(mapped, 10))),
                    'p90': round(float(np.percentile(mapped, 90))), 'normal': round(float(np.median(obs))),
                    'wet': round(float((mapped > hi).mean()), 2), 'dry': round(float((mapped < lo).mean()), 2)})
    return out


# ---------------------------------------------------------------- extended-range (EC46) weekly rainfall

EC46_POINTS = ((51.95, -1.95), (51.70, -1.90), (51.65, -1.60), (51.85, -1.50))   # across the catchment


def ec46_fetch(timeout: int = 60) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The latest EC46 run via Open-Meteo: (days, rain members x days), mm/day, mean over the catchment points."""
    import requests
    r = requests.get('https://seasonal-api.open-meteo.com/v1/seasonal', timeout=timeout, params={
        'latitude': ','.join(str(p[0]) for p in EC46_POINTS), 'longitude': ','.join(str(p[1]) for p in EC46_POINTS),
        'daily': 'precipitation_sum', 'models': 'ecmwf_ec46', 'forecast_days': 46})
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    keys = [k for k in js[0]['daily'] if k.startswith('precipitation_sum')]
    days = pd.to_datetime(js[0]['daily']['time'])
    rain = np.nanmean([[np.array(j['daily'][k], dtype=float) for k in keys] for j in js], axis=0)   # members x days
    ok = np.isfinite(rain).all(axis=0)
    return days[ok], rain[:, ok]


# How well EC46's ensemble-mean rain over 7 days matches what fell (correlation), for days 1-7, 8-14 and 15-21
# after the issue day. Provisional: typical ECMWF extended-range skill for weekly UK rainfall, rounded down;
# to be refitted on the EC46 re-forecasts (fetch_ec46.py) when they arrive. Beyond three weeks: no skill.
EC46_SKILL = (0.6, 0.3, 0.15)
EC46_WEIGHTING = True           # provisional skill; refit on the EC46 re-forecasts when they arrive
ERA5_DAILY = Path('data') / 'termly' / 'era5_daily_rain.csv'
MIN_EFFECTIVE_YEARS = 15


def forecast_year_weights(ens: 'Ensemble', root: Path, days: pd.DatetimeIndex, rain: np.ndarray) -> tuple[np.ndarray, dict]:
    """Weight each member year by how well its rain (ERA5) over the next three weeks fits the EC46 forecast.

    For each week ahead: the forecast's ensemble-mean total is ranked among that week's totals in every year since
    1993 (a normal score z_f), and so is each member year's own total (z_y). With skill r for that week, the
    weight is the likelihood ratio p(z_y | forecast) / p(z_y) = N(z_y; r z_f, 1 - r^2) / N(z_y; 0, 1), multiplied
    over the weeks. Never fewer than MIN_EFFECTIVE_YEARS years' worth of weight (flattened if needed)."""
    from scipy.special import ndtri
    obs = pd.read_csv(Path(root) / ERA5_DAILY, index_col=0, parse_dates=True).iloc[:, 0]
    acad0 = academic_year(ens.t0)
    years = np.array(ens.years)
    lw, weeks = np.zeros(len(years)), []
    start = max(days[0], ens.t0 + pd.Timedelta(days=1))
    for k, r in enumerate(EC46_SKILL):
        a = start + pd.Timedelta(days=7 * k)
        sel = (days >= a) & (days < a + pd.Timedelta(days=7))
        if sel.sum() < 7 or r <= 0:
            break
        fc = float(rain[:, sel].sum(axis=1).mean())
        def total(y):
            b = _same_date(a, acad0 - y)
            v = obs.reindex(pd.date_range(b, periods=7))
            return float(v.sum()) if v.notna().all() else np.nan
        clim = np.array([total(y) for y in range(FIRST_YEAR, acad0)])
        clim = clim[np.isfinite(clim)]
        def z(x):
            return float(ndtri(np.clip(((clim < x).sum() + 0.5 * (clim == x).sum() + 0.5) / (len(clim) + 1), 0.01, 0.99)))
        zf = z(fc)
        zy = np.array([z(v) if np.isfinite(v := total(y)) else 0.0 for y in years])
        lw += -0.5 * (zy - r * zf) ** 2 / (1 - r * r) + 0.5 * zy ** 2
        weeks.append({'from': f'{a:%Y-%m-%d}', 'forecast_mm': round(fc, 1), 'normal_mm': round(float(np.median(clim)), 1),
                      'z': round(zf, 2), 'skill': r})
    lam = 1.0
    for _ in range(30):
        w = np.exp(lam * (lw - lw.max()))
        w /= w.sum()
        if 1 / np.sum(w ** 2) >= MIN_EFFECTIVE_YEARS:
            break
        lam *= 0.8
    return w, {'weeks': weeks, 'effective_years': round(float(1 / np.sum(w ** 2)), 1), 'flattened': round(lam, 2),
               'top_years': [int(years[i]) for i in np.argsort(-w)[:5]]}


def ec46_rain(root: Path, timeout: int = 60, fetched=None) -> list[dict]:
    """ECMWF extended range (EC46, latest run, all members, via Open-Meteo): rain over the catchment above
    Farmoor for each whole week (Mon-Sun) ahead, against the CRU TS normal for that time of year (the month's
    normal, scaled to 7 days). Raw model output, shown as it comes."""
    days, rain = fetched if fetched is not None else ec46_fetch(timeout)
    era = pd.read_csv(Path(root) / OBS_RAIN_FILE, index_col=0, parse_dates=True).iloc[:, 0]
    era = era[era.index.year >= FIRST_YEAR]
    out = []
    first = days[0] + pd.Timedelta(days=(7 - days[0].dayofweek) % 7)      # the next Monday (or today)
    for w in range(7):
        a = first + pd.Timedelta(days=7 * w)
        sel = (days >= a) & (days < a + pd.Timedelta(days=7))
        if sel.sum() < 7:
            break
        tot = rain[:, sel].sum(axis=1)
        mid = a + pd.Timedelta(days=3)
        normal = float(era[era.index.month == mid.month].median()) * 7 / mid.days_in_month
        out.append({'week': f'{a:%Y-%m-%d}', 'median': round(float(np.median(tot))), 'p10': round(float(np.percentile(tot, 10))),
                    'p90': round(float(np.percentile(tot, 90))), 'normal': round(normal), 'members': int(rain.shape[0])})
    return out


# Torpids in its last five weeks: the outlook's chance adjusted by the rain since 1 January (Anu Dudhia's
# observation that a wet January and February loses Torpids). logit p' = a + b logit p + c z (days so far / 60),
# z = this year's mean rain since 1 Jan against the same days in every year since 1994 (ERA5). Fitted
# leave-one-year-out on hindcasts 1994-2026 (Brier skill vs the record: last week 36 -> 52%, 1-3 weeks 22 -> 27%,
# 3-5 weeks 18 -> 22%; worse beyond five weeks, so not used then).
TORPIDS_RAIN_COEF = (0.38, 0.82, 1.29)
TORPIDS_RAIN_DAYS = 35


def rain_since_jan1(root: Path, t0: pd.Timestamp, timeout: int = 60) -> Optional[pd.Series]:
    """Daily catchment rain from 1 January to t0: ERA5 (stored, then Open-Meteo's archive), with Open-Meteo's
    recent analysis for the last few days the archive does not have yet."""
    import requests
    jan1 = pd.Timestamp(t0.year, 1, 1)
    obs = pd.read_csv(Path(root) / ERA5_DAILY, index_col=0, parse_dates=True).iloc[:, 0]
    s = obs.loc[jan1:t0]
    if len(s) and s.index[-1] >= t0:
        return s
    lat, lon = '51.75,52.0,52.0,51.75,51.75,52.0', '-2.0,-2.0,-1.5,-1.5,-1.0,-1.0'
    got = []
    for url, extra in (('https://archive-api.open-meteo.com/v1/archive', {'start_date': f'{jan1:%Y-%m-%d}', 'end_date': f'{t0:%Y-%m-%d}'}),
                       ('https://api.open-meteo.com/v1/forecast', {'past_days': 92, 'forecast_days': 1})):
        try:
            r = requests.get(url, timeout=timeout, params={'latitude': lat, 'longitude': lon, 'daily': 'precipitation_sum',
                                                           'timezone': 'GMT', **extra})
            r.raise_for_status()
            js = r.json()
            js = js if isinstance(js, list) else [js]
            v = np.nanmean([np.array(j['daily']['precipitation_sum'], dtype=float) for j in js], axis=0)
            got.append(pd.Series(v, index=pd.to_datetime(js[0]['daily']['time'])).dropna())
        except Exception as e:          # noqa: BLE001
            print(f'  rain since 1 Jan: {url.split("/")[2]} failed ({e})')
    for g in got:
        s = s.combine_first(g)
    s = s.loc[jan1:t0]
    full = pd.date_range(jan1, t0)
    return s if s.reindex(full).notna().all() else None


def torpids_rain_adjust(root: Path, p: float, t0: pd.Timestamp, race_start: pd.Timestamp) -> tuple[float, Optional[dict]]:
    """(adjusted chance, details) in the last TORPIDS_RAIN_DAYS before Torpids from 1 January; else (p, None)."""
    before = (race_start - t0).days
    if not (0 < before <= TORPIDS_RAIN_DAYS) or t0 < pd.Timestamp(race_start.year, 1, 1):
        return p, None
    s = rain_since_jan1(root, t0)
    if s is None:
        return p, None
    obs = pd.read_csv(Path(root) / ERA5_DAILY, index_col=0, parse_dates=True).iloc[:, 0]
    n = len(s)
    others = [obs.loc[pd.Timestamp(y, 1, 1):pd.Timestamp(y, 1, 1) + pd.Timedelta(days=n - 1)].mean()
              for y in range(1994, race_start.year)]
    z = (s.mean() - np.mean(others)) / np.std(others)
    a, b, c = TORPIDS_RAIN_COEF
    lp = np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
    p2 = float(1 / (1 + np.exp(-(a + b * lp + c * z * n / 60))))
    return p2, {'rain_mm_per_day': round(float(s.mean()), 2), 'days': n, 'normal_mm_per_day': round(float(np.mean(others)), 2),
                'chance_before_rain': round(p, 2)}


# Bias correction of the page's percentages (calibrate_termly.py, from leave-one-year-out hindcasts):
# logit p' = a + b logit p for each lead group.
CALIBRATION_FILE = Path('data') / 'termly' / 'calibration.json'


def calibration_group(lead_days: int) -> str:
    return 'during' if lead_days <= 0 else 'up to 1 month' if lead_days <= 31 else '1-3 months' if lead_days <= 92 else '3+ months'


def calibration_group_torpids(days_before: int) -> str:
    return 'last 5 weeks' if days_before <= 35 else '5 weeks-3 months' if days_before <= 92 else '3+ months'


def _calibration(root: Path) -> Optional[dict]:
    f = Path(root) / CALIBRATION_FILE
    return json.loads(f.read_text()) if f.exists() else None


def _cal(b, p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return float(1 / (1 + np.exp(-(b[0] + b[1] * np.log(p / (1 - p))))))


def calibrate_tercile(root: Path, t: dict, lead_days: int) -> dict:
    """{'lower','usual','higher'} bias-corrected for this lead."""
    c = _calibration(root)
    b = c and c['terms'].get(calibration_group(lead_days))
    if not b:
        return t
    lo, hi = _cal(b['lower'], t['lower']), _cal(b['higher'], t['higher'])
    s = max(1.0, lo + hi)
    lo, hi = round(lo / s, 2), round(hi / s, 2)
    return {'lower': lo, 'higher': hi, 'usual': round(1 - lo - hi, 2)}


def calibrate_torpids(root: Path, p_go: float, days_before: int) -> float:
    c = _calibration(root)
    b = c and c['torpids'].get(calibration_group_torpids(days_before))
    return _cal(b, p_go) if b else p_go
