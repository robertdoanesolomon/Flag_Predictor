"""
Hydrological long-range outlook (ensemble streamflow prediction, as in the UK Hydrological Outlook).

A daily catchment model (GR6J: Pushpalatha et al. 2011, J. Hydrol.; soil store, fast routing store, slow
exponential groundwater store, water exchange) turns rain and evaporation over the catchment above Farmoor
(ERA5, data/termly/era5_daily_met.csv) into Farmoor's daily flow. For an outlook issued on day t0 the model is
run with the weather that actually happened up to t0, which sets how wet the soil and groundwater are now; then
it is run on from that state once with each past year's weather from the same date. Each run is one possible
future: "if the coming months rain like 1995's, the river does this".

Before the runs go forward, the model's flow on t0 is matched to the gauged flow by rescaling its stores (a
simple update for what the model has got wrong so far).

    python -m flag_predictor.esp --calibrate      # fit the six parameters, write models/gr6j_farmoor.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

AREA_KM2 = 1609.0            # Thames at Farmoor
MM_TO_M3S = AREA_KM2 / 86.4  # mm/day over the catchment -> m3/s
MET_FILE = Path('data') / 'termly' / 'era5_daily_met.csv'
PARAM_FILE = Path('models') / 'gr6j_farmoor.json'
NAMES = ('X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'PETf')
BOUNDS = ((50, 2500), (-5, 3), (10, 800), (0.6, 8), (-2, 2), (0.5, 80), (0.6, 1.1))
NUH = 20


@njit(cache=True)
def _uh(x4):
    uh1, uh2 = np.zeros(NUH), np.zeros(NUH)
    def sh1(t):
        return 0.0 if t <= 0 else (1.0 if t >= x4 else (t / x4) ** 2.5)
    def sh2(t):
        if t <= 0:
            return 0.0
        if t <= x4:
            return 0.5 * (t / x4) ** 2.5
        if t < 2 * x4:
            return 1 - 0.5 * (2 - t / x4) ** 2.5
        return 1.0
    for j in range(NUH):
        uh1[j] = sh1(j + 1) - sh1(j)
        uh2[j] = sh2(j + 1) - sh2(j)
    return uh1, uh2


@njit(cache=True)
def run(p, rain, pet, state):
    """Daily flow (mm/day). ``state`` = [S, R, Exp, uh1 queue (NUH), uh2 queue (NUH)] is updated in place."""
    x1, x2, x3, x4, x5, x6, pf = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    uh1, uh2 = _uh(x4)
    n = len(rain)
    q = np.empty(n)
    S, R, E = state[0], state[1], state[2]
    q1 = state[3:3 + NUH]
    q2 = state[3 + NUH:3 + 2 * NUH]
    for t in range(n):
        P, Ep = rain[t], pf * pet[t]
        if P >= Ep:
            pn, en = P - Ep, 0.0
            tw = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1 - (S / x1) ** 2) * tw / (1 + S / x1 * tw)
            es = 0.0
        else:
            pn, en = 0.0, Ep - P
            tw = np.tanh(min(en / x1, 13.0))
            es = S * (2 - S / x1) * tw / (1 + (1 - S / x1) * tw)
            ps = 0.0
        S = S + ps - es
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + pn - ps
        for j in range(NUH - 1):
            q1[j] = q1[j + 1] + uh1[j] * 0.9 * pr
            q2[j] = q2[j + 1] + uh2[j] * 0.1 * pr
        q1[NUH - 1] = uh1[NUH - 1] * 0.9 * pr
        q2[NUH - 1] = uh2[NUH - 1] * 0.1 * pr
        q9, qq1 = q1[0], q2[0]
        F = x2 * (R / x3 - x5)
        R = max(0.0, R + 0.6 * q9 + F)
        qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
        R -= qr
        E = E + 0.4 * q9 + F
        a = E / x6
        if a > 33:
            qe = E
        elif a < -33:
            qe = 0.0
        else:
            qe = x6 * np.log1p(np.exp(a))
        E -= qe
        qd = max(0.0, qq1 + F)
        q[t] = qr + qe + qd
    state[0], state[1], state[2] = S, R, E
    return q


def initial_state(p) -> np.ndarray:
    st = np.zeros(3 + 2 * NUH)
    st[0], st[1], st[2] = 0.5 * p[0], 0.5 * p[2], 0.0
    return st


def load_met(root: Path) -> pd.DataFrame:
    return pd.read_csv(Path(root) / MET_FILE, index_col=0, parse_dates=True)


def simulate(p, met: pd.DataFrame, spinup_years: int = 3) -> pd.Series:
    """Flow (m3/s) through the whole record, after cycling the first few years to warm the stores."""
    p = np.asarray(p, float)
    st = initial_state(p)
    first = met.iloc[:365 * spinup_years]
    run(p, first.rain_mm.to_numpy(), first.pet_mm.to_numpy(), st)
    q = run(p, met.rain_mm.to_numpy(), met.pet_mm.to_numpy(), st)
    return pd.Series(q * MM_TO_M3S, index=met.index)


def kge(sim, obs) -> float:
    r = np.corrcoef(sim, obs)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)


def objective(p, met, q, mask) -> float:
    s = simulate(p, met).to_numpy()[mask]
    o = q[mask]
    if not np.isfinite(s).all() or s.std() < 1e-6:
        return 10.0
    v = -(0.5 * kge(np.sqrt(np.maximum(s, 0)), np.sqrt(o)) + 0.5 * kge(np.log(np.maximum(s, 0) + 0.5), np.log(o + 0.5)))
    return float(v) if np.isfinite(v) else 10.0


def calibrate(root: Path, q: pd.Series, met: pd.DataFrame) -> dict:
    from scipy.optimize import differential_evolution
    qq = q.reindex(met.index).to_numpy()
    mask = np.isfinite(qq) & (met.index >= '1994-01-01')
    res = differential_evolution(objective, BOUNDS, args=(met, qq, mask), seed=1, maxiter=120, popsize=20,
                                 tol=1e-6, polish=False, workers=1, updating='deferred')
    p = dict(zip(NAMES, map(float, res.x)))
    s = simulate(res.x, met).to_numpy()[mask]
    o = qq[mask]
    p['fit'] = {'kge_sqrt': round(kge(np.sqrt(s), np.sqrt(o)), 3), 'kge': round(kge(s, o), 3),
                'nse': round(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2), 3),
                'nse_log': round(1 - np.sum((np.log(s + .5) - np.log(o + .5)) ** 2)
                                 / np.sum((np.log(o + .5) - np.log(o + .5).mean()) ** 2), 3),
                'bias': round(float(s.mean() / o.mean() - 1), 3), 'period': '1994-2026'}
    (Path(root) / PARAM_FILE).write_text(json.dumps(p, indent=1))
    return p


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flag_predictor import termly as T
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', action='store_true')
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    met = load_met(root)
    q = T.load_flow(root)
    if a.calibrate:
        print(json.dumps(calibrate(root, q, met), indent=1))


class ESPOutlook:
    """Same interface as termly.Outlook: ``ensemble(t0)`` -> termly.Ensemble whose members are the catchment
    model run from today's state through each member year's weather (the climatology members are unchanged:
    those years' own flow). ``exclude``: an academic year left out of the member years, for hindcasts."""

    def __init__(self, q: pd.Series, root: Path, exclude=None, tau: float = 60.0, qmap: bool = True, params=None):
        from . import termly as T
        self.T, self.q, self.tau, self.qmap = T, q, tau, qmap
        self.base = T.Outlook(q, exclude=exclude)
        self.met = load_met(root)
        pj = params or json.loads((Path(root) / PARAM_FILE).read_text())
        self.p = np.array([pj[k] for k in NAMES], float)
        self.rain, self.pet = self.met.rain_mm.to_numpy(), self.met.pet_mm.to_numpy()
        # the model through the whole record once: its own flow, for the month-by-month mapping onto gauged flow
        sim = simulate(self.p, self.met)
        obs = q.reindex(sim.index)
        ok = obs.notna() & (sim.index.year >= 1994)
        self.maps = {}
        for m in range(1, 13):
            sel = ok & (sim.index.month == m)
            self.maps[m] = (np.sort(sim[sel].to_numpy()), np.sort(obs[sel].to_numpy()))

    def _map(self, x: np.ndarray, months: np.ndarray) -> np.ndarray:
        if not self.qmap:
            return x
        out = np.empty_like(x)
        for m in np.unique(months):
            s, o = self.maps[m]
            pr = np.interp(x[..., months == m], s, np.linspace(0, 1, len(s)))
            out[..., months == m] = np.interp(pr, np.linspace(0, 1, len(o)), o)
        return out

    def state_at(self, t0: pd.Timestamp) -> tuple[np.ndarray, float]:
        """Model stores at the end of day t0 (run with the weather up to then), and its flow (m3/s) over the last 3 days."""
        i = int(self.met.index.get_indexer([t0])[0])
        if i < 0:
            raise ValueError(f'no weather for {t0:%Y-%m-%d}')
        st = initial_state(self.p)
        run(self.p, self.rain[:365 * 3], self.pet[:365 * 3], st)
        qq = run(self.p, self.rain[:i + 1], self.pet[:i + 1], st) * MM_TO_M3S
        return st, qq[-3:]

    def ensemble(self, t0: pd.Timestamp, trend: float = 0.0):
        T = self.T
        ens = self.base.ensemble(t0, trend)
        t0 = ens.t0
        st0, sim3 = self.state_at(t0)
        days = ens.days
        months = days.month.to_numpy()
        acad0 = T.academic_year(t0)
        traces = []
        for y in ens.years:
            b = T._same_date(t0, acad0 - y)
            j = int(self.met.index.get_indexer([b])[0])
            if j < 0 or j + len(days) >= len(self.met):
                raise ValueError(f'no weather for member year {y}')
            st = st0.copy()
            traces.append(run(self.p, self.rain[j + 1:j + 1 + len(days)], self.pet[j + 1:j + 1 + len(days)], st))
        tr = self._map(np.array(traces) * MM_TO_M3S, months)
        # match the model to the gauge on t0, the correction fading over ``tau`` days
        now_sim = float(self._map(sim3[None, :], np.full(3, t0.month))[0].mean())
        now_obs = float(self.q.loc[t0 - pd.Timedelta(days=2):t0].mean())
        if np.isfinite(now_obs) and now_sim > 0:
            k = np.clip(now_obs / now_sim, 0.2, 5.0)
            tr = tr * k ** np.exp(-np.arange(1, len(days) + 1) / self.tau)
        ens.members = tr
        return ens
