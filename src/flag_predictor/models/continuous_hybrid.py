"""
Continuous-state physics hybrid (``hybrid_cs``).

The September-2026 hybrid (``physics_models.ReservoirHybrid``) guesses the
catchment state at every forecast start from 100 h of encoder features plus a
30-day rain sum. A summer soil-moisture deficit builds over months, so that
guess cannot know about it. Here the reservoir model instead runs
*continuously* over the whole hourly rain record (from 2000), so soil wetness
and groundwater carry their own long memory, and the encoder only *corrects*
those simulated states at t0 from recent observations (data assimilation by
nudging; with zero corrections the forecast starts from the pure simulation).

Changes to the hydrology, relative to ReservoirHybrid:

* ``f_direct``: a small share of rain runs off regardless of soil wetness
  (riparian / saturated / impervious ground near the channel), so a storm on
  dry soil still produces a response.
* evapotranspiration is either the old seasonal cosine (``et_mode='cos'``) or
  ``k_et * ET0`` from ERA5 reference evapotranspiration (``et_mode='pet'``).
* The hydrology is pre-trained on the long Farmoor flow record (1992+, rain
  2000+) before the differential is ever seen, so the stores are in physical
  units (mm/h over the Farmoor catchment) and rare floods inform them.

All structural guarantees of ReservoirHybrid are kept, because the forecast
step is the same: the forecast starts at the observed D(t0); quick/medium
stores are capped by the last 24 h / 168 h of rain and groundwater outflow by
Q0, so with no recent and no future rain the river can only hold or fall;
every path from rain to Q is non-decreasing, so more rain never lowers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .physics_models import M3S_PER_MM_H, SmoothRatingCurve, _bounded, _unbound

STATE_NAMES = ('s', 'quick', 'm1', 'm2', 'm3', 'slow', 'channel')
N_STATE = len(STATE_NAMES)
Q_EPS = 1e-3          # mm/h (~0.45 m3/s at Farmoor) for log-flow losses


def _logit(p: torch.Tensor, lo: float = 1e-3) -> torch.Tensor:
    p = p.clamp(lo, 1 - lo)
    return torch.log(p) - torch.log1p(-p)


class ReservoirCore(nn.Module):
    """Soil bucket -> quick / 3-cascade medium / groundwater -> channel.

    Parameters start at the values the September hybrid learned (Isis seed 0).
    """

    N_MEDIUM = 3
    RANGES = {
        'k_quick': (1.0, 36.0, 9.0),
        'k_medium': (3.0, 96.0, 18.5),
        'k_slow': (48.0, 4000.0, 475.0),
        'k_channel': (1.0, 72.0, 22.0),
        'beta': (0.3, 6.0, 2.15),
        'smax': (10.0, 400.0, 76.0),
        'et_base': (0.0, 0.1, 0.012),     # et_mode='cos'
        'et_summer': (0.0, 0.3, 0.106),   # et_mode='cos'
        'k_et': (0.2, 1.6, 0.8),          # et_mode='pet'
        'f_direct': (0.0, 0.3, 0.03),
    }

    def __init__(self, et_mode: str = 'cos', direct: bool = True):
        super().__init__()
        assert et_mode in ('cos', 'pet')
        self.et_mode = et_mode
        self.direct = direct
        self.params_raw = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(_unbound(init, lo, hi)))
            for name, (lo, hi, init) in self.RANGES.items()
        })
        # quick / medium / slow split of soil runoff, starting at 18 / 58 / 24 %.
        self.split_logits = nn.Parameter(torch.tensor([0.0, 1.17, 0.29]))

    def param(self, name: str, shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        lo, hi, _ = self.RANGES[name]
        raw = self.params_raw[name]
        if shift is not None:
            raw = raw + shift
        return _bounded(raw, lo, hi)

    def describe(self) -> Dict[str, float]:
        out = {n: float(self.param(n)) for n in self.RANGES}
        if not self.direct:
            out['f_direct'] = 0.0
        split = torch.softmax(self.split_logits, 0).tolist()
        out.update(split_quick=split[0], split_medium=split[1], split_slow=split[2])
        return out

    def et_demand(self, season: torch.Tensor, et0: Optional[torch.Tensor]) -> torch.Tensor:
        """Potential ET (mm/h); actual ET is this times soil wetness s."""
        if self.et_mode == 'pet':
            return self.param('k_et') * et0.clamp(min=0)
        return self.param('et_base') + self.param('et_summer') * season.clamp(min=0)

    def params(self, mod: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Step parameters; ``mod`` (B, 4) shifts k_quick, k_medium, beta and the split."""
        P = {
            'k_q': self.param('k_quick', mod[:, 0] if mod is not None else None),
            'k_m': self.param('k_medium', mod[:, 1] if mod is not None else None),
            'beta': self.param('beta', mod[:, 2] if mod is not None else None),
            'k_s': self.param('k_slow'),
            'k_c': self.param('k_channel'),
            'smax': self.param('smax'),
            'f_dir': self.param('f_direct') if self.direct else torch.zeros(()),
        }
        if mod is not None:
            logits = self.split_logits[None, :] + torch.stack(
                [mod[:, 3], torch.zeros_like(mod[:, 3]), -mod[:, 3]], dim=1)
            split = torch.softmax(logits, 1)
            P['f_q'], P['f_m'], P['f_s'] = split[:, 0], split[:, 1], split[:, 2]
        else:
            split = torch.softmax(self.split_logits, 0)
            P['f_q'], P['f_m'], P['f_s'] = split[0], split[1], split[2]
        for k in ('q', 'm', 's', 'c'):
            P[f'a_{k}'] = 1 - torch.exp(-1 / P[f'k_{k}'])
        return P

    @staticmethod
    def step(st: Sequence[torch.Tensor], p: torch.Tensor, demand: torch.Tensor,
             P: Dict[str, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        s, quick, m1, m2, m3, slow, ch = st
        p = p.clamp(min=0)
        direct = p * P['f_dir']
        soil_runoff = (p - direct) * s.clamp(min=1e-6) ** P['beta']
        s = torch.clamp(s + (p - direct - soil_runoff - demand * s) / P['smax'], 0.0, 1.0)
        quick = quick + direct + P['f_q'] * soil_runoff
        o_q = P['a_q'] * quick
        quick = quick - o_q
        m1 = m1 + P['f_m'] * soil_runoff
        o1 = P['a_m'] * m1
        m1 = m1 - o1
        m2 = m2 + o1
        o2 = P['a_m'] * m2
        m2 = m2 - o2
        m3 = m3 + o2
        o3 = P['a_m'] * m3
        m3 = m3 - o3
        slow = slow + P['f_s'] * soil_runoff
        o_s = P['a_s'] * slow
        slow = slow - o_s
        ch = ch + o_q + o3 + o_s
        q = P['a_c'] * ch
        ch = ch - q
        return (s, quick, m1, m2, m3, slow, ch), q

    def simulate(self, rain: torch.Tensor, demand: torch.Tensor, state0: torch.Tensor,
                 P: Optional[Dict] = None, keep_states: bool = False):
        """rain, demand: (B, T) mm/h; state0: (B, 7). Returns Q (B, T), final (B, 7)[, states (B, T, 7)]."""
        P = P or self.params()
        st = tuple(state0[:, i] for i in range(N_STATE))
        qs, states = [], []
        for t in range(rain.shape[1]):
            st, q = self.step(st, rain[:, t], demand[:, t], P)
            qs.append(q)
            if keep_states:
                states.append(torch.stack(st, 1))
        out = (torch.stack(qs, 1), torch.stack(st, 1))
        if keep_states:
            out = out + (torch.stack(states, 1),)
        return out

    def steady_state(self, q: torch.Tensor, s: float = 0.6) -> torch.Tensor:
        """A plausible starting state with groundwater carrying 60% of flow q (mm/h)."""
        P = self.params()
        st = torch.zeros(q.shape[0], N_STATE)
        st[:, 0] = s
        st[:, 5] = 0.6 * q * (1 - P['a_s']) / P['a_s']
        st[:, 6] = q * (1 - P['a_c']) / P['a_c']
        return st.detach()


# --------------------------------------------------------------------------
# Long hourly record (rain from 2000, Farmoor flow from 1992, optional ET0)
# --------------------------------------------------------------------------

@dataclass
class LongRecord:
    index: pd.DatetimeIndex
    rain: np.ndarray        # (T,) mean-station mm/h (over the gauges given)
    season: np.ndarray      # (T,) cos day-of-year phase
    flow_mmh: np.ndarray    # (T,) Farmoor flow in mm/h over the catchment (NaN gaps)
    et0: Optional[np.ndarray] = None   # (T,) mm/h


def load_hourly_npz(folder, names: Sequence[str]) -> pd.DataFrame:
    """Per-gauge hourly series written by to_hourly.py (t: int64 ns UTC, v: float32)."""
    from pathlib import Path
    cols = {}
    for path in sorted(Path(folder).glob('*.npz')):
        short = path.stem.split('-')[0]
        if short not in names:
            continue
        z = np.load(path)
        cols[short] = pd.Series(z['v'], index=pd.to_datetime(z['t'], utc=True))
    return pd.DataFrame(cols)


def build_long_record(folder, gauges: Sequence[str], et0: Optional[pd.Series] = None,
                      start: str = '2000-01-02', end: Optional[str] = None) -> LongRecord:
    from .physics_data import doy_phase
    rain_st = load_hourly_npz(folder, gauges)
    flow = load_hourly_npz(folder, ['Farmoor'])['Farmoor']
    end = end or str(rain_st.index.max())
    idx = pd.date_range(pd.Timestamp(start, tz='UTC'), pd.Timestamp(end).tz_convert('UTC')
                        if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz='UTC'), freq='1h')
    rain_st = rain_st.reindex(idx)
    rain = rain_st.clip(lower=0).mean(axis=1)
    # Hours with no reporting gauge at all: treat as dry (rare; ~0.1%).
    rain = rain.fillna(0.0).to_numpy(np.float32)
    flow = flow.reindex(idx).to_numpy(np.float32) / M3S_PER_MM_H
    e = None
    if et0 is not None:
        e = fill_et0(et0, idx)
    return LongRecord(index=idx, rain=rain, season=doy_phase(idx), flow_mmh=flow, et0=e)


def et0_climatology(et0: pd.Series) -> pd.Series:
    """Mean ET0 by (day of year, hour)."""
    key = et0.index.dayofyear * 24 + et0.index.hour
    return et0.groupby(key).mean()


def fill_et0(et0: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """ET0 on idx; gaps (incl. beyond the record) from the day-of-year/hour climatology."""
    clim = et0_climatology(et0)
    out = et0.reindex(idx)
    key = idx.dayofyear * 24 + idx.hour
    fill = pd.Series(clim.reindex(key).to_numpy(), index=idx)
    return out.fillna(fill).fillna(float(et0.mean())).to_numpy(np.float32)


# --------------------------------------------------------------------------
# Pre-training on the long Farmoor flow record (multiple shooting)
# --------------------------------------------------------------------------

def loss_mask(index: pd.DatetimeIndex, flow: np.ndarray, burn_in_end: str = '2001-01-01') -> Dict[str, np.ndarray]:
    """Train: 2001 -> 2022-11, plus 2024-01-08 -> 2024-10-21. Val: 2023. Test: 2024-11-01 on."""
    ok = np.isfinite(flow)
    t = index
    tz = t.tz
    ts = lambda s: pd.Timestamp(s, tz=tz)  # noqa: E731
    train = ((t >= ts(burn_in_end)) & (t < ts('2022-11-20'))) | ((t >= ts('2024-01-08')) & (t < ts('2024-10-21')))
    val = (t >= ts('2023-01-01')) & (t < ts('2024-01-01'))
    test = t >= ts('2024-11-01')
    return {'train': ok & np.asarray(train), 'val': ok & np.asarray(val), 'test': ok & np.asarray(test)}


def flow_scores(q: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    q, o = q[mask], obs[mask]
    nse = 1 - np.sum((q - o) ** 2) / np.sum((o - o.mean()) ** 2)
    lq, lo = np.log(q + Q_EPS), np.log(o + Q_EPS)
    lnse = 1 - np.sum((lq - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)
    return {'nse': float(nse), 'log_nse': float(lnse), 'bias_pct': float(100 * (q.sum() / o.sum() - 1)),
            'mae_m3s': float(np.mean(np.abs(q - o)) * M3S_PER_MM_H)}


def run_full(core: ReservoirCore, rec: LongRecord, state0: Optional[torch.Tensor] = None,
             keep_states: bool = True):
    """One sequential no-grad simulation over the whole record."""
    with torch.no_grad():
        rain = torch.from_numpy(rec.rain)[None]
        dem = core.et_demand(torch.from_numpy(rec.season)[None],
                             torch.from_numpy(rec.et0)[None] if rec.et0 is not None else None)
        if dem.dim() == 0:
            dem = dem.expand_as(rain)
        dem = dem.expand_as(rain).contiguous()
        if state0 is None:
            q0 = np.nanmean(rec.flow_mmh[: 24 * 30]) if np.isfinite(rec.flow_mmh[: 24 * 30]).any() else 0.05
            state0 = core.steady_state(torch.tensor([float(q0)]))
        return core.simulate(rain, dem, state0, keep_states=keep_states)


def pretrain_core(core: ReservoirCore, rec: LongRecord, epochs: int = 300, chunk: int = 2016,
                  lr: float = 0.02, rel_weight: float = 0.5, verbose: bool = True, seed: int = 0):
    """Fit the hydrology to Farmoor flow by multiple shooting.

    The record is cut into chunks that are simulated in parallel. Each chunk
    starts from the state the previous chunk ended in on the previous epoch,
    so after a few epochs the chunks join into one continuous simulation
    while gradients only run back one chunk (truncated BPTT).
    """
    torch.manual_seed(seed)
    T = len(rec.rain)
    n = int(math.ceil(T / chunk))
    pad = n * chunk - T

    def chunked(a, fill=0.0):
        a = np.concatenate([a, np.full(pad, fill, dtype=np.float32)])
        return torch.from_numpy(a.reshape(n, chunk).astype(np.float32))

    rain = chunked(rec.rain)
    season = chunked(rec.season)
    et0 = chunked(rec.et0) if rec.et0 is not None else None
    masks = loss_mask(rec.index, rec.flow_mmh)
    obs = np.nan_to_num(rec.flow_mmh, nan=0.0)
    obs_t = chunked(obs)
    m_train = chunked(np.concatenate([masks['train']]).astype(np.float32))
    mean_obs = float(obs[masks['train']].mean())

    q_start = obs_t[:, :24 * 7].mean(1).clamp(min=0.01)
    state0 = core.steady_state(q_start)
    opt = torch.optim.Adam(core.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        dem = core.et_demand(season, et0)
        dem = dem.expand_as(rain)
        q, final = core.simulate(rain, dem, state0)
        lq = torch.log(q + Q_EPS)
        lo = torch.log(obs_t + Q_EPS)
        w = m_train
        loss_log = ((lq - lo).abs() * w).sum() / w.sum()
        loss_rel = ((q - obs_t).abs() * w).sum() / w.sum() / mean_obs
        loss = loss_log + rel_weight * loss_rel
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            # Carry states forward: chunk k+1 starts where chunk k ended.
            state0 = torch.cat([state0[:1], final[:-1].detach()], 0)
        history.append({'epoch': ep + 1, 'loss': float(loss), 'log': float(loss_log), 'rel': float(loss_rel)})
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            d = core.describe()
            print(f"ep {ep + 1:3d} loss {float(loss):.4f} (log {float(loss_log):.4f} rel {float(loss_rel):.4f}) "
                  + ' '.join(f"{k}={v:.3g}" for k, v in d.items()), flush=True)
    return history


# --------------------------------------------------------------------------
# Forecast model: simulated state at t0, nudged by an encoder
# --------------------------------------------------------------------------

N_PRIOR = 6


def prior_features(state_t0: torch.Tensor, q_sim0: torch.Tensor, q0: torch.Tensor,
                   rain24: torch.Tensor, rain168: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
    """Summaries of the simulated state at t0, for the encoder head (B, N_PRIOR)."""
    s = state_t0[:, 0]
    quick, med, slow = state_t0[:, 1], state_t0[:, 2:5].sum(1), state_t0[:, 5]
    return torch.stack([
        _logit(s),
        _logit(quick / (rain24 + 1e-3)),
        _logit(med / (rain168 + 1e-3)),
        _logit(a_s * slow / q0.clamp(min=1e-6)),
        torch.log(q_sim0 + Q_EPS) - torch.log(q0 + Q_EPS),
        torch.log(q0 + Q_EPS),
    ], dim=1)


class ContinuousHybrid(nn.Module):
    architecture = 'continuous_hybrid'

    def __init__(self, n_features: int, d_lo: float, d_hi: float, core: ReservoirCore,
                 hidden: int = 64, dropout: float = 0.2, z_min: float = math.log(1e-4),
                 z_max: float = math.log(3.0)):
        super().__init__()
        self.core = core
        self.q_floor = math.exp(z_min)
        self.encoder = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        # Corrections: s, rho, quick fill, medium fill, 3 medium logits; then 4 parameter shifts.
        self.n_corr = 4 + core.N_MEDIUM
        self.head = nn.Linear(hidden + N_PRIOR, self.n_corr + 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.rating = SmoothRatingCurve(d_lo, d_hi, z_min, z_max)

    def forward(self, x, rain, season, rain24, rain168, d0, state_t0, q_sim0,
                et0: Optional[torch.Tensor] = None, return_q: bool = False, return_states: bool = False):
        """
        x: (B, SEQ, F) encoder features; rain/season/et0: (B, H) future forcing;
        rain24/rain168/d0/q_sim0: (B,); state_t0: (B, 7) simulated state at t0.
        """
        core = self.core
        P0 = core.params()
        z0 = torch.clamp(self.rating.inverse(d0), self.rating.z_min - 5.0, self.rating.z_max + 3.0)
        q0 = torch.clamp(torch.exp(z0) - self.q_floor, min=1e-7)
        prior = prior_features(state_t0, q_sim0, q0, rain24, rain168, P0['a_s'])
        h, _ = self.encoder(x)
        out = self.head(torch.cat([self.drop(h[:, -1]), prior], 1))
        corr, mod = out[:, :self.n_corr], out[:, self.n_corr:]
        P = core.params(mod)

        s = torch.sigmoid(prior[:, 0] + corr[:, 0])
        rho = torch.sigmoid(prior[:, 3] + corr[:, 1])
        quick = torch.sigmoid(prior[:, 1] + corr[:, 2]) * rain24
        medium_total = torch.sigmoid(prior[:, 2] + corr[:, 3]) * rain168
        med_prior = torch.log(state_t0[:, 2:5].clamp(min=0) + 1e-4)
        medium_w = torch.softmax(med_prior + corr[:, 4:4 + core.N_MEDIUM], dim=1)
        m = [medium_total * medium_w[:, i] for i in range(core.N_MEDIUM)]

        a_c, a_s = P['a_c'], P['a_s']
        channel = q0 * (1 - a_c) / a_c
        slow = rho * q0 * (1 - a_s) / a_s
        st = (s, quick, m[0], m[1], m[2], slow, channel)
        demand = core.et_demand(season, et0)
        if demand.dim() < 2:
            demand = demand.expand_as(rain)
        qs, states = [], []
        for t in range(rain.shape[1]):
            st, q = core.step(st, rain[:, t], demand[:, t], P)
            qs.append(q)
            if return_states:
                states.append(torch.stack(st, 1))
        q = torch.stack(qs, 1)
        d = self.rating(torch.log(q + self.q_floor))
        if return_states:
            return d, q, torch.stack(states, 1)
        if return_q:
            return d, q
        return d

    def describe(self) -> Dict[str, float]:
        return self.core.describe()
