"""
Shared catchment model (``hybrid_shared``): one rainfall-runoff model for all three stretches.

The deployed hybrid (physics_models.ReservoirHybrid) is trained separately per stretch, so the
Isis and Godstow models each infer their own soil wetness from their own last 100 h, and can
disagree badly about the same ground (Nov 2026: 26 % vs 47 % wet). Here the catchment is modelled
once and every stretch reads from it:

    sub-catchment (own gauges)      soil → quick / 3-cascade medium      (shared by every stretch)
      upper    Cotswold Thames, Windrush, Evenlode, Coln, Churn + Eynsham, Osney
      cherwell Cherwell and Ray: Grimsbury, Bicester, Byfield
      lower    Ock, Thame and the reach below Oxford: Stanford, Abingdon, Wheatley, Aylesbury,
               Benson, Cleeve
                      │ fast outflow, weighted by learned non-negative contributions
                      ▼
    stretch            groundwater store (pinned to its own flow) + channel store → rating curve
      godstow      ← upper
      isis         ← upper + cherwell   (the Isis differential carries the King's Mill / Cherwell head)
      wallingford  ← upper + cherwell + lower

Each forecast still starts exactly at its stretch's observed differential (the channel is pinned
through that stretch's rating curve), and the guarantees of the per-stretch hybrid carry over: no
rise without water, more rain never lowers a forecast (all contributions are non-negative), smooth
trajectories. One LSTM encoder reads the recent past of all three stretches and the sub-catchment
rain and sets the shared stores; a stretch whose gauges are missing is masked (its inputs zeroed,
flagged unavailable) rather than dropping the window.
"""

from __future__ import annotations

import copy
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_data import (ENC_FFILL_LIMIT, HORIZON, SEQ_LEN, TRAIN_END, VAL_YEAR, Standardiser, doy_phase,
                           encoder_frame)
from .physics_models import M3S_PER_MM_H, SmoothRatingCurve, _bounded, _unbound

STRETCHES = ('godstow', 'isis', 'wallingford')
SUBCATCHMENTS = ('upper', 'cherwell', 'lower')
# Short gauge names as in the hourly frames (hybrid_live._short_gauge_name).
GAUGES = {
    'upper': ['Worsham', 'Osney', 'Bourton', 'St', 'Swindon', 'Eynsham', 'Chipping', 'Rapsgate', 'Stowell',
              'Shorncote'],
    'cherwell': ['Grimsbury', 'Bicester', 'Byfield'],
    'lower': ['Aylesbury', 'Abingdon', 'Stanford', 'Wheatley', 'Cleeve', 'Benson'],
}
ALL_GAUGES = [g for k in SUBCATCHMENTS for g in GAUGES[k]]
# Which sub-catchments drain to each stretch (rows: STRETCHES, columns: SUBCATCHMENTS).
FEEDS = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
# Approximate areas (km²) set the starting contributions relative to the upper Thames at Farmoor;
# the model learns the actual (non-negative) weights.
AREA_KM2 = {'upper': 1609.0, 'cherwell': 900.0, 'lower': 900.0}
# The Wallingford differential's errors run ~1.4x the Oxford stretches'; this keeps it from
# dominating the joint loss.
STRETCH_LOSS_WEIGHT = np.array([1.0, 1.0, 0.7], dtype=np.float32)
SUB_RAIN_WINDOWS = (6, 24, 72, 168, 720)
CHECKPOINT_GROUPS = {'oxford': ('godstow', 'isis'), 'wallingford': ('wallingford',)}
CHECKPOINT_FOR = {loc: g for g, locs in CHECKPOINT_GROUPS.items() for loc in locs}
TEST_START = '2024-11-01'
TEST_END = '2026-01-10'


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def subcatchment_rain(rain: pd.DataFrame) -> pd.DataFrame:
    """Mean rain over each sub-catchment's reporting gauges (mm/h); the all-gauge mean where none
    of a sub-catchment's gauges report; 0 where no gauge reports at all."""
    rain = rain.clip(lower=0)
    allmean = rain.reindex(columns=[g for g in ALL_GAUGES if g in rain.columns]).mean(axis=1)
    out = {}
    for k in SUBCATCHMENTS:
        cols = [g for g in GAUGES[k] if g in rain.columns]
        m = rain[cols].mean(axis=1) if cols else pd.Series(np.nan, index=rain.index)
        out[k] = m.fillna(allmean).fillna(0.0)
    return pd.DataFrame(out, index=rain.index)


_DIFF_COLS: Dict[tuple, List[str]] = {}


def differential_columns(enc_cols: List[str], sample: pd.DataFrame) -> List[str]:
    """Encoder columns that depend on the differential (found by perturbing it)."""
    key = tuple(enc_cols)
    if key not in _DIFF_COLS:
        a = encoder_frame(sample)
        b = sample.copy()
        b['differential'] = b['differential'] * 1.37 + 0.11
        b = encoder_frame(b)
        tail = slice(len(a) // 2, None)
        _DIFF_COLS[key] = [c for c in enc_cols
                           if not np.allclose(a[c].to_numpy()[tail], b[c].to_numpy()[tail], equal_nan=True)]
    return _DIFF_COLS[key]


def joint_features(frames: Dict[str, pd.DataFrame], rain: pd.DataFrame,
                   diff_cols: Dict[str, List[str]]) -> pd.DataFrame:
    """Hourly encoder inputs on frames['godstow']'s grid.

    frames: per stretch, an hourly frame with 'differential', its gauge columns and
    'flow_m3s_Farmoor' (the per-stretch hybrid's input). rain: all gauges (short names).
    diff_cols: per stretch, its encoder columns that depend on the differential.
    Inputs: Godstow's full feature set (rain, Farmoor flow, time, its differential), the
    differential features of the Isis and Wallingford, sub-catchment rain sums and, per stretch,
    an availability flag (1 where its differential features are finite).
    """
    grid = frames['godstow'].index
    parts = []
    for loc in STRETCHES:
        enc = encoder_frame(frames[loc].reindex(grid))
        cols = list(enc.columns) if loc == 'godstow' else diff_cols[loc]
        sub = enc[cols].add_prefix(f'{loc}:')
        dcols = [f'{loc}:{c}' for c in diff_cols[loc]]
        sub[f'{loc}:available'] = np.isfinite(sub[dcols].to_numpy()).all(axis=1).astype(np.float32)
        parts.append(sub)
    sr = subcatchment_rain(rain.reindex(grid))
    csum = sr.cumsum()
    for k in SUBCATCHMENTS:
        for w in SUB_RAIN_WINDOWS:
            parts.append((csum[k] - csum[k].shift(w, fill_value=0.0)).rename(f'sub:{k}_rain_{w}h').to_frame())
    return pd.concat(parts, axis=1).astype(np.float32)


def shared_mask(enc_cols: List[str], diff_cols: Dict[str, List[str]]) -> np.ndarray:
    """Columns every window needs finite: everything except the stretches' differential features
    and availability flags."""
    per = {f'{loc}:{c}' for loc in STRETCHES for c in diff_cols[loc]}
    return np.array([c not in per and not c.endswith(':available') for c in enc_cols])


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

@dataclass
class JointData:
    index: pd.DatetimeIndex
    enc: np.ndarray          # (T, F) unscaled
    enc_cols: list
    shared_cols: np.ndarray  # (F,) bool: columns every window needs finite (rain, flow, time)
    rain_k: np.ndarray       # (T, K) sub-catchment mm/h
    rain24_k: np.ndarray     # (T, K)
    rain168_k: np.ndarray    # (T, K)
    season: np.ndarray       # (T,)
    diff: np.ndarray         # (T, L)
    stretch_ok: np.ndarray   # (T, L) usable as a start for that stretch
    flow: np.ndarray         # (T,) Farmoor m³/s
    train_t: np.ndarray
    val_t: np.ndarray
    test_t: np.ndarray
    diff_cols: dict


def load_all_rain(project_root: Path) -> pd.DataFrame:
    """Every gauge's hourly rain (short names), NaN where a gauge did not report."""
    from ..data.loader import load_historical_rainfall
    raw = load_historical_rainfall(project_root=project_root, location='wallingford', verbose=False)
    raw.index = pd.DatetimeIndex(raw.index)
    raw.index = raw.index.tz_localize('UTC') if raw.index.tz is None else raw.index.tz_convert('UTC')
    renamed = {}
    for col in raw.columns:
        after = col.split('mm_', 1)[1] if 'mm_' in col else col
        renamed[col] = after.split('-')[0]
    raw = raw.rename(columns=renamed)
    raw = raw.loc[:, ~raw.columns.duplicated()].astype(float)
    return raw.resample('1h').sum(min_count=1).clip(lower=0, upper=50)


def build_joint_data(project_root: Path, stride: int = 4) -> JointData:
    cache = project_root / 'data' / 'shared_catchment_data.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    from ..evaluation import load_merged
    merged = {loc: load_merged(loc, project_root) for loc in STRETCHES}
    lo = min(m.index.min() for m in merged.values())
    hi = max(m.index.max() for m in merged.values())
    grid = pd.date_range(lo, hi, freq='1h', tz=merged['godstow'].index.tz)
    rain = load_all_rain(project_root).reindex(grid)
    flow = None
    frames = {}
    for loc in STRETCHES:
        m = merged[loc].reindex(grid)
        gauges = [c for c in merged[loc].columns if c not in ('differential',) and not c.startswith('flow_m3s_')]
        f = pd.concat([m['differential'], rain.reindex(columns=gauges)], axis=1)
        if flow is None:
            flow = pd.concat([mm['flow_m3s_Farmoor'] for mm in merged.values()], axis=1).mean(axis=1).reindex(grid)
        f['flow_m3s_Farmoor'] = flow
        frames[loc] = f
    diff_cols = {}
    for loc in STRETCHES:
        sample = frames[loc].dropna(subset=['differential']).iloc[-4000:]
        diff_cols[loc] = differential_columns(list(encoder_frame(sample).columns), sample)
    enc_df = joint_features(frames, rain, diff_cols)
    enc_cols = list(enc_df.columns)
    enc = enc_df.to_numpy(dtype=np.float32)
    shared = shared_mask(enc_cols, diff_cols)
    sr = subcatchment_rain(rain).to_numpy(dtype=np.float32)
    csum = np.concatenate([np.zeros((1, sr.shape[1])), np.cumsum(sr, 0, dtype=np.float64)])
    pos = np.arange(len(sr))
    rain24 = (csum[pos + 1] - csum[np.maximum(0, pos - 23)]).astype(np.float32)
    rain168 = (csum[pos + 1] - csum[np.maximum(0, pos - 167)]).astype(np.float32)
    diff = np.stack([frames[loc]['differential'].to_numpy(dtype=np.float32) for loc in STRETCHES], 1)

    # A window is usable when the shared inputs are finite over the encoder window; a stretch is
    # usable when its own differential features are finite over the window and it has d0 and
    # most of its 10-day target.
    def window_all(ok: np.ndarray) -> np.ndarray:
        c = np.convolve(ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[:len(ok)]
        return c == SEQ_LEN

    shared_ok = window_all(np.isfinite(enc[:, shared]).all(axis=1))
    stretch_ok = np.zeros_like(diff, dtype=bool)
    for j, loc in enumerate(STRETCHES):
        avail = enc[:, enc_cols.index(f'{loc}:available')] > 0.5
        fut = np.convolve(np.isfinite(diff[:, j]).astype(np.int32), np.ones(HORIZON, dtype=np.int32), 'full')
        fut_ok = np.zeros(len(diff), bool)
        fut_ok[:len(diff) - HORIZON - 1] = fut[HORIZON:len(diff) - 1] >= 200
        stretch_ok[:, j] = window_all(avail) & np.isfinite(diff[:, j]) & fut_ok
    cand = np.arange(SEQ_LEN, len(grid) - HORIZON - 1, stride)
    good = shared_ok[cand] & stretch_ok[cand].any(axis=1)
    cand = cand[good]
    t0 = grid[cand]
    train_end = pd.Timestamp(TRAIN_END, tz=grid.tz)
    val_lo = pd.Timestamp(f'{VAL_YEAR}-01-01', tz=grid.tz)
    val_hi = pd.Timestamp(f'{VAL_YEAR + 1}-01-01', tz=grid.tz)
    near_val = (t0 >= val_lo - pd.Timedelta(hours=HORIZON)) & (t0 < val_hi)
    is_val = (t0 >= val_lo) & (t0 < val_hi - pd.Timedelta(hours=HORIZON))
    is_train = (t0 < train_end) & ~near_val
    is_test = (t0 >= pd.Timestamp(TEST_START, tz=grid.tz)) & (t0 < pd.Timestamp(TEST_END, tz=grid.tz))
    data = JointData(index=grid, enc=enc, enc_cols=enc_cols, shared_cols=shared, rain_k=sr,
                     rain24_k=rain24, rain168_k=rain168, season=doy_phase(grid), diff=diff,
                     stretch_ok=stretch_ok, flow=flow.to_numpy(dtype=np.float32),
                     train_t=cand[np.asarray(is_train)], val_t=cand[np.asarray(is_val)],
                     test_t=cand[np.asarray(is_test)], diff_cols=diff_cols)
    pd.to_pickle(data, cache)
    return data


class JointSampler:
    def __init__(self, data: JointData, scaler: Standardiser, device: torch.device):
        self.device = device
        enc = data.enc.copy()
        # A stretch's differential features are zeroed (its scaled mean) where it is unavailable.
        z = scaler.transform(enc)
        for loc in STRETCHES:
            avail = enc[:, data.enc_cols.index(f'{loc}:available')] > 0.5
            cols = [i for i, c in enumerate(data.enc_cols) if c.startswith(f'{loc}:') and not c.endswith(':available')
                    and not data.shared_cols[i]]
            z[np.ix_(~avail, cols)] = 0.0
        self.enc = torch.from_numpy(z).to(device)
        self.rain = torch.from_numpy(data.rain_k).to(device)
        self.rain24 = torch.from_numpy(data.rain24_k).to(device)
        self.rain168 = torch.from_numpy(data.rain168_k).to(device)
        self.season = torch.from_numpy(data.season).to(device)
        self.diff = torch.from_numpy(np.nan_to_num(data.diff, nan=0.0)).to(device)
        self.diff_ok = torch.from_numpy(np.isfinite(data.diff)).to(device)
        self.stretch_ok = torch.from_numpy(data.stretch_ok).to(device)
        months = data.index.month.to_numpy()
        self.winter = torch.from_numpy(np.isin(months, (11, 12, 1, 2, 3))).to(device)
        self.past_off = torch.arange(-SEQ_LEN + 1, 1, device=device)
        self.fut_off = torch.arange(1, HORIZON + 1, device=device)

    def batch(self, t0: np.ndarray) -> Dict[str, torch.Tensor]:
        t = torch.as_tensor(t0, device=self.device, dtype=torch.long)
        past = t[:, None] + self.past_off[None, :]
        fut = t[:, None] + self.fut_off[None, :]
        ok0 = self.stretch_ok[t]                               # (B, L)
        return {
            'x': self.enc[past], 'rain': self.rain[fut], 'season': self.season[fut],
            'rain24': self.rain24[t], 'rain168': self.rain168[t], 'd0': self.diff[t],
            'y': self.diff[fut], 'y_ok': self.diff_ok[fut] & ok0[:, None, :], 'winter': self.winter[t],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SharedCatchmentHybrid(nn.Module):
    architecture = 'shared_catchment'
    N_MEDIUM = 3
    SUB_RANGES = {
        'k_quick': (1.0, 36.0, 6.0),
        'k_medium': (3.0, 96.0, 16.0),
        'beta': (0.3, 6.0, 2.0),
        'smax': (10.0, 400.0, 100.0),
        'et_base': (0.0, 0.1, 0.01),
        'et_summer': (0.0, 0.3, 0.08),
    }
    STRETCH_RANGES = {
        'k_slow': (48.0, 4000.0, 600.0),
        'k_channel': (1.0, 72.0, 8.0),
    }
    N_SUB_OUT = 4 + N_MEDIUM + 4      # s0, quick fill, medium fill, medium logits, 4 param shifts

    def __init__(self, n_features: int, d_lo: List[float], d_hi: List[float], hidden: int = 96,
                 dropout: float = 0.2, z_min: float = math.log(1e-4), z_max: float = math.log(3.0)):
        super().__init__()
        K, L = len(SUBCATCHMENTS), len(STRETCHES)
        self.K, self.L = K, L
        self.n_features, self.hidden = n_features, hidden
        self.q_floor = math.exp(z_min)
        self.encoder = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, K * self.N_SUB_OUT + L)
        self.sub_raw = nn.ParameterDict({
            name: nn.Parameter(torch.full((K,), _unbound(init, lo, hi)))
            for name, (lo, hi, init) in self.SUB_RANGES.items()
        })
        self.split_logits = nn.Parameter(torch.tensor([[0.0, 0.5, 0.0]] * K))
        self.stretch_raw = nn.ParameterDict({
            name: nn.Parameter(torch.full((L,), _unbound(init, lo, hi)))
            for name, (lo, hi, init) in self.STRETCH_RANGES.items()
        })
        self.register_buffer('feeds', torch.from_numpy(FEEDS))
        rel = torch.tensor([AREA_KM2[k] / AREA_KM2['upper'] for k in SUBCATCHMENTS])
        self.contrib_raw = nn.Parameter(torch.log(torch.expm1(rel)).repeat(L, 1))
        self.ratings = nn.ModuleList([SmoothRatingCurve(d_lo[j], d_hi[j], z_min, z_max) for j in range(L)])

    def sub_param(self, name: str, shift: torch.Tensor = None) -> torch.Tensor:
        lo, hi, _ = self.SUB_RANGES[name]
        raw = self.sub_raw[name]
        return _bounded(raw if shift is None else raw + shift, lo, hi)

    def stretch_param(self, name: str) -> torch.Tensor:
        lo, hi, _ = self.STRETCH_RANGES[name]
        return _bounded(self.stretch_raw[name], lo, hi)

    def contributions(self) -> torch.Tensor:
        """(L, K) non-negative weight of each sub-catchment's runoff at each stretch."""
        return F.softplus(self.contrib_raw) * self.feeds

    def forward(self, x, rain, season, rain24, rain168, d0, return_states: bool = False):
        """x (B, SEQ, F); rain (B, H, K) mm/h; season (B, H); rain24/rain168 (B, K) mm; d0 (B, L).
        Returns D (B, H, L) [and Q (B, H, L), states]."""
        B, H, K = rain.shape
        L = self.L
        h, _ = self.encoder(x)
        out = self.head(self.drop(h[:, -1]))
        sub = out[:, :K * self.N_SUB_OUT].reshape(B, K, self.N_SUB_OUT)
        rho = torch.sigmoid(out[:, K * self.N_SUB_OUT:])                       # (B, L)
        s = torch.sigmoid(sub[:, :, 0])
        quick = torch.sigmoid(sub[:, :, 1]) * rain24
        medium_total = torch.sigmoid(sub[:, :, 2]) * rain168
        medium_w = torch.softmax(sub[:, :, 3:3 + self.N_MEDIUM], dim=2)
        medium = [medium_total * medium_w[:, :, i] for i in range(self.N_MEDIUM)]
        mod = sub[:, :, 3 + self.N_MEDIUM:]
        k_q = self.sub_param('k_quick', mod[:, :, 0])
        k_m = self.sub_param('k_medium', mod[:, :, 1])
        beta = self.sub_param('beta', mod[:, :, 2])
        smax = self.sub_param('smax')
        et_base, et_summer = self.sub_param('et_base'), self.sub_param('et_summer')
        logits = self.split_logits[None] + torch.stack([mod[:, :, 3], torch.zeros_like(mod[:, :, 3]),
                                                        -mod[:, :, 3]], dim=2)
        split = torch.softmax(logits, 2)
        f_q, f_m, f_s = split[:, :, 0], split[:, :, 1], split[:, :, 2]
        k_s, k_c = self.stretch_param('k_slow'), self.stretch_param('k_channel')
        a_q, a_m = 1 - torch.exp(-1 / k_q), 1 - torch.exp(-1 / k_m)
        a_s, a_c = 1 - torch.exp(-1 / k_s), 1 - torch.exp(-1 / k_c)
        w = self.contributions()                                               # (L, K)

        q0 = []
        for j in range(L):
            r = self.ratings[j]
            z0 = torch.clamp(r.inverse(d0[:, j].contiguous()), r.z_min - 5.0, r.z_max + 3.0)
            q0.append(torch.clamp(torch.exp(z0) - self.q_floor, min=1e-7))
        q0 = torch.stack(q0, 1)                                                # (B, L)
        channel = q0 * (1 - a_c) / a_c
        slow = rho * q0 * (1 - a_s) / a_s

        et = et_base[None, None, :] + et_summer[None, None, :] * torch.clamp(season, min=0)[:, :, None]
        qs, states = [], []
        if return_states:
            states.append(torch.cat([s, quick, sum(medium), slow, channel], 1))
        for t in range(H):
            p = torch.clamp(rain[:, t], min=0)
            runoff = p * s.clamp(min=1e-6) ** beta
            s = torch.clamp(s + (p - runoff - et[:, t] * s) / smax, 0.0, 1.0)
            quick = quick + f_q * runoff
            o_q = a_q * quick
            quick = quick - o_q
            inflow_m = f_m * runoff
            for i in range(self.N_MEDIUM):
                medium[i] = medium[i] + inflow_m
                inflow_m = a_m * medium[i]
                medium[i] = medium[i] - inflow_m
            fast = (o_q + inflow_m) @ w.T                                      # (B, L)
            slow = slow + (f_s * runoff) @ w.T
            o_s = a_s * slow
            slow = slow - o_s
            channel = channel + fast + o_s
            q = a_c * channel
            channel = channel - q
            qs.append(q)
            if return_states:
                states.append(torch.cat([s, quick, sum(medium), slow, channel], 1))
        q = torch.stack(qs, 1)                                                 # (B, H, L)
        d = torch.stack([self.ratings[j](torch.log(q[:, :, j] + self.q_floor)) for j in range(L)], 2)
        if return_states:
            return d, q, torch.stack(states, 1)
        return d

    def describe(self) -> Dict[str, float]:
        out = {}
        for name in self.SUB_RANGES:
            for k, v in zip(SUBCATCHMENTS, self.sub_param(name).tolist()):
                out[f'{name}_{k}'] = v
        split = torch.softmax(self.split_logits, 1).tolist()
        for k, sp in zip(SUBCATCHMENTS, split):
            out.update({f'split_quick_{k}': sp[0], f'split_medium_{k}': sp[1], f'split_slow_{k}': sp[2]})
        for name in self.STRETCH_RANGES:
            for loc, v in zip(STRETCHES, self.stretch_param(name).tolist()):
                out[f'{name}_{loc}'] = v
        c = self.contributions().tolist()
        for loc, row in zip(STRETCHES, c):
            for k, v in zip(SUBCATCHMENTS, row):
                if v > 0:
                    out[f'contrib_{loc}_{k}'] = v
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULTS = {
    'epochs': 30, 'batch_size': 128, 'lr': 1e-3, 'physics_lr': 5e-3, 'patience': 7, 'hidden': 96,
    'dropout': 0.2, 'winter_weight': 1.5, 'w_1_24': 2.0, 'w_25_72': 1.5, 'w_73_240': 1.0,
    'rating_smooth_weight': 1e-3, 'max_train_batches': None, 'seed': 0,
    # Exponential moving average of the weights (0 = off): validation and the saved model use the
    # averaged weights, which damps the epoch-to-epoch noise of joint training.
    'ema_decay': 0.0,
}


def horizon_weights(cfg, device) -> torch.Tensor:
    w = torch.full((HORIZON,), float(cfg['w_73_240']), device=device)
    w[:72] = float(cfg['w_25_72'])
    w[:24] = float(cfg['w_1_24'])
    return w


def build_model(data: JointData, cfg: Dict) -> SharedCatchmentHybrid:
    d_lo, d_hi = [], []
    for j in range(len(STRETCHES)):
        d = data.diff[np.isfinite(data.diff[:, j]), j]
        d_lo.append(float(d.min()) - 0.05)
        d_hi.append(float(d.max()) + 0.3)
    model = SharedCatchmentHybrid(data.enc.shape[1], d_lo, d_hi, hidden=cfg['hidden'], dropout=cfg['dropout'])
    before = np.asarray(data.index < pd.Timestamp(TRAIN_END, tz=data.index.tz))
    for j, r in enumerate(model.ratings):
        ok = before & np.isfinite(data.diff[:, j]) & np.isfinite(data.flow)
        r.init_from_data(np.log(data.flow[ok] / M3S_PER_MM_H + model.q_floor), data.diff[ok, j])
    return model


def compute_loss(model, b, cfg, hw, lw):
    pred = model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'])
    w = hw[None, :, None] * b['y_ok'].float() * lw[None, None, :]
    w = w * torch.where(b['winter'], cfg['winter_weight'], 1.0)[:, None, None]
    mae = ((pred - b['y']).abs() * w).sum() / w.sum().clamp(min=1)
    return mae, pred


@torch.no_grad()
def evaluate(model, sampler: JointSampler, t_idx: np.ndarray, batch_size: int = 256) -> Dict[str, float]:
    model.eval()
    sums = torch.zeros(HORIZON, len(STRETCHES))
    counts = torch.zeros(HORIZON, len(STRETCHES))
    for i in range(0, len(t_idx), batch_size):
        b = sampler.batch(t_idx[i:i + batch_size])
        pred = model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'])
        m = b['y_ok'].float()
        sums += ((pred - b['y']).abs() * m).sum(0)
        counts += m.sum(0)
    per = (sums / counts.clamp(min=1)).numpy()
    out = {'mae': float(per.mean())}
    for j, loc in enumerate(STRETCHES):
        out[f'mae_{loc}'] = float(per[:, j].mean())
        out[f'mae_{loc}_1_24h'] = float(per[:24, j].mean())
        out[f'mae_{loc}_73_240h'] = float(per[72:, j].mean())
    return out


def train(data: JointData, cfg_override: Optional[Dict] = None, verbose: bool = True):
    cfg = {**DEFAULTS, **(cfg_override or {})}
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    device = torch.device('cpu')
    rows = np.unique((data.train_t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]).ravel())
    scaler = Standardiser.fit(data.enc[rows])
    sampler = JointSampler(data, scaler, device)
    model = build_model(data, cfg)
    hw = horizon_weights(cfg, device)
    lw = torch.from_numpy(STRETCH_LOSS_WEIGHT)
    phys_prefix = ('sub_raw', 'split_logits', 'stretch_raw', 'contrib_raw', 'ratings')
    phys = [p for n, p in model.named_parameters() if n.startswith(phys_prefix)]
    rest = [p for n, p in model.named_parameters() if not n.startswith(phys_prefix)]
    opt = torch.optim.Adam([{'params': rest, 'lr': cfg['lr']}, {'params': phys, 'lr': cfg['physics_lr']}])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    ema = copy.deepcopy(model) if cfg['ema_decay'] > 0 else None
    scored = ema if ema is not None else model
    history, bad = [], 0
    # Checkpoints are picked per group of stretches: Godstow and the Isis together (so the two
    # Oxford stretches always share one model and one catchment state), Wallingford on its own
    # (it peaks earlier in joint training). best[group] = (val, state, epoch).
    best = {g: (math.inf, None, -1) for g in CHECKPOINT_GROUPS}
    for epoch in range(cfg['epochs']):
        t_start = time.time()
        model.train()
        order = rng.permutation(data.train_t)
        n_batches = int(math.ceil(len(order) / cfg['batch_size']))
        if cfg['max_train_batches']:
            n_batches = min(n_batches, cfg['max_train_batches'])
        tr = []
        for i in range(n_batches):
            b = sampler.batch(order[i * cfg['batch_size']:(i + 1) * cfg['batch_size']])
            loss, _ = compute_loss(model, b, cfg, hw, lw)
            rough = sum(((r.slopes()[1:] - r.slopes()[:-1]) ** 2).sum() for r in model.ratings)
            total = loss + cfg['rating_smooth_weight'] * rough
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(cfg['ema_decay']).add_(pm.detach(), alpha=1 - cfg['ema_decay'])
            tr.append(float(loss))
        val = evaluate(scored, sampler, data.val_t)
        sched.step(val['mae'])
        rec = {'epoch': epoch + 1, 'secs': time.time() - t_start, 'train_mae': float(np.mean(tr)),
               **{f'val_{k}': v for k, v in val.items()}}
        history.append(rec)
        if verbose:
            print(f"epoch {epoch + 1:2d} ({rec['secs']:.0f}s) train {rec['train_mae']:.4f} val {val['mae']:.4f} "
                  f"[G {val['mae_godstow']:.4f} I {val['mae_isis']:.4f} W {val['mae_wallingford']:.4f}]", flush=True)
        improved = False
        for g, locs in CHECKPOINT_GROUPS.items():
            v = float(np.mean([val[f'mae_{loc}'] for loc in locs]))
            if v < best[g][0] - 1e-5:
                best[g] = (v, copy.deepcopy(scored.state_dict()), epoch + 1)
                improved = True
        bad = 0 if improved else bad + 1
        if bad >= cfg['patience']:
            if verbose:
                print('early stop; best ' + ', '.join(f'{g} epoch {b[2]} ({b[0]:.4f})' for g, b in best.items()), flush=True)
            break
    states = {g: b[1] for g, b in best.items()}
    model.load_state_dict(states['oxford'])
    model.eval()
    model.checkpoints = states
    model.best_epochs = {g: b[2] for g, b in best.items()}
    return model, scaler, cfg, history


def stretch_gauges(project_root: Path) -> Dict[str, List[str]]:
    """The gauges each stretch's encoder rain features sum over (its training frame's columns)."""
    out = {}
    for loc in STRETCHES:
        cols = pd.read_pickle(Path(project_root) / 'data' / f'full_frame_{loc}.pkl').columns
        out[loc] = [c for c in cols if c != 'differential' and not c.startswith('flow_m3s_')]
    return out


def save(model, scaler, cfg, history, data: JointData, name: str, models_dir: Path) -> Path:
    stem = f'shared_{name}'
    states = getattr(model, 'checkpoints', None) or {'oxford': model.state_dict()}
    torch.save({g: {k: v.cpu() for k, v in sd.items()} for g, sd in states.items()}, models_dir / f'{stem}.pt')
    meta = {'name': name, 'cfg': cfg, 'scaler': scaler.state(), 'enc_cols': data.enc_cols,
            'shared_cols': data.shared_cols, 'diff_cols': data.diff_cols, 'history': history,
            'n_features': data.enc.shape[1], 'physics': model.describe(),
            'gauges': stretch_gauges(Path(models_dir).parent),
            'd_lo': [r.d_lo for r in model.ratings], 'd_hi': [float(r.d_knots()[-1]) for r in model.ratings],
            'best_epochs': getattr(model, 'best_epochs', None)}
    with open(models_dir / f'{stem}.pkl', 'wb') as f:
        pickle.dump(meta, f)
    return models_dir / f'{stem}.pt'


def load(name: str, models_dir: Path, location: str = 'godstow'):
    """The model a stretch uses: Godstow and the Isis share the 'oxford' checkpoint, Wallingford has
    its own (CHECKPOINT_GROUPS). Older files hold a single state dict, used for every stretch."""
    stem = f'shared_{name}'
    with open(models_dir / f'{stem}.pkl', 'rb') as f:
        meta = pickle.load(f)
    cfg = meta['cfg']
    model = SharedCatchmentHybrid(meta['n_features'], meta['d_lo'], meta['d_hi'], hidden=cfg['hidden'],
                                  dropout=cfg['dropout'])
    states = torch.load(models_dir / f'{stem}.pt', map_location='cpu')
    if 'oxford' in states:
        states = states.get(CHECKPOINT_FOR[location]) or states['oxford']
    model.load_state_dict(states)
    model.eval()
    return model, Standardiser(meta['scaler']['mean'], meta['scaler']['std']), meta


@torch.no_grad()
def batch_forecast(model, sampler: JointSampler, t_idx: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """(N, 240, L) forecasts with the rain that fell, for backtests."""
    model.eval()
    out = []
    for i in range(0, len(t_idx), batch_size):
        b = sampler.batch(t_idx[i:i + batch_size])
        out.append(model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0']).numpy())
    return np.concatenate(out)
