"""
Training, saving, loading and prediction for the continuous-state hybrid.

The forecast windows are the same as for the September hybrid
(``physics_data.LocationData``: 100 h encoder, 240 h horizon, train through
2024-10-21 except 2023, 2023 for validation). What is new is that every window
also gets the reservoir state that the continuous simulation had reached at
its t0, and the hourly ET0 over its horizon.
"""

from __future__ import annotations

import copy
import math
import pickle
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..evaluation import mean_station_rain
from .continuous_hybrid import (
    N_STATE, ContinuousHybrid, LongRecord, ReservoirCore, build_long_record, fill_et0, run_full,
)
from .physics_data import (
    HORIZON, SEQ_LEN, TRAIN_END, LocationData, Standardiser, WindowSampler, doy_phase,
    load_location_data,
)
from .physics_models import M3S_PER_MM_H

# Gauges each location's models use (the column order of the training frame).
ALL_GAUGES = [
    'Aylesbury', 'Worsham', 'Osney', 'Bourton', 'Abingdon', 'St', 'Swindon', 'Eynsham',
    'Chipping', 'Stanford', 'Rapsgate', 'Wheatley', 'Stowell', 'Cleeve', 'Shorncote',
    'Benson', 'Grimsbury',
]
LOCATION_GAUGES = {
    'isis': ALL_GAUGES,
    'wallingford': ALL_GAUGES,
    'godstow': [g for g in ALL_GAUGES if g != 'Grimsbury'],
}

DEFAULTS = {
    'epochs': 25,
    'batch_size': 128,
    'lr': 1e-3,
    'rating_lr': 5e-3,
    'core_lr': 0.0,          # 0 = hydrology frozen at its flow pre-training
    'patience': 5,
    'hidden': 64,
    'dropout': 0.2,
    'winter_weight': 1.5,
    'w_1_24': 2.0,
    'w_25_72': 1.5,
    'w_73_240': 1.0,
    'rating_smooth_weight': 1e-3,
    'seed': 0,
    'max_train_batches': None,   # cap per epoch for quick runs
    # Water allowed in the quick / medium stores at t0 is capped by the rain of
    # the last quick_cap_h / medium_cap_h hours (the no-rise-without-rain
    # guarantee). The September hybrid used 24 h / 168 h.
    'quick_cap_h': 24,
    'medium_cap_h': 168,
}


def load_core(core_name: str, models_dir: Path) -> ReservoirCore:
    with open(models_dir / f'core_{core_name}.pkl', 'rb') as f:
        meta = pickle.load(f)
    core = ReservoirCore(et_mode=meta['et_mode'], direct=meta['direct'])
    core.load_state_dict(torch.load(models_dir / f'core_{core_name}.pt', map_location='cpu'))
    return core


def long_rain_for_location(root: Path, location: str, data: Optional[LocationData] = None,
                           et0: Optional[pd.Series] = None) -> LongRecord:
    """Long hourly record for one location; over the differential period the
    rain is exactly the training frame's, so the state at t0 is consistent
    with the past rain the encoder sees."""
    rec = build_long_record(root / 'data' / 'hourly', LOCATION_GAUGES[location], et0=et0)
    if data is not None:
        rain = pd.Series(rec.rain, index=rec.index)
        frame_rain = pd.Series(data.rain, index=data.index)
        common = frame_rain.index.intersection(rain.index)
        rain.loc[common] = frame_rain.loc[common].to_numpy()
        # Extend past the hourly files if the frame runs later.
        extra = frame_rain.loc[frame_rain.index > rec.index[-1]]
        if len(extra):
            idx = rec.index.append(extra.index)
            rain = pd.concat([rain, extra])
            rec = LongRecord(index=idx, rain=rain.to_numpy(np.float32), season=doy_phase(idx),
                             flow_mmh=np.concatenate([rec.flow_mmh, np.full(len(extra), np.nan, np.float32)]),
                             et0=fill_et0(et0, idx) if et0 is not None else None)
        else:
            rec.rain = rain.to_numpy(np.float32)
    return rec


class StateSimulation:
    """Continuous simulation of one core over a long record, indexed by time."""

    def __init__(self, core: ReservoirCore, rec: LongRecord):
        q, _, states = run_full(core, rec)
        self.index = rec.index
        self.q = q[0].numpy()
        self.states = states[0].numpy()
        self.pos = pd.Series(np.arange(len(rec.index)), index=rec.index)

    def at(self, times) -> tuple:
        p = self.pos.reindex(pd.DatetimeIndex(times)).to_numpy()
        if np.isnan(p).any():
            raise KeyError('forecast start outside the simulated record')
        p = p.astype(int)
        return self.states[p], self.q[p]


def trailing_sum(rain: np.ndarray, hours: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(rain, dtype=np.float64)])
    pos = np.arange(len(rain))
    return (csum[pos + 1] - csum[np.maximum(0, pos - hours + 1)]).astype(np.float32)


class ContinuousSampler(WindowSampler):
    def __init__(self, data: LocationData, scaler: Standardiser, sim: StateSimulation,
                 et0: Optional[np.ndarray], device: torch.device, cap_hours=(24, 168)):
        super().__init__(data, scaler, device)
        # 'rain24' / 'rain168' carry the store caps (trailing rain over cap_hours).
        self.rain24 = torch.from_numpy(trailing_sum(data.rain, cap_hours[0])).to(device)
        self.rain168 = torch.from_numpy(trailing_sum(data.rain, cap_hours[1])).to(device)
        st, q = sim.at(data.index)
        self.state = torch.from_numpy(st.astype(np.float32)).to(device)
        self.q_sim = torch.from_numpy(q.astype(np.float32)).to(device)
        self.et0 = torch.from_numpy(et0).to(device) if et0 is not None else None

    def batch(self, t0: np.ndarray) -> Dict[str, torch.Tensor]:
        b = super().batch(t0)
        t = torch.as_tensor(t0, device=self.device, dtype=torch.long)
        b['state_t0'] = self.state[t]
        b['q_sim0'] = self.q_sim[t]
        if self.et0 is not None:
            b['et0'] = self.et0[t[:, None] + self.fut_off[None, :]]
        return b


def _forward(model: ContinuousHybrid, b: Dict, **kw):
    return model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'],
                 b['state_t0'], b['q_sim0'], et0=b.get('et0'), **kw)


def _horizon_weights(cfg: Dict) -> torch.Tensor:
    w = torch.full((HORIZON,), float(cfg['w_73_240']))
    w[:72] = float(cfg['w_25_72'])
    w[:24] = float(cfg['w_1_24'])
    return w


@torch.no_grad()
def evaluate_windows(model, sampler, t_idx, batch_size: int = 512) -> Dict[str, float]:
    model.eval()
    sums = torch.zeros(HORIZON)
    counts = torch.zeros(HORIZON)
    for i in range(0, len(t_idx), batch_size):
        b = sampler.batch(t_idx[i:i + batch_size])
        pred = _forward(model, b)
        m = b['y_ok'].float()
        sums += ((pred - b['y']).abs() * m).sum(0)
        counts += m.sum(0)
    per_lead = (sums / counts.clamp(min=1)).numpy()
    return {'mae': float(per_lead.mean()), 'mae_1_24h': float(per_lead[:24].mean()),
            'mae_25_72h': float(per_lead[24:72].mean()), 'mae_73_240h': float(per_lead[72:].mean())}


def location_et0(root: Path, core: ReservoirCore):
    if core.et_mode != 'pet':
        return None
    from ..data.download_et import load_et0
    et0 = load_et0(root / 'data' / 'era5' / 'et0_hourly.csv.gz')
    if et0 is None:
        raise FileNotFoundError('data/era5/et0_hourly.csv.gz is needed for an ET0 core')
    return et0


def train_location(location: str, core_name: str, root: Path, cfg_override: Optional[Dict] = None,
                   verbose: bool = True):
    cfg = {**DEFAULTS, **(cfg_override or {})}
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    models_dir = root / 'models'
    data = load_location_data(location, root)
    core_name = core_name.format(location=location)   # per-location cores: 'old_{location}'
    core = load_core(core_name, models_dir)
    et0_series = location_et0(root, core)
    rec = long_rain_for_location(root, location, data, et0_series)
    sim = StateSimulation(core, rec)
    et0_frame = fill_et0(et0_series, data.index) if et0_series is not None else None

    scaler = Standardiser.fit(data.enc[np.unique((data.train_t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]).ravel())])
    sampler = ContinuousSampler(data, scaler, sim, et0_frame, torch.device('cpu'),
                                cap_hours=(cfg['quick_cap_h'], cfg['medium_cap_h']))

    d = data.diff[np.isfinite(data.diff)]
    model = ContinuousHybrid(data.enc.shape[1], d_lo=float(d.min()) - 0.05, d_hi=float(d.max()) + 0.3,
                             core=core, hidden=cfg['hidden'], dropout=cfg['dropout'])
    before = np.asarray(data.index < pd.Timestamp(TRAIN_END, tz=data.index.tz))
    ok = before & np.isfinite(data.diff) & np.isfinite(data.flow)
    model.rating.init_from_data(np.log(data.flow[ok] / M3S_PER_MM_H + model.q_floor), data.diff[ok])

    core_params = list(model.core.parameters())
    if cfg['core_lr'] <= 0:
        for p in core_params:
            p.requires_grad_(False)
    groups = [
        {'params': list(model.encoder.parameters()) + list(model.head.parameters()), 'lr': cfg['lr']},
        {'params': list(model.rating.parameters()), 'lr': cfg['rating_lr']},
    ]
    if cfg['core_lr'] > 0:
        groups.append({'params': core_params, 'lr': cfg['core_lr']})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    hw = _horizon_weights(cfg)

    history, best, bad = [], (math.inf, None, -1), 0
    for epoch in range(cfg['epochs']):
        t_start = time.time()
        if cfg['core_lr'] > 0 and epoch > 0:
            # Keep the simulated states consistent with the updated hydrology.
            sim = StateSimulation(model.core, rec)
            st, q = sim.at(data.index)
            sampler.state = torch.from_numpy(st.astype(np.float32))
            sampler.q_sim = torch.from_numpy(q.astype(np.float32))
        model.train()
        order = rng.permutation(data.train_t)
        n_batches = int(math.ceil(len(order) / cfg['batch_size']))
        if cfg['max_train_batches']:
            n_batches = min(n_batches, cfg['max_train_batches'])
        tr = []
        for i in range(n_batches):
            b = sampler.batch(order[i * cfg['batch_size']:(i + 1) * cfg['batch_size']])
            pred = _forward(model, b)
            w = hw[None, :] * b['y_ok'].float() * torch.where(b['winter'], cfg['winter_weight'], 1.0)[:, None]
            mae = ((pred - b['y']).abs() * w).sum() / w.sum().clamp(min=1)
            slopes = model.rating.slopes()
            loss = mae + cfg['rating_smooth_weight'] * ((slopes[1:] - slopes[:-1]) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
            opt.step()
            tr.append(float(mae.detach()))
        val = evaluate_windows(model, sampler, data.val_t)
        sched.step(val['mae'])
        rec_h = {'epoch': epoch + 1, 'secs': time.time() - t_start, 'train_mae': float(np.mean(tr)),
                 **{f'val_{k}': v for k, v in val.items()}}
        history.append(rec_h)
        if verbose:
            extra = ' '.join(f"{k}={v:.3g}" for k, v in model.describe().items()) if cfg['core_lr'] > 0 else ''
            print(f"epoch {epoch + 1:2d} ({rec_h['secs']:.0f}s) train_mae={rec_h['train_mae']:.4f} "
                  f"val_mae={val['mae']:.4f} [1-24h {val['mae_1_24h']:.4f} 25-72h {val['mae_25_72h']:.4f} "
                  f"73-240h {val['mae_73_240h']:.4f}] {extra}", flush=True)
        if val['mae'] < best[0] - 1e-5:
            best, bad = (val['mae'], copy.deepcopy(model.state_dict()), epoch + 1), 0
        else:
            bad += 1
            if bad >= cfg['patience']:
                if verbose:
                    print(f"early stop; best epoch {best[2]} val_mae={best[0]:.4f}")
                break
    model.load_state_dict(best[1])
    model.eval()
    extras = {'d_lo': model.rating.d_lo, 'd_hi': float(model.rating.d_knots()[-1]),
              'enc_cols': data.enc_cols, 'n_features': data.enc.shape[1],
              'gauges': LOCATION_GAUGES[location], 'best_val_mae': best[0], 'best_epoch': best[2]}
    return model, scaler, cfg, history, extras


def save_forecast_model(model: ContinuousHybrid, scaler: Standardiser, cfg: Dict, history, name: str,
                        core_name: str, location: str, extras: Dict, models_dir: Path) -> Path:
    stem = f'redesign_{name}_{location}'
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, models_dir / f'{stem}.pt')
    meta = {'family': 'continuous_hybrid', 'name': name, 'location': location, 'core': core_name,
            'et_mode': model.core.et_mode, 'direct': model.core.direct, 'cfg': cfg,
            'scaler': {'mean': scaler.mean.tolist(), 'std': scaler.std.tolist()},
            'history': history, 'physics': model.describe(), **extras}
    with open(models_dir / f'{stem}.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f'saved {stem} (best val MAE {extras["best_val_mae"]:.4f} at epoch {extras["best_epoch"]})')
    return models_dir / f'{stem}.pt'


def load_forecast_model(name: str, location: str, models_dir: Path):
    stem = f'redesign_{name}_{location}'
    with open(models_dir / f'{stem}.pkl', 'rb') as f:
        meta = pickle.load(f)
    core = ReservoirCore(et_mode=meta['et_mode'], direct=meta['direct'])
    cfg = meta['cfg']
    model = ContinuousHybrid(meta['n_features'], d_lo=meta['d_lo'], d_hi=meta['d_hi'], core=core,
                             hidden=cfg['hidden'], dropout=cfg['dropout'])
    model.load_state_dict(torch.load(models_dir / f'{stem}.pt', map_location='cpu'))
    model.eval()
    scaler = Standardiser(np.asarray(meta['scaler']['mean'], np.float32), np.asarray(meta['scaler']['std'], np.float32))
    return model, scaler, meta


# --------------------------------------------------------------------------
# Prediction in the evaluation interface: predict(history, future_rain) -> 241
# --------------------------------------------------------------------------

def simulate_states(core: ReservoirCore, rain: pd.Series, et0: Optional[pd.Series] = None,
                    flow_m3s: Optional[pd.Series] = None) -> StateSimulation:
    """Continuous simulation over an hourly rain series (mean-station mm/h)."""
    idx = pd.DatetimeIndex(rain.index)
    flow = (flow_m3s.reindex(idx).to_numpy(np.float32) / M3S_PER_MM_H if flow_m3s is not None
            else np.full(len(idx), np.nan, np.float32))
    rec = LongRecord(index=idx, rain=rain.fillna(0.0).to_numpy(np.float32), season=doy_phase(idx),
                     flow_mmh=flow, et0=fill_et0(et0, idx) if et0 is not None else None)
    return StateSimulation(core, rec)


def continuous_predictor(names, location: str, models_dir: Path, state_sims: Dict[str, StateSimulation],
                         et0: Optional[pd.Series] = None):
    """Mean of several continuous-hybrid models (seeds).

    ``state_sims`` maps each model name to the StateSimulation of its core;
    models sharing a frozen core can share one. ``et0`` covers the forecast
    horizon for ET0 cores (gaps fall back to the day-of-year climatology).
    """
    from .candidates import _past_inputs
    members = []
    for n in names:
        model, scaler, meta = load_forecast_model(n, location, models_dir)
        members.append((n, model, scaler, meta))

    @torch.no_grad()
    def predict(history: pd.DataFrame, future_rain: pd.DataFrame) -> np.ndarray:
        hist, enc_all, rain_hist = _past_inputs(history)
        t0 = hist.index[-1]
        fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
        rain_fut = future_rain.clip(lower=0).mean(axis=1).fillna(0.0).to_numpy(dtype=np.float32)
        d0 = float(hist['differential'].iloc[-1])
        season = torch.from_numpy(doy_phase(fut_idx))[None]
        outs = []
        for n, model, scaler, meta in members:
            caps = (meta['cfg'].get('quick_cap_h', 24), meta['cfg'].get('medium_cap_h', 168))
            x = scaler.transform(enc_all.reindex(columns=meta['enc_cols']).to_numpy(dtype=np.float32)[-SEQ_LEN:])
            st, q = state_sims[n].at([t0])
            e = None
            if model.core.et_mode == 'pet':
                e = torch.from_numpy(fill_et0(et0, fut_idx))[None]
            out = model(torch.from_numpy(x)[None], torch.from_numpy(rain_fut)[None], season,
                        torch.tensor([rain_hist[-caps[0]:].sum()]), torch.tensor([rain_hist[-caps[1]:].sum()]),
                        torch.tensor([d0], dtype=torch.float32),
                        torch.from_numpy(st.astype(np.float32)), torch.from_numpy(q.astype(np.float32)),
                        et0=e)
            outs.append(np.concatenate([[d0], out[0].numpy().astype(float)]))
        return np.mean(outs, axis=0)

    return predict


def backtest_state_sims(names, location: str, root: Path, models_dir: Path):
    """StateSimulations for backtests: long record + the training frame's rain."""
    data = load_location_data(location, root)
    sims, by_core = {}, {}
    et0 = None
    for n in names:
        model, _, meta = load_forecast_model(n, location, models_dir)
        if model.core.et_mode == 'pet' and et0 is None:
            et0 = location_et0(root, model.core)
        key = (meta['core'], meta['cfg'].get('core_lr', 0) > 0 and n)
        if key not in by_core:
            rec = long_rain_for_location(root, location, data, et0)
            by_core[key] = StateSimulation(model.core, rec)
        sims[n] = by_core[key]
    return sims, et0


# --------------------------------------------------------------------------
# Continuous-simulation states as extra encoder inputs (hybrid_simfeat)
# --------------------------------------------------------------------------

SIM_FEATURES = ['sim_soil_wetness', 'sim_log_q', 'sim_log_groundwater_out', 'sim_log_fast_stores',
                'sim_log_q_ratio']


def sim_features(core: ReservoirCore, sim: StateSimulation, index: pd.DatetimeIndex,
                 flow_m3s: Optional[np.ndarray]) -> np.ndarray:
    """(T, 5) features from the continuous simulation at the given hours.

    Soil wetness, simulated flow, groundwater outflow, water in the fast stores,
    and how far the simulated flow is from the observed Farmoor flow (0 where
    the gauge has no reading).
    """
    from .continuous_hybrid import Q_EPS
    st, q = sim.at(index)
    a_s = float(core.params()['a_s'].detach())
    flow_mmh = (np.asarray(flow_m3s, dtype=float) / M3S_PER_MM_H if flow_m3s is not None
                else np.full(len(index), np.nan))
    ratio = np.where(np.isfinite(flow_mmh), np.log(q + Q_EPS) - np.log(np.nan_to_num(flow_mmh) + Q_EPS), 0.0)
    return np.stack([st[:, 0], np.log(q + Q_EPS), np.log(a_s * st[:, 5] + Q_EPS),
                     np.log1p(st[:, 1:5].sum(1)), ratio], axis=1).astype(np.float32)
