"""
Training for the physics-redesign candidates.

Two families share this loop:

* ``hybrid``   – ReservoirHybrid (realism by construction; loss is plain MAE).
* ``lstm_v2``  – the September HourlyDecoderModel, fixed properly: no future
  observed flow (so no train/inference mismatch), mean-station rain as the
  only decoder covariate, and realism taught through the loss instead of
  clamped afterwards: a dry-climb penalty based on accumulated rain (the same
  72h / 2mm rule as the evaluation), a curvature penalty against jitter, a
  paired "more rain must not lower the forecast" penalty, and a soft recession
  limit.

Weights are saved as models/redesign_{name}_{location}.pt with a matching
.pkl config; the live models are never touched.
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
import torch.nn.functional as F

from ..config import PHYSICAL_CONSTRAINTS
from ..evaluation import DRY_LOOKBACK_H, DRY_RAIN_MM
from .lstm import HourlyDecoderModel
from .physics_data import HORIZON, LocationData, Standardiser, WindowSampler
from .physics_data import TRAIN_END
from .physics_models import M3S_PER_MM_H, ReservoirHybrid

MAX_DROP = PHYSICAL_CONSTRAINTS['max_recession_m_per_day'] / 24.0

DEFAULTS = {
    'epochs': 25,
    'batch_size': 128,
    'lr': 1e-3,
    'physics_lr': 5e-3,
    'patience': 5,
    'hidden': 64,
    'dropout': 0.2,
    'winter_weight': 1.5,
    'w_1_24': 2.0,
    'w_25_72': 1.5,
    'w_73_240': 1.0,
    'max_train_batches': None,   # cap per epoch for quick runs
    # lstm_v2 realism penalties (per-forecast totals, metres)
    'dry_climb_weight': 2.0,
    'smooth_weight': 0.5,
    'mono_weight': 5.0,
    'recession_weight': 1.0,
    # hybrid
    'per_sample_params': False,
    'smooth_rating': False,
    'station_weights': False,
    'rating_smooth_weight': 1e-3,
    'flow_aux_weight': 0.0,
    # Exponential moving average of weights (0 = off); validation and the
    # saved model use the averaged weights, which damps epoch-to-epoch noise.
    'ema_decay': 0.0,
    # lstm_v2: feed stage-1 predicted Farmoor flow to the decoder
    'use_pred_flow': False,
    'seed': 0,
}


def horizon_weights(cfg: Dict, device) -> torch.Tensor:
    w = torch.full((HORIZON,), float(cfg['w_73_240']), device=device)
    w[:72] = float(cfg['w_25_72'])
    w[:24] = float(cfg['w_1_24'])
    return w


def dry_mask(rain_past72: torch.Tensor, rain: torch.Tensor) -> torch.Tensor:
    """(B, H) True where mean-station rain over the preceding 72h < 2mm."""
    series = torch.cat([rain_past72, rain], dim=1)
    csum = F.pad(torch.cumsum(series, 1), (1, 0))
    n_past = rain_past72.shape[1]
    end = torch.arange(n_past + 1, n_past + rain.shape[1] + 1, device=rain.device)
    start = torch.clamp(end - DRY_LOOKBACK_H, min=0)
    total = csum[:, end] - csum[:, start]
    return total < DRY_RAIN_MM


class LSTMv2(torch.nn.Module):
    """HourlyDecoderModel with delta outputs and a log-rain decoder input.

    With use_flow, the decoder also sees stage-1 *predicted* Farmoor log-flow,
    in training as well as at forecast time (the September model trained on
    observed flow but forecast with predicted flow).
    """

    architecture = 'lstm_v2'

    def __init__(self, n_features: int, hidden_sizes=(128, 64), dropout: float = 0.2,
                 use_flow: bool = False):
        super().__init__()
        self.use_flow = use_flow
        self.core = HourlyDecoderModel(
            input_size=n_features, decoder_input_size=3 if use_flow else 2,
            hidden_sizes=list(hidden_sizes), dropout_rate=dropout, horizon=HORIZON,
        )

    def forward(self, x, rain, season, rain24, rain168, d0, pred_flow_log=None):
        parts = [torch.log1p(rain.clamp(min=0)), season]
        if self.use_flow:
            parts.append((pred_flow_log - 2.5) / 1.5)
        return d0[:, None] + self.core(x, torch.stack(parts, dim=-1))


def build_model(family: str, data: LocationData, cfg: Dict):
    n_feat = data.enc.shape[1]
    if family == 'hybrid':
        d = data.diff[np.isfinite(data.diff)]
        d_lo = float(d.min()) - 0.05
        d_hi = float(d.max()) + 0.3
        model = ReservoirHybrid(
            n_feat, d_lo=d_lo, d_hi=d_hi, hidden=cfg['hidden'], dropout=cfg['dropout'],
            per_sample_params=cfg['per_sample_params'],
            smooth_rating=cfg['smooth_rating'],
            n_stations=len(data.station_names) if cfg['station_weights'] else 0,
        )
        # Start the rating curve from the observed Farmoor-flow → D relation
        # (training period only), so latent Q starts out meaning real flow.
        before = np.asarray(data.index < pd.Timestamp(TRAIN_END, tz=data.index.tz))
        ok = before & np.isfinite(data.diff) & np.isfinite(data.flow)
        z_obs = np.log(data.flow[ok] / M3S_PER_MM_H + model.q_floor)
        model.rating.init_from_data(z_obs, data.diff[ok])
        return model
    if family == 'lstm_v2':
        return LSTMv2(n_feat, hidden_sizes=(128, cfg['hidden']), dropout=cfg['dropout'],
                      use_flow=cfg['use_pred_flow'])
    raise ValueError(family)


def _forward(model, b):
    if isinstance(model, ReservoirHybrid) and model.n_stations:
        return model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'],
                     rain_st=b['rain_st'], rain_st_ok=b['rain_st_ok'])
    if isinstance(model, LSTMv2) and model.use_flow:
        return model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'],
                     pred_flow_log=b['pred_flow_log'])
    return model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'])


def compute_loss(model, family: str, b: Dict, cfg: Dict, hw: torch.Tensor, train: bool):
    q = None
    if family == 'hybrid' and cfg['flow_aux_weight'] > 0:
        pred, q = model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0'], return_q=True,
                        rain_st=b.get('rain_st'), rain_st_ok=b.get('rain_st_ok'))
    else:
        pred = _forward(model, b)
    mask = b['y_ok'].float()
    w = hw[None, :] * mask
    w = w * torch.where(b['winter'], cfg['winter_weight'], 1.0)[:, None]
    err = (pred - b['y']).abs()
    mae = (err * w).sum() / w.sum().clamp(min=1)
    parts = {'mae': mae.detach()}
    loss = mae
    if family == 'lstm_v2' and train:
        full = torch.cat([b['d0'][:, None], pred], 1)
        step = full[:, 1:] - full[:, :-1]
        dry = dry_mask(b['rain_past72'], b['rain']).float()
        dry_climb = (F.relu(step) * dry).sum(1).mean()
        curv = (step[:, 1:] - step[:, :-1]).abs().sum(1).mean()
        rec = F.relu(-step - MAX_DROP).sum(1).mean()
        loss = loss + cfg['dry_climb_weight'] * dry_climb + cfg['smooth_weight'] * curv \
            + cfg['recession_weight'] * rec
        parts.update(dry_climb=dry_climb.detach(), curv=curv.detach(), rec=rec.detach())
        if cfg['mono_weight'] > 0:
            factor = 1.25 + 0.75 * torch.rand(pred.shape[0], 1, device=pred.device)
            more = dict(b, rain=b['rain'] * factor)
            pred_more = _forward(model, more)
            mono = F.relu(pred - pred_more).mean(1).mean()
            loss = loss + cfg['mono_weight'] * mono
            parts['mono'] = mono.detach()
    if q is not None:
        m = b['flow_ok'].float()
        z_obs = torch.log(b['flow'] / M3S_PER_MM_H + model.q_floor)
        aux = ((torch.log(q + model.q_floor) - z_obs).abs() * m).sum() / m.sum().clamp(min=1)
        loss = loss + cfg['flow_aux_weight'] * aux
        parts['flow_aux'] = aux.detach()
    if family == 'hybrid' and train:
        slopes = model.rating.slopes()
        rough = ((slopes[1:] - slopes[:-1]) ** 2).sum()
        loss = loss + cfg['rating_smooth_weight'] * rough
    return loss, pred, parts


def train_candidate(
    family: str,
    data: LocationData,
    cfg_override: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    cfg = {**DEFAULTS, **(cfg_override or {})}
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    device = device or torch.device('cpu')

    scaler = Standardiser.fit(data.enc[np.unique(np.concatenate(
        [data.train_t[:, None] + np.arange(-99, 1)[None, :]]).ravel())])
    sampler = WindowSampler(data, scaler, device)
    model = build_model(family, data, cfg).to(device)
    hw = horizon_weights(cfg, device)

    phys = [p for n, p in model.named_parameters() if n.startswith(('params_raw', 'split_logits', 'rating'))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(('params_raw', 'split_logits', 'rating'))]
    groups = [{'params': rest, 'lr': cfg['lr']}]
    if phys:
        groups.append({'params': phys, 'lr': cfg['physics_lr']})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    ema = copy.deepcopy(model) if cfg['ema_decay'] > 0 else None
    scored = ema if ema is not None else model

    history = []
    best = (math.inf, None, -1)
    bad = 0
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
            loss, _, parts = compute_loss(model, family, b, cfg, hw, train=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(cfg['ema_decay']).add_(pm.detach(), alpha=1 - cfg['ema_decay'])
            tr.append({k: float(v) for k, v in parts.items()} | {'loss': float(loss)})
        val = evaluate_windows(scored, family, sampler, data.val_t, cfg, hw)
        sched.step(val['mae'])
        rec = {'epoch': epoch + 1, 'secs': time.time() - t_start,
               **{f'train_{k}': float(np.mean([r[k] for r in tr])) for k in tr[0]},
               **{f'val_{k}': v for k, v in val.items()}}
        history.append(rec)
        if verbose:
            extra = ''
            if family == 'hybrid':
                extra = '  ' + ' '.join(f"{k}={v:.3g}" for k, v in scored.describe().items())
            print(f"epoch {epoch + 1:2d} ({rec['secs']:.0f}s) train_mae={rec['train_mae']:.4f} "
                  f"val_mae={val['mae']:.4f} [1-24h {val['mae_1_24h']:.4f} 25-72h {val['mae_25_72h']:.4f} "
                  f"73-240h {val['mae_73_240h']:.4f}]{extra}", flush=True)
        if val['mae'] < best[0] - 1e-5:
            best = (val['mae'], copy.deepcopy(scored.state_dict()), epoch + 1)
            bad = 0
        else:
            bad += 1
            if bad >= cfg['patience']:
                if verbose:
                    print(f"early stop; best epoch {best[2]} val_mae={best[0]:.4f}")
                break
    model.load_state_dict(best[1])
    model.eval()
    return model, scaler, cfg, history


@torch.no_grad()
def evaluate_windows(model, family, sampler, t_idx, cfg, hw, batch_size: int = 512) -> Dict[str, float]:
    """Unweighted masked MAE on validation windows, overall and by lead."""
    model.eval()
    sums = torch.zeros(HORIZON, device=sampler.device)
    counts = torch.zeros(HORIZON, device=sampler.device)
    for i in range(0, len(t_idx), batch_size):
        b = sampler.batch(t_idx[i:i + batch_size])
        pred = _forward(model, b)
        m = b['y_ok'].float()
        sums += ((pred - b['y']).abs() * m).sum(0)
        counts += m.sum(0)
    per_lead = (sums / counts.clamp(min=1)).cpu().numpy()
    return {
        'mae': float(per_lead.mean()),
        'mae_1_24h': float(per_lead[:24].mean()),
        'mae_25_72h': float(per_lead[24:72].mean()),
        'mae_73_240h': float(per_lead[72:].mean()),
    }


def save_candidate(model, scaler: Standardiser, cfg: Dict, history, family: str, name: str,
                   location: str, data: LocationData, models_dir: Path) -> Path:
    stem = f'redesign_{name}_{location}'
    if 'latest' in stem or 'experiment_2026_09' in stem:
        raise ValueError(f'refusing to overwrite a live model name: {stem}')
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, models_dir / f'{stem}.pt')
    meta = {
        'family': family, 'name': name, 'location': location, 'cfg': cfg,
        'scaler': scaler.state(), 'enc_cols': data.enc_cols, 'history': history,
        'n_features': data.enc.shape[1],
        'station_names': getattr(data, 'station_names', None),
    }
    if family == 'hybrid' and model.n_stations:
        meta['station_weights'] = dict(zip(data.station_names,
                                           torch.softmax(model.station_logits, 0).tolist()))
    if family == 'hybrid':
        meta['d_lo'] = model.rating.d_lo
        meta['d_hi'] = float(model.rating.d_knots()[-1])
        meta['physics'] = model.describe()
    with open(models_dir / f'{stem}.pkl', 'wb') as f:
        pickle.dump(meta, f)
    return models_dir / f'{stem}.pt'


def load_candidate(name: str, location: str, models_dir: Path, device=None):
    device = device or torch.device('cpu')
    stem = f'redesign_{name}_{location}'
    with open(models_dir / f'{stem}.pkl', 'rb') as f:
        meta = pickle.load(f)
    cfg = meta['cfg']
    if meta['family'] == 'hybrid':
        model = ReservoirHybrid(
            meta['n_features'], d_lo=meta['d_lo'], d_hi=meta['d_hi'], hidden=cfg['hidden'],
            dropout=cfg['dropout'], per_sample_params=cfg['per_sample_params'],
            smooth_rating=cfg.get('smooth_rating', False),
            n_stations=len(meta.get('station_names') or []) if cfg.get('station_weights') else 0,
        )
    else:
        model = LSTMv2(meta['n_features'], hidden_sizes=(128, cfg['hidden']), dropout=cfg['dropout'],
                       use_flow=cfg.get('use_pred_flow', False))
    model.load_state_dict(torch.load(models_dir / f'{stem}.pt', map_location=device))
    model.to(device).eval()
    scaler = Standardiser(meta['scaler']['mean'], meta['scaler']['std'])
    return model, scaler, meta
