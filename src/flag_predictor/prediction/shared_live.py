"""
Live forecasts with the shared catchment model (models/shared_catchment.py), for the preview page.

The model needs the recent past of all three stretches at once. The per-stretch live frames are
registered by hybrid_live.forecast_with_uncertainty as the main forecast runs (hybrid_live.
register_frame); generate_all_location_figures runs the shared trial after every stretch's main
forecast, so all three are there. Each stretch's forecast starts from its own last reading (t0);
the other stretches' data are used up to that hour, and a stretch with no recent reading is
treated as unavailable, exactly as in training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..models.physics_data import HORIZON, SEQ_LEN, doy_phase
from ..models.physics_models import M3S_PER_MM_H
from ..models.shared_catchment import (ALL_GAUGES, CHECKPOINT_FOR, STRETCHES, SUBCATCHMENTS, joint_features, load,
                                       shared_mask, subcatchment_rain)

SHARED_MODELS = ['hybrid_shared'] + [f'hybrid_shared_s{i}' for i in range(1, 5)]
TAG = 'hybrid_shared'
_cache: Dict[str, object] = {}


def names_available(models_dir: Path, names: Optional[List[str]] = None) -> List[str]:
    return [n for n in (names or SHARED_MODELS) if (Path(models_dir) / f'shared_{n}.pt').exists()]


def _models(names: List[str], models_dir: Path, location: str):
    key = ('models', tuple(names), CHECKPOINT_FOR[location])
    if key not in _cache:
        _cache[key] = [load(n, models_dir, location) for n in names]
    return _cache[key]


def gauge_rain(grid: pd.DatetimeIndex, project_root: Path) -> pd.DataFrame:
    """Hourly rain at every gauge (short names) over grid: the EA API where it has the hour, else the CSVs."""
    from .hybrid_live import _api_rain, _gauge_csv_rain
    csv = _gauge_csv_rain(project_root).reindex(index=grid, columns=ALL_GAUGES)
    api = _api_rain()
    return api.reindex(index=grid, columns=ALL_GAUGES).combine_first(csv) if len(api) else csv


def encoder_input(meta: dict, scaler, frames: Dict[str, pd.DataFrame], rain: pd.DataFrame) -> np.ndarray:
    """(SEQ_LEN, F) scaled encoder window ending at the grid's last hour, as in training
    (unavailable stretches' differential features zeroed)."""
    enc = joint_features(frames, rain, meta['diff_cols']).reindex(columns=meta['enc_cols'])
    raw = enc.to_numpy(dtype=np.float32)[-SEQ_LEN:]
    z = scaler.transform(raw)
    shared = shared_mask(meta['enc_cols'], meta['diff_cols'])
    for loc in STRETCHES:
        avail = raw[:, meta['enc_cols'].index(f'{loc}:available')] > 0.5
        cols = [i for i, c in enumerate(meta['enc_cols']) if c.startswith(f'{loc}:')
                and not c.endswith(':available') and not shared[i]]
        z[np.ix_(~avail, cols)] = 0.0
    return z


def stretch_frames(location: str, registered: Dict[str, pd.DataFrame], rain: pd.DataFrame,
                   gauges: Dict[str, List[str]], t0: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """The frames the encoder saw in training: each stretch's differential (up to t0; missing if it has
    no reading within 3 h), the gauges it was trained with, and Farmoor flow."""
    grid = rain.index
    flow = registered[location]['flow_m3s_Farmoor'].reindex(grid) if 'flow_m3s_Farmoor' in registered[location] \
        else pd.Series(np.nan, index=grid)
    out = {}
    for loc in STRETCHES:
        f = registered.get(loc)
        diff = (f['differential'].reindex(grid) if f is not None else pd.Series(np.nan, index=grid))
        diff = diff.loc[:t0].reindex(grid)
        last = diff.last_valid_index()
        if last is not None and t0 - last <= pd.Timedelta(hours=3):
            diff = diff.ffill(limit=3)
        frame = pd.concat([diff.rename('differential'), rain.reindex(columns=gauges[loc])], axis=1)
        frame['flow_m3s_Farmoor'] = flow
        out[loc] = frame
    return out


def forecast(location: str, frame: pd.DataFrame, registered: Dict[str, pd.DataFrame], forecast_rain: pd.DataFrame,
             n_members: int, project_root: Path, models_dir: Path, names: Optional[List[str]] = None):
    """(members (241, M) DataFrame, member mean-station rain (240, M), states dict, physics dict, names)."""
    names = names_available(models_dir, names)
    if not names:
        raise FileNotFoundError('no shared catchment models')
    models = _models(names, models_dir, location)
    j = STRETCHES.index(location)
    t0 = frame.index[-1]
    d_t0 = float(frame['differential'].iloc[-1])
    if not np.isfinite(d_t0):
        raise ValueError(f'no differential at t0 for {location}')
    grid = pd.date_range(t0 - pd.Timedelta(days=45), t0, freq='1h')
    rain_hist = gauge_rain(grid, project_root)
    gauges = models[0][2]['gauges']
    frames = stretch_frames(location, registered, rain_hist, gauges, t0)
    sr_hist = subcatchment_rain(rain_hist).to_numpy(dtype=np.float32)
    rain24 = sr_hist[-24:].sum(0)
    rain168 = sr_hist[-168:].sum(0)
    d0 = np.array([np.nan_to_num(float(frames[loc]['differential'].iloc[-1]), nan=0.0) for loc in STRETCHES],
                  dtype=np.float32)
    d0[j] = d_t0
    fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
    members, mean_rain = [], {}
    sub_rain = []
    for i in range(n_members):
        cols = {f'{g}_member_{i}': g for g in ALL_GAUGES if f'{g}_member_{i}' in forecast_rain.columns}
        if not cols:
            continue
        fr = forecast_rain[list(cols)].rename(columns=cols).reindex(fut_idx).fillna(0.0)
        sub_rain.append(subcatchment_rain(fr).to_numpy(dtype=np.float32))
        loc_g = [g for g in gauges[location] if g in fr.columns]
        mean_rain[f'member_{i}'] = fr[loc_g].clip(lower=0).mean(axis=1).to_numpy()
        members.append(f'member_{i}')
    if not members:
        raise RuntimeError('no forecast-rain members for the shared model')
    M = len(members)
    rain_t = torch.from_numpy(np.stack(sub_rain))                                   # (M, H, K)
    season = torch.from_numpy(doy_phase(fut_idx))[None].expand(M, -1)
    outs, qs, sts, physics = [], [], [], []
    with torch.no_grad():
        for model, scaler, meta in models:
            x = torch.from_numpy(encoder_input(meta, scaler, frames, rain_hist))[None].expand(M, -1, -1)
            d, q, st = model(x, rain_t, season, torch.from_numpy(rain24)[None].expand(M, -1),
                             torch.from_numpy(rain168)[None].expand(M, -1),
                             torch.from_numpy(d0)[None].expand(M, -1), return_states=True)
            outs.append(d[:, :, j].numpy())
            qs.append(q[:, :, j].numpy())
            sts.append(st.numpy())
            physics.append(model.describe())
    pred = np.mean(outs, axis=0)                                                    # (M, H)
    index = pd.date_range(t0, periods=HORIZON + 1, freq='1h')
    members_df = pd.DataFrame(np.concatenate([np.full((M, 1), d_t0), pred], 1).T, index=index, columns=members)
    rain_df = pd.DataFrame(mean_rain, index=fut_idx)
    state = {'q': np.mean(qs, axis=0), 'st': np.mean(sts, axis=0), 'stretch': j}
    phys = {k: float(np.mean([p[k] for p in physics if k in p])) for k in physics[0]}
    return members_df, rain_df, state, phys, names


def state_payload(location: str, index: pd.DatetimeIndex, state: dict, physics: dict, every_h: int = 3) -> dict:
    """The preview's 'under the hood' stores, in the per-stretch model's format: soil wetness and the
    quick/medium stores of the sub-catchments feeding this stretch (contribution-weighted), its own
    groundwater and channel stores, and its outflow; plus each sub-catchment's soil wetness."""
    K, L = len(SUBCATCHMENTS), len(STRETCHES)
    j = state['stretch']
    st = state['st']                                    # (M, 241, 3K + 2L)
    loc = STRETCHES[j]
    w = np.array([physics.get(f'contrib_{loc}_{k}', 0.0) for k in SUBCATCHMENTS])
    wn = w / w.sum()
    soil = (st[:, :, :K] * wn).sum(-1)
    quick = (st[:, :, K:2 * K] * w).sum(-1)
    medium = (st[:, :, 2 * K:3 * K] * w).sum(-1)
    slow = st[:, :, 3 * K + j]
    channel = st[:, :, 3 * K + L + j]
    q = state['q'] * M3S_PER_MM_H
    flow = np.concatenate([q[:, :1], q], axis=1)
    keep = np.arange(0, st.shape[1], every_h)
    series = {}
    for name, v in (('soil', soil * 100), ('quick', quick), ('medium', medium), ('slow', slow),
                    ('channel', channel), ('flow', flow)):
        v = v[:, keep]
        series[name] = {'mean': np.round(v.mean(0), 2).tolist(),
                        'p10': np.round(np.percentile(v, 10, axis=0), 2).tolist(),
                        'p90': np.round(np.percentile(v, 90, axis=0), 2).tolist()}
    now = {n: series[n]['mean'][0] for n in series}
    subs = {k: round(float(st[:, 0, i].mean() * 100), 1) for i, k in enumerate(SUBCATCHMENTS) if w[i] > 0}
    params = {k: round(v, 4) for k, v in physics.items()
              if k.endswith(f'_{loc}') or any(k.endswith(f'_{s}') for s in SUBCATCHMENTS if w[SUBCATCHMENTS.index(s)] > 0)}
    # The page's store notes read the per-stretch model's names: contribution-weighted means here.
    for name in ('k_quick', 'k_medium', 'beta', 'smax'):
        params[name] = round(float(sum(wn[i] * physics[f'{name}_{k}'] for i, k in enumerate(SUBCATCHMENTS))), 4)
    params['k_slow'] = round(physics[f'k_slow_{loc}'], 4)
    params['k_channel'] = round(physics[f'k_channel_{loc}'], 4)
    print(f"  [shared] {loc}: soil {now['soil']:.0f}% wet (sub-catchments {subs}); flow {now['flow']:.1f} m³/s")
    return {'times': [t.tz_convert('UTC').tz_localize(None).isoformat() for t in index[keep]],
            'now': now, 'series': series, 'params': params, 'every_h': every_h,
            'subcatchments': subs, 'variant': TAG, 'shared': True}
