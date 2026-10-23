#!/usr/bin/env python
"""
Build the model-error library used for the forecast's total-uncertainty band.

Re-runs the deployed ensemble every 6 hours over the periods it never trained
on (2023 validation year; Nov 2024 - Jan 2026 test window) with observed
rain, and stores each run's 10-day error trajectory with a description of the
situation it started in (see src/flag_predictor/prediction/model_error.py).

    python build_error_library.py --family hybrid --tag hybrid_c1_ps \
        --names hybrid_c1_ps,hybrid_c1_ps_s1,hybrid_c1_ps_s2,hybrid_c1_ps_s3,hybrid_c1_ps_s4

Cross-validation: errors from one period build the bands for the other.
That picks the number of neighbours k and a per-lead inflation factor that
makes 80% of held-out observations fall inside the 10-90% band (the band is
otherwise a little too narrow, because each period alone has seen fewer
kinds of weather than the two together).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.models.physics_data import HORIZON, SEQ_LEN, WindowSampler, load_location_data  # noqa: E402
from flag_predictor.prediction.model_error import (  # noqa: E402
    ErrorLibrary, coverage, crps_samples, start_features, total_uncertainty,
)

PERIODS = {
    'val2023': ('2023-01-01', '2023-12-21'),
    'test': ('2024-11-01', '2026-01-10'),
}
LEAD_BUCKETS = {'1-6h': (1, 6), '7-24h': (7, 24), '25-72h': (25, 72), '73-168h': (73, 168), '169-240h': (169, 240)}


def start_positions(data, stride_h: int = 6) -> dict:
    enc_ok = np.isfinite(data.enc).all(axis=1)
    ok_count = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[: len(enc_ok)]
    fut_count = np.convolve(np.isfinite(data.diff).astype(np.int32), np.ones(HORIZON, dtype=np.int32), 'full')
    idx = data.index
    out = {}
    for name, (lo, hi) in PERIODS.items():
        lo_t, hi_t = pd.Timestamp(lo, tz=idx.tz), pd.Timestamp(hi, tz=idx.tz)
        cand = np.where((idx >= lo_t) & (idx < hi_t) & (idx.hour % stride_h == 0))[0]
        cand = cand[(cand >= SEQ_LEN) & (cand < len(idx) - HORIZON - 1)]
        good = (ok_count[cand] == SEQ_LEN) & np.isfinite(data.diff[cand]) & (fut_count[cand + HORIZON] >= 200)
        out[name] = cand[good]
    return out


@torch.no_grad()
def predict_hybrid(names, location, data, t_idx) -> np.ndarray:
    from flag_predictor.models.physics_train import load_candidate
    preds = []
    for n in names:
        model, scaler, _ = load_candidate(n, location, ROOT / 'models')
        sampler = WindowSampler(data, scaler, torch.device('cpu'))
        out = []
        for i in range(0, len(t_idx), 256):
            b = sampler.batch(t_idx[i:i + 256])
            out.append(model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0']).numpy())
        preds.append(np.concatenate(out))
    return np.mean(preds, axis=0)


@torch.no_grad()
def predict_continuous(names, location, data, t_idx) -> np.ndarray:
    from flag_predictor.models.continuous_train import (
        ContinuousSampler, _forward, backtest_state_sims, load_forecast_model,
    )
    from flag_predictor.models.continuous_hybrid import fill_et0
    sims, et0 = backtest_state_sims(names, location, ROOT, ROOT / 'models')
    et0_frame = fill_et0(et0, data.index) if et0 is not None else None
    preds = []
    for n in names:
        model, scaler, meta = load_forecast_model(n, location, ROOT / 'models')
        caps = (meta['cfg'].get('quick_cap_h', 24), meta['cfg'].get('medium_cap_h', 168))
        sampler = ContinuousSampler(data, scaler, sims[n], et0_frame, torch.device('cpu'), cap_hours=caps)
        out = []
        for i in range(0, len(t_idx), 256):
            out.append(_forward(model, sampler.batch(t_idx[i:i + 256])).numpy())
        preds.append(np.concatenate(out))
    return np.mean(preds, axis=0)


def predict_simfeat(names, location, data, t_idx, core_name: str) -> np.ndarray:
    """Hybrid models trained with continuous-simulation states as extra encoder inputs."""
    import copy
    from train_continuous_models import sim_feature_frame
    from flag_predictor.models.continuous_train import SIM_FEATURES, load_core, long_rain_for_location
    from flag_predictor.models.continuous_hybrid import ReservoirCore  # noqa: F401
    core = load_core(core_name, ROOT / 'models')
    et0 = None
    if core.et_mode == 'pet':
        from flag_predictor.models.continuous_train import location_et0
        et0 = location_et0(ROOT, core)
    d2 = copy.copy(data)
    rec = long_rain_for_location(ROOT, location, data, et0)
    d2.enc = np.concatenate([data.enc, sim_feature_frame(core, rec, data)], axis=1)
    d2.enc_cols = list(data.enc_cols) + SIM_FEATURES
    return predict_hybrid(names, location, d2, t_idx)


def features_for(data, t_idx) -> np.ndarray:
    rows = []
    for t in t_idx:
        d24 = data.diff[t - 24] if t >= 24 else np.nan
        rows.append(start_features(float(data.diff[t]), float(d24), data.rain[t + 1:t + 1 + HORIZON],
                                   float(data.rain168[t]), data.index[t]))
    return np.stack(rows)


K_GRID = (40, 100, 200)
C_GRID = np.round(np.arange(0.6, 3.01, 0.05), 2)


def cv_samples(lib: ErrorLibrary, feats, is_val, k: int, n: int = 60, seed: int = 0):
    """Residual samples for every start, drawn only from the *other* period's errors."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(feats), n, lib.residuals.shape[1]), np.float32)
    for fit_val in (True, False):
        fit = is_val if fit_val else ~is_val
        sub = lib.subset(fit)
        held = np.where(~fit)[0]
        for i0 in range(0, len(held), 200):
            idx = held[i0:i0 + 200]
            out[idx] = sub.sample(feats[idx], n=n, k=k, rng=rng, inflate=False)
    return out


def fit_inflation(samples, err_obs) -> np.ndarray:
    """Per-lead factor c so that observed errors fall inside c x the 10-90% sample band 80% of the time."""
    q10 = np.percentile(samples, 10, axis=1)        # (N, H)
    q90 = np.percentile(samples, 90, axis=1)
    ok = np.isfinite(err_obs)
    cov = np.stack([np.nanmean(np.where(ok, (err_obs >= c * q10) & (err_obs <= c * q90), np.nan), axis=0)
                    for c in C_GRID])                     # (C, H)
    c = C_GRID[np.argmin(np.abs(cov - 0.8), axis=0)].astype(float)
    # Smooth over lead time (errors grow smoothly), keep it monotone non-decreasing after day 1.
    kernel = np.ones(25) / 25
    c = np.convolve(np.pad(c, 12, mode='edge'), kernel, mode='valid')
    c[24:] = np.maximum.accumulate(c[24:])
    return c.astype(np.float32)


def cv_scores(samples, preds, obs, c, chunk: int = 300) -> dict:
    """CRPS / coverage of held-out bands (point = clairvoyant forecast) by lead bucket."""
    o = obs[:, 1:]
    crps = np.empty_like(o, dtype=np.float64)
    cov = np.empty_like(o, dtype=np.float64)
    for i in range(0, len(o), chunk):
        s = np.sort(preds[i:i + chunk, None, :] + samples[i:i + chunk] * c[None, None, :], axis=1)
        S = s.shape[1]
        oo = o[i:i + chunk, None, :]
        term1 = np.mean(np.abs(s - oo), axis=1)
        w = (2 * np.arange(1, S + 1) - S - 1)[None, :, None]
        term2 = (2.0 / (S * S)) * np.sum(w * s, axis=1)
        crps[i:i + chunk] = term1 - 0.5 * term2
        lo = s[:, int(round(0.1 * (S - 1))), :]
        hi = s[:, int(round(0.9 * (S - 1))), :]
        cov[i:i + chunk] = ((oo[:, 0] >= lo) & (oo[:, 0] <= hi)).astype(float)
    bad = ~np.isfinite(o)
    crps[bad] = np.nan
    cov[bad] = np.nan
    mae = np.abs(preds - o)
    out = {}
    for name, (lo_h, hi_h) in LEAD_BUCKETS.items():
        sl = slice(lo_h - 1, hi_h)
        out[name] = {'in_band_10_90': float(np.nanmean(cov[:, sl])), 'crps': float(np.nanmean(crps[:, sl])),
                     'mae_point': float(np.nanmean(mae[:, sl]))}
    out['all'] = {'crps': float(np.nanmean(crps)), 'mae_point': float(np.nanmean(mae)),
                  'in_band_10_90': float(np.nanmean(cov))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', choices=['hybrid', 'continuous', 'simfeat'], required=True)
    ap.add_argument('--core', default='cs_cos', help='simfeat: the core whose simulation feeds the encoder')
    ap.add_argument('--tag', required=True, help='library name, e.g. the deployed model name')
    ap.add_argument('--names', required=True, help='comma-separated model names (seeds) to average')
    ap.add_argument('--locations', default='isis,godstow,wallingford')
    ap.add_argument('--k-grid', default=','.join(map(str, K_GRID)), help='neighbour counts to try')
    ap.add_argument('--no-save', action='store_true', help='report only')
    ap.add_argument('--extra', default=None, choices=['none', 'all', 'soil', 'soil_rise', 'soil_rise_han'],
                    help='models with extra soil-moisture / upstream-level inputs (train_extra_models.py --inputs)')
    args = ap.parse_args()
    k_grid = [int(k) for k in args.k_grid.split(',')]
    names = args.names.split(',')
    report = {}
    for loc in args.locations.split(','):
        data = load_location_data(loc, ROOT)
        starts = start_positions(data)
        t_all = np.concatenate([starts['val2023'], starts['test']])
        if args.family == 'simfeat':
            preds = predict_simfeat(names, loc, data, t_all, args.core)
        else:
            predict = predict_hybrid if args.family == 'hybrid' else predict_continuous
            if args.extra:
                from train_extra_models import data_with_extra
                preds = predict(names, loc, data_with_extra(loc, args.extra), t_all)
            else:
                preds = predict(names, loc, data, t_all)
        obs = np.stack([data.diff[t:t + HORIZON + 1] for t in t_all])
        feats = features_for(data, t_all)
        resid = obs[:, 1:] - preds
        lib = ErrorLibrary.build(loc, args.tag, data.index[t_all].tz_convert('UTC').tz_localize(None).to_numpy(),
                                 feats, resid)
        is_val = np.arange(len(t_all)) < len(starts['val2023'])
        best = None
        rep = {'n_starts': {k: int(len(v)) for k, v in starts.items()}, 'by_k': {}}
        for k in k_grid:
            samp = cv_samples(lib, feats, is_val, k)
            c = fit_inflation(samp, resid)
            raw = cv_scores(samp, preds, obs, np.ones_like(c))
            cal = cv_scores(samp, preds, obs, c)
            rep['by_k'][k] = {'raw': raw, 'calibrated': cal, 'inflation_24h': float(c[23]),
                              'inflation_240h': float(c[-1])}
            print(f"  {loc} k={k:3d}: CRPS raw {raw['all']['crps']:.4f} calibrated {cal['all']['crps']:.4f} "
                  f"(point MAE {raw['all']['mae_point']:.4f}); in-band raw {raw['all']['in_band_10_90']:.0%}; "
                  f"inflation 24h {c[23]:.2f} 240h {c[-1]:.2f}", flush=True)
            if best is None or cal['all']['crps'] < best[0]:
                best = (cal['all']['crps'], k, c, cal)
        lib.k, lib.inflation = best[1], best[2]
        path = ROOT / 'models' / f'error_library_{args.tag}_{loc}.npz'
        if not args.no_save:
            lib.save(path)
        rep['chosen_k'] = best[1]
        rep['chosen'] = best[3]
        rep['inflation'] = [round(float(x), 3) for x in best[2]]
        report[loc] = rep
        print(f'\n{loc}: {path.name} ({len(t_all)} starts, k={best[1]})')
        for b, v in best[3].items():
            print(f"    {b:>9}: in 10-90% band {v['in_band_10_90']:.0%}  CRPS {v['crps']:.4f}  "
                  f"(point MAE {v['mae_point']:.4f})")
    out = ROOT / 'figures' / f'error_library_{args.tag}_check.json'
    if not args.no_save:
        out.write_text(json.dumps(report, indent=1))
    print(f'\nreport: {out}')


if __name__ == '__main__':
    main()
