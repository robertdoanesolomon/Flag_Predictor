"""
September 2026 hourly-decoder training.

Kept separate from the June multi-horizon trainer so May/June weights still
load unchanged.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..config import (
    MODEL_CONFIG,
    PHYSICAL_CONSTRAINTS,
    SEPTEMBER_2026_TRAINING,
)
from .lstm import HourlyDecoderModel, get_device
from .training import save_model


def create_hourly_decoder_arrays(
    encoder_values: np.ndarray,
    decoder_values: np.ndarray,
    target_values: np.ndarray,
    index: pd.DatetimeIndex,
    sequence_length: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Sliding windows: encoder past, decoder future covariates, delta targets."""
    n = len(index)
    xs: List[np.ndarray] = []
    cs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    curs: List[float] = []
    times: List[pd.Timestamp] = []

    start = sequence_length - 1
    stop = n - horizon
    n_windows = max(0, (stop - start + stride - 1) // stride)
    print(f"Scanning {n_windows} candidate windows (n={n}, stride={stride})...")
    for i in range(start, stop, stride):
        x = encoder_values[i - sequence_length + 1 : i + 1]
        c = decoder_values[i + 1 : i + 1 + horizon]
        y_abs = target_values[i + 1 : i + 1 + horizon]
        cur = target_values[i]
        if not (
            np.isfinite(x).all()
            and np.isfinite(c).all()
            and np.isfinite(y_abs).all()
            and np.isfinite(cur)
        ):
            continue
        xs.append(x.astype(np.float32, copy=False))
        cs.append(c.astype(np.float32, copy=False))
        ys.append((y_abs - cur).astype(np.float32))
        curs.append(float(cur))
        times.append(index[i])

    if not xs:
        raise ValueError("No valid hourly-decoder sequences (check NaNs / date range).")

    return (
        np.stack(xs),
        np.stack(cs),
        np.stack(ys),
        np.asarray(curs, dtype=np.float32),
        pd.DatetimeIndex(times),
    )


def apply_event_sampling(
    X: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    cur: np.ndarray,
    times: pd.DatetimeIndex,
    target_kind: str,
    quiet_keep_frac: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Keep all event windows; subsample quiet ones."""
    rng = rng or np.random.default_rng(0)
    if target_kind == 'diff':
        is_event = (
            (np.abs(y).max(axis=1) > 0.05)
            | (cur > 0.30)
            | (np.abs(y[:, min(23, y.shape[1] - 1)]) > 0.03)
        )
    else:
        q80 = float(np.quantile(cur, 0.80)) if len(cur) else 0.0
        # log-flow deltas of 0.20 over 10 days are routine; require a real event.
        is_event = (np.abs(y).max(axis=1) > 0.40) | (cur > q80)

    event_idx = np.where(is_event)[0]
    quiet_idx = np.where(~is_event)[0]
    if len(quiet_idx):
        n_keep = max(1, int(quiet_keep_frac * len(quiet_idx)))
        n_keep = min(n_keep, len(quiet_idx))
        keep_quiet = rng.choice(quiet_idx, size=n_keep, replace=False)
        keep = np.concatenate([event_idx, keep_quiet])
    else:
        keep = event_idx
    keep.sort()
    return X[keep], C[keep], y[keep], cur[keep], times[keep]


def chronological_masks(times: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """Train: t < train_end and not val_year. Val: calendar val_year."""
    cfg = SEPTEMBER_2026_TRAINING
    ts = pd.DatetimeIndex(times)
    tz = ts.tz
    train_end = pd.Timestamp(cfg['train_end'], tz=tz)
    val_year = int(cfg['val_year'])
    is_val = np.asarray(ts.year == val_year, dtype=bool)
    is_train = np.asarray(ts < train_end, dtype=bool) & ~is_val
    if not is_train.any() or not is_val.any():
        raise ValueError(
            f"Empty train or val split (train={is_train.sum()} val={is_val.sum()} "
            f"range={ts.min()} → {ts.max()})"
        )
    return is_train, is_val


def _september_loss(
    outputs: torch.Tensor,
    batch_y: torch.Tensor,
    current: torch.Tensor,
    sample_weights: torch.Tensor,
    future_rain: torch.Tensor,
    target_kind: str,
    device: torch.device,
    cfg: Optional[Dict] = None,
) -> torch.Tensor:
    cfg = cfg or SEPTEMBER_2026_TRAINING
    err = (outputs - batch_y).abs()
    weights = torch.ones_like(err)

    if target_kind == 'diff':
        event = batch_y.abs() > 0.05
        rising = batch_y > 0.03
    else:
        event = batch_y.abs() > 0.15
        rising = batch_y > 0.05

    weights = torch.where(event, weights * cfg['event_weight'], weights)
    weights = torch.where(rising, weights * cfg['rising_weight'], weights)
    weights = weights * sample_weights.unsqueeze(1)

    H = err.shape[1]
    hw = torch.full((H,), float(cfg.get('w_72_240h', 1.0)), device=device)
    hw[: min(72, H)] = float(cfg.get('w_24_72h', 2.0))
    hw[: min(24, H)] = float(cfg.get('w_6_24h', 4.0))
    hw[: min(6, H)] = float(cfg.get('w_0_6h', 6.0))
    mae = (err * weights * hw.unsqueeze(0)).mean()

    rain_sum = future_rain.sum(dim=1).clamp(min=0)
    rain_scale = rain_sum / (rain_sum.mean() + 1e-3)
    rain_term = (
        torch.relu(-outputs[:, -1]) * rain_scale * cfg['rain_response_weight']
    ).mean()

    rec = torch.tensor(0.0, device=device)
    dry_rise = torch.tensor(0.0, device=device)
    dry_persist = torch.tensor(0.0, device=device)
    if target_kind == 'diff':
        abs_pred = current.unsqueeze(1) + outputs
        drops = abs_pred[:, :-1] - abs_pred[:, 1:]
        max_drop = PHYSICAL_CONSTRAINTS['max_recession_m_per_day'] / 24.0
        rec = (
            torch.clamp(drops - max_drop, min=0).mean()
            * cfg['recession_penalty_weight']
        )
        dry_h = future_rain < cfg['dry_hour_mm']
        dry_rise = (
            torch.relu(outputs) * dry_h.float() * cfg['dry_rise_weight']
        ).mean()
        # Persist only on dry hours so wet 0–24h rain response is not delayed.
        dry_persist = (
            outputs.abs() * dry_h.float() * cfg['dry_persist_weight']
        ).mean()

    return mae + rain_term + rec + dry_rise + dry_persist


def train_hourly_decoder(
    encoder_df: pd.DataFrame,
    decoder_cov: pd.DataFrame,
    target: pd.Series,
    target_kind: str,
    *,
    sequence_length: int = SEPTEMBER_2026_TRAINING['sequence_length'],
    horizon: int = SEPTEMBER_2026_TRAINING['horizon'],
    stride: int = SEPTEMBER_2026_TRAINING['stride'],
    epochs: int = SEPTEMBER_2026_TRAINING['epochs'],
    batch_size: int = SEPTEMBER_2026_TRAINING['batch_size'],
    learning_rate: float = SEPTEMBER_2026_TRAINING['learning_rate'],
    patience: int = SEPTEMBER_2026_TRAINING['patience'],
    hidden_sizes: Optional[List[int]] = None,
    dropout_rate: float = MODEL_CONFIG['dropout_rate'],
    max_grad_norm: float = 1.0,
    verbose: bool = True,
    cfg_override: Optional[Dict] = None,
) -> Tuple[HourlyDecoderModel, MinMaxScaler, MinMaxScaler, Dict]:
    """
    Train an HourlyDecoderModel.

    encoder_df: past features (unscaled), same index as target.
    decoder_cov: future covariates at each timestamp (unscaled).
    target: series to predict (log-flow or differential); outputs are deltas.
    """
    hidden_sizes = hidden_sizes or list(MODEL_CONFIG['hidden_sizes'])
    device = get_device()
    cfg = {**SEPTEMBER_2026_TRAINING, **(cfg_override or {})}

    aligned = encoder_df.join(decoder_cov.add_prefix('_dec_'), how='inner')
    aligned = aligned.join(target.rename('_target'), how='inner')
    aligned = aligned.sort_index()
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.ffill().bfill().fillna(0)
    aligned = aligned.loc[aligned.index < pd.Timestamp(cfg['train_end'], tz=aligned.index.tz) + pd.Timedelta(days=1)]

    enc_cols = list(encoder_df.columns)
    dec_cols = list(decoder_cov.columns)
    enc_vals = aligned[enc_cols].to_numpy(dtype=np.float32)
    dec_vals = aligned[[f'_dec_{c}' for c in dec_cols]].to_numpy(dtype=np.float32)
    y_vals = aligned['_target'].to_numpy(dtype=np.float32)

    X, C, y, cur, times = create_hourly_decoder_arrays(
        enc_vals, dec_vals, y_vals, aligned.index,
        sequence_length=sequence_length, horizon=horizon, stride=stride,
    )
    if verbose:
        print(f"Raw sequences: {len(times)}  ({times.min()} → {times.max()})")

    X, C, y, cur, times = apply_event_sampling(
        X, C, y, cur, times, target_kind, cfg['quiet_keep_frac']
    )
    if verbose:
        print(f"After event sampling: {len(times)}")

    is_train, is_val = chronological_masks(times)
    if verbose:
        print(f"Train {is_train.sum()} | Val {is_val.sum()} (val year {cfg['val_year']})")

    past_scaler = MinMaxScaler()
    n_feat = X.shape[2]
    past_scaler.fit(X[is_train].reshape(-1, n_feat))
    X = past_scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype(np.float32)

    dec_scaler = MinMaxScaler()
    n_cov = C.shape[2]
    rain_mm = C[:, :, 0].copy()  # unscaled catchment mm, for dry-weather loss
    dec_scaler.fit(C[is_train].reshape(-1, n_cov))
    C = dec_scaler.transform(C.reshape(-1, n_cov)).reshape(C.shape).astype(np.float32)

    months = pd.DatetimeIndex(times).month.to_numpy()
    season_w = np.where(
        np.isin(months, cfg['winter_months']),
        np.float32(cfg['winter_weight']),
        np.float32(1.0),
    )

    def _pack(mask):
        return (
            torch.from_numpy(X[mask]),
            torch.from_numpy(C[mask]),
            torch.from_numpy(y[mask]),
            torch.from_numpy(cur[mask]),
            torch.from_numpy(season_w[mask]),
            torch.from_numpy(rain_mm[mask]),  # unscaled catchment rain (mm)
        )

    Xtr, Ctr, ytr, curtr, wtr, raintr = _pack(is_train)
    Xva, Cva, yva, curva, wva, rainva = _pack(is_val)

    loader = DataLoader(
        TensorDataset(Xtr, Ctr, ytr, curtr, wtr, raintr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = HourlyDecoderModel(
        input_size=n_feat,
        decoder_input_size=n_cov,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        horizon=horizon,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val = float('inf')
    best_state = None
    patience_counter = 0

    if verbose:
        print(model)
        print(f"Decoder covariates: {dec_cols}")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        train_maes = []
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', disable=not verbose, leave=False, mininterval=2.0)
        for batch_X, batch_C, batch_y, batch_cur, batch_w, batch_rain in pbar:
            batch_X = batch_X.to(device)
            batch_C = batch_C.to(device)
            batch_y = batch_y.to(device)
            batch_cur = batch_cur.to(device)
            batch_w = batch_w.to(device)
            batch_rain = batch_rain.to(device)

            outputs = model(batch_X, batch_C)
            loss = _september_loss(
                outputs, batch_y, batch_cur, batch_w, batch_rain, target_kind, device, cfg
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())
            train_maes.append(torch.mean(torch.abs(outputs - batch_y)).item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            val_losses = []
            val_maes = []
            val_bs = max(32, min(256, len(Xva)))
            for i in range(0, len(Xva), val_bs):
                sl = slice(i, i + val_bs)
                bX = Xva[sl].to(device)
                bC = Cva[sl].to(device)
                by = yva[sl].to(device)
                bcur = curva[sl].to(device)
                bw = wva[sl].to(device)
                brain = rainva[sl].to(device)
                vout = model(bX, bC)
                vloss = _september_loss(
                    vout, by, bcur, bw, brain, target_kind, device, cfg
                )
                val_losses.append(vloss.item() * len(bX))
                val_maes.append(torch.mean(torch.abs(vout - by)).item() * len(bX))
            n_val = max(len(Xva), 1)
            val_loss_value = float(sum(val_losses) / n_val)
            val_mae = float(sum(val_maes) / n_val)

        avg_tr = float(np.mean(train_losses))
        avg_tr_mae = float(np.mean(train_maes))
        history['train_loss'].append(avg_tr)
        history['val_loss'].append(val_loss_value)
        history['train_mae'].append(avg_tr_mae)
        history['val_mae'].append(val_mae)
        scheduler.step(val_loss_value)

        if verbose:
            print(
                f"Epoch [{epoch+1}/{epochs}] ({time.time()-t0:.1f}s) - "
                f"Train {avg_tr:.5f}  Val {val_loss_value:.5f}  "
                f"Train MAE {avg_tr_mae:.5f}  Val MAE {val_mae:.5f}"
            )

        if val_loss_value < best_val:
            best_val = val_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print("✓ Restored best September weights")

    extra = {
        'architecture': 'hourly_decoder',
        'feature_columns': enc_cols,
        'decoder_columns': dec_cols,
        'decoder_scaler': dec_scaler,
        'target_kind': target_kind,
        'horizon': horizon,
        'sequence_length': sequence_length,
        'hidden_sizes': hidden_sizes,
        'dropout_rate': dropout_rate,
        'input_size': n_feat,
        'decoder_input_size': n_cov,
        'predicts_delta': True,
        'training_history': history,
        'horizons': list(range(1, horizon + 1)),
    }
    return model, past_scaler, dec_scaler, extra


# Re-export save_model for callers
__all__ = [
    'create_hourly_decoder_arrays',
    'apply_event_sampling',
    'chronological_masks',
    'train_hourly_decoder',
    'save_model',
]
