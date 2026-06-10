"""
Model training utilities.

Functions for training, saving, and loading the LSTM model.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .lstm import MultiHorizonLSTMModel, get_device
from ..config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    PHYSICAL_CONSTRAINTS,
    JUNE_2026_TRAINING,
)


def create_sequences(
    X: pd.DataFrame,
    y_multi: pd.DataFrame,
    sequence_length: int = 24
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Create sequences for multi-horizon LSTM training.
    
    Args:
        X: Feature DataFrame
        y_multi: Target DataFrame with multiple horizons (columns)
        sequence_length: Number of time steps to look back
        
    Returns:
        Tuple of (X_sequences, y_sequences, valid_indices)
    """
    X_values = X.values
    y_values = y_multi.values
    
    n_samples = len(X) - sequence_length + 1
    n_features = X_values.shape[1]
    n_horizons = y_values.shape[1]
    
    X_sequences = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)
    y_sequences = np.zeros((n_samples, n_horizons), dtype=np.float32)
    valid_indices = []
    
    for i in range(n_samples):
        X_sequences[i] = X_values[i:i+sequence_length]
        # Targets aligned with the LAST row of the input sequence (time t),
        # so target_{h} really is diff(t+h). The previous version took the
        # row AFTER the sequence end, introducing a 1-hour misalignment.
        y_sequences[i] = y_values[i+sequence_length-1]
        valid_indices.append(X.index[i+sequence_length-1])
    
    return X_sequences, y_sequences, valid_indices


def _compute_loss(
    outputs: torch.Tensor,
    batch_y: torch.Tensor,
    current: torch.Tensor,
    sample_weights: torch.Tensor,
    horizon_gaps: torch.Tensor,
    horizon_importance: torch.Tensor,
    device: torch.device,
    recession_penalty_weight: float = JUNE_2026_TRAINING['recession_penalty_weight'],
    max_recession_m_per_hour: float = PHYSICAL_CONSTRAINTS['max_recession_m_per_day'] / 24.0,
) -> torch.Tensor:
    """
    Compute weighted loss with custom penalties (June 2026, delta formulation).

    `outputs` and `batch_y` are DELTAS from the current differential
    (diff(t+h) - diff(t)); `current` is diff(t). Absolute levels are
    reconstructed for the flow-dependent weighting.

    Components:
    - Weighted MSE (high-flow / rising events weighted more, winter samples
      upweighted via `sample_weights`)
    - Early-rise / false-rise penalties
    - Persistence penalty (no spurious upticks in stable low flow)
    - Recession-rate penalty: predicted drops faster than the physical
      maximum (~3 inches/day) are penalized

    The old "extended anchor" and "continuity" losses are gone: the delta
    formulation anchors predictions to the current value by construction.
    """
    current_col = current.unsqueeze(1)            # (batch, 1)
    abs_y = current_col + batch_y                 # absolute future levels
    sample_w = sample_weights.unsqueeze(1)        # (batch, 1)

    # Base weights: penalize errors during high-flow events more
    weights = torch.where(abs_y > 0.3, 3.0, 1.0)

    # Also weight rising events (rise of >5cm from current)
    rising_mask = (batch_y > 0.05)
    weights = torch.where(rising_mask, weights * 1.5, weights)

    # Reduce weight for stable low-flow predictions
    low_stable_mask = (current_col < 0.2) & (abs_y[:, -1:] < 0.25)
    weights = torch.where(
        low_stable_mask.expand_as(weights),
        torch.clamp(weights, max=1.5),
        weights
    )

    # Seasonal (winter) upweighting
    weights = weights * sample_w

    # Weighted MSE loss (identical in delta and absolute space)
    mse_loss = (outputs - batch_y) ** 2
    weighted_loss = (mse_loss * weights * horizon_importance).mean()

    # Early rise detection loss: reality rises but prediction doesn't
    early_rise_loss = torch.tensor(0.0, device=device)
    actual_rising = (batch_y[:, -1] > 0.02)
    pred_rising = (outputs[:, -1] > 0.02)
    missed_rise_mask = actual_rising & ~pred_rising
    if missed_rise_mask.any():
        rise_error = torch.clamp(batch_y[:, -1] - outputs[:, -1], min=0)
        early_rise_loss = (rise_error * missed_rise_mask.float()).mean()

    # Penalize false early rises: model predicts a rise but reality stays flat
    false_rise_loss = torch.tensor(0.0, device=device)
    actual_flat = (batch_y[:, -1] <= 0.02)
    false_rise_mask = actual_flat & pred_rising
    if false_rise_mask.any():
        false_rise_penalty = torch.clamp(outputs[:, -1], min=0)
        false_rise_loss = (false_rise_penalty * false_rise_mask.float()).mean()

    # Persistence loss: in stable low flow, predicted deltas should stay ~0
    persistence_loss = torch.tensor(0.0, device=device)
    is_stable_low = (current < 0.2) & (abs_y[:, -1] < 0.25)
    if is_stable_low.any():
        pred_rise = torch.clamp(outputs - 0.02, min=0)
        persistence_loss = (pred_rise.mean(dim=1) * is_stable_low.float()).mean()

    # Recession-rate penalty: between consecutive horizons (including a
    # virtual point at h=0 where the delta is 0 by definition), the predicted
    # drop rate must not exceed the physical maximum.
    zeros = torch.zeros(outputs.shape[0], 1, device=device)
    deltas_full = torch.cat([zeros, outputs], dim=1)        # (batch, n_horizons+1)
    drop = deltas_full[:, :-1] - deltas_full[:, 1:]         # positive = falling
    drop_rate = drop / horizon_gaps                          # m per hour
    recession_violation = torch.clamp(drop_rate - max_recession_m_per_hour, min=0)
    recession_loss = recession_violation.mean()

    loss = (
        weighted_loss
        + 0.3 * early_rise_loss
        + 0.2 * false_rise_loss
        + 0.2 * persistence_loss
        + recession_penalty_weight * recession_loss
    )
    
    return loss


def train_model(
    X: pd.DataFrame,
    y_multi: pd.DataFrame,
    horizons: List[int],
    sequence_length: int = MODEL_CONFIG['sequence_length'],
    epochs: int = TRAINING_CONFIG['epochs'],
    batch_size: int = TRAINING_CONFIG['batch_size'],
    learning_rate: float = TRAINING_CONFIG['learning_rate'],
    patience: int = TRAINING_CONFIG['patience'],
    validation_split: float = TRAINING_CONFIG['validation_split'],
    max_grad_norm: float = TRAINING_CONFIG['max_grad_norm'],
    hidden_sizes: List[int] = MODEL_CONFIG['hidden_sizes'],
    dropout_rate: float = MODEL_CONFIG['dropout_rate'],
    verbose: bool = True
) -> Tuple[MultiHorizonLSTMModel, MinMaxScaler, Dict, int, List[int]]:
    """
    Train multi-horizon PyTorch LSTM model.
    
    Args:
        X: Feature DataFrame
        y_multi: Target DataFrame with multiple horizons
        horizons: List of prediction horizons
        sequence_length: Number of timesteps for input sequence
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        patience: Early stopping patience
        validation_split: Fraction of data for validation
        max_grad_norm: Maximum gradient norm for clipping
        hidden_sizes: List of LSTM hidden layer sizes
        dropout_rate: Dropout rate
        verbose: Whether to print progress
        
    Returns:
        Tuple of (model, scaler, history, sequence_length, horizons)
    """
    device = get_device()

    if 'current_differential' not in y_multi.columns:
        raise ValueError(
            "June 2026 training expects delta targets. Build y_multi with "
            "create_target_and_features(..., delta_targets=True)."
        )
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Create sequences (y includes the current differential as last column)
    X_seq, y_seq_all, valid_indices = create_sequences(X_scaled, y_multi, sequence_length)
    y_seq = y_seq_all[:, :-1]            # delta targets
    current_seq = y_seq_all[:, -1]       # diff(t) at sequence end
    
    if verbose:
        print(f"\nSequence shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")

    # Winter upweighting: structural error is worst when the differential
    # varies the most (winter), so those samples count more in the loss.
    months = pd.DatetimeIndex(valid_indices).month.to_numpy()
    winter_months = JUNE_2026_TRAINING['winter_months']
    season_weights = np.where(
        np.isin(months, winter_months),
        JUNE_2026_TRAINING['winter_weight'],
        1.0
    ).astype(np.float32)

    # Feature columns that depend on FUTURE rainfall. During training these
    # are computed from the rain that actually fell, but at forecast time the
    # model only gets uncertain forecasts -> inject multiplicative noise so
    # the model does not over-trust them.
    future_rain_col_idx = [
        i for i, col in enumerate(X.columns)
        if col.startswith('rainfall_future_')
        or col.startswith('future_rainfall_intensity_')
        or col.startswith('future_flood_risk_')
    ]
    future_rain_noise_std = JUNE_2026_TRAINING['future_rain_noise_std']
    if verbose:
        print(f"Future-rain noise augmentation on {len(future_rain_col_idx)} "
              f"features (std={future_rain_noise_std})")
        print(f"Winter-upweighted samples: {(season_weights > 1).sum()} / {len(season_weights)}")
    
    # Train/validation split (time series)
    n_train = int(len(X_seq) * (1 - validation_split))
    X_train, X_val = X_seq[:n_train], X_seq[n_train:]
    y_train, y_val = y_seq[:n_train], y_seq[n_train:]
    cur_train, cur_val = current_seq[:n_train], current_seq[n_train:]
    w_train, w_val = season_weights[:n_train], season_weights[n_train:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    cur_train_tensor = torch.FloatTensor(cur_train)
    w_train_tensor = torch.FloatTensor(w_train)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    cur_val_tensor = torch.FloatTensor(cur_val).to(device)
    w_val_tensor = torch.FloatTensor(w_val).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(
        X_train_tensor, y_train_tensor, cur_train_tensor, w_train_tensor
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    
    # Initialize model
    n_features = X_seq.shape[2]
    n_horizons = len(horizons)
    model = MultiHorizonLSTMModel(
        input_size=n_features,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        n_horizons=n_horizons
    )
    model = model.to(device)
    
    # Horizon importance weights
    # Put extra emphasis on getting the very first few hours right,
    # then gradually taper importance with horizon.
    horizon_importance = []
    for h in horizons:
        if h <= 6:
            horizon_importance.append(2.0)   # very near-term: strongest weight
        elif h <= 24:
            horizon_importance.append(2.0)   # rest of first day
        elif h <= 72:
            horizon_importance.append(1.5)   # next couple of days
        else:
            horizon_importance.append(0.75)   # far horizons: still weighted, but less
    horizon_importance = torch.tensor(horizon_importance, device=device).view(1, -1)

    # Hour gaps between consecutive horizons (incl. virtual h=0 point),
    # used by the recession-rate penalty.
    horizon_gaps = torch.tensor(
        np.diff([0] + list(horizons)).astype(np.float32), device=device
    ).view(1, -1)
    
    if verbose:
        print(f"\nModel Architecture:")
        print(model)
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"{'='*70}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        train_maes = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True, disable=not verbose)
        
        for batch_X, batch_y, batch_cur, batch_w in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_cur = batch_cur.to(device)
            batch_w = batch_w.to(device)

            # Future-rain noise augmentation (training only): one mean-1
            # log-normal factor per sample per feature, constant across the
            # sequence so temporal coherence is preserved.
            if future_rain_col_idx and future_rain_noise_std > 0:
                noise = torch.exp(
                    torch.randn(batch_X.shape[0], 1, len(future_rain_col_idx), device=device)
                    * future_rain_noise_std
                    - 0.5 * future_rain_noise_std ** 2
                )
                batch_X[:, :, future_rain_col_idx] = (
                    batch_X[:, :, future_rain_col_idx] * noise
                )
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = _compute_loss(
                outputs, batch_y, batch_cur, batch_w,
                horizon_gaps, horizon_importance, device
            )
            
            if torch.isnan(loss):
                raise ValueError("NaN loss encountered")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_maes.append(torch.mean(torch.abs(outputs - batch_y)).item())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{torch.mean(torch.abs(outputs - batch_y)).item():.4f}'
            })
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            # Use the same weighted loss function for validation to make it comparable
            val_loss = _compute_loss(
                val_outputs, y_val_tensor, cur_val_tensor, w_val_tensor,
                horizon_gaps, horizon_importance, device
            )
            val_mae = torch.mean(torch.abs(val_outputs - y_val_tensor))
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(val_mae.item())
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) - "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}, "
                  f"Train MAE: {avg_train_mae:.6f}, Val MAE: {val_mae.item():.6f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print("\n✓ Restored best model weights")
    
    if verbose:
        print("✓ Model training complete!")
    
    return model, scaler, history, sequence_length, horizons


def save_model(
    model: MultiHorizonLSTMModel,
    scaler: MinMaxScaler,
    config: Dict,
    save_dir: str = 'models',
    name: str = 'latest'
) -> None:
    """
    Save trained model, scaler, and configuration.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        config: Configuration dictionary
        save_dir: Directory to save to
        name: Name prefix for files
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_path = save_path / f'multihorizon_model_{name}.pth'
    scaler_path = save_path / f'scaler_{name}.pkl'
    config_path = save_path / f'config_{name}.pkl'
    
    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save config
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ Config saved to: {config_path}")


def load_model(
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    config_path: Optional[str] = None,
    models_dir: str = 'models',
    location: Optional[str] = None,
) -> Tuple[MultiHorizonLSTMModel, MinMaxScaler, Dict]:
    """
    Load a saved model, scaler, and configuration.
    
    Args:
        model_path: Path to model weights (overrides defaults when provided)
        scaler_path: Path to scaler (overrides defaults when provided)
        config_path: Path to config (overrides defaults when provided)
        models_dir: Directory containing model files
        location: Optional location name; when provided and explicit paths are
            not given, files named with the pattern
            'multihorizon_model_{location}_latest.pth',
            'scaler_{location}_latest.pkl', and
            'config_{location}_latest.pkl' will be used. Falls back to the
            generic '*_latest' filenames for backward compatibility when
            location is None.
        
    Returns:
        Tuple of (model, scaler, config)
    """
    device = get_device()
    models_dir = Path(models_dir)
    
    # Use location-specific "latest" files when a location is provided,
    # otherwise fall back to the generic "*_latest" filenames.
    if model_path is None:
        if location is not None:
            model_path = models_dir / f'multihorizon_model_{location}_latest.pth'
        else:
            model_path = models_dir / 'multihorizon_model_latest.pth'
    if scaler_path is None:
        if location is not None:
            scaler_path = models_dir / f'scaler_{location}_latest.pkl'
        else:
            scaler_path = models_dir / 'scaler_latest.pkl'
    if config_path is None:
        if location is not None:
            config_path = models_dir / f'config_{location}_latest.pkl'
        else:
            config_path = models_dir / 'config_latest.pkl'
    
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    config_path = Path(config_path)
    
    # Load config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Recreate model
    model = MultiHorizonLSTMModel(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        n_horizons=len(config['horizons']),
        dropout_rate=config['dropout_rate']
    )
    model = model.to(device)
    
    # Load weights
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load scaler
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Scaler loaded from: {scaler_path}")
    print(f"✓ Sequence length: {config['sequence_length']}")
    print(f"✓ Horizons: {config['horizons']}")
    
    return model, scaler, config
