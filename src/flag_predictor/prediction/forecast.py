"""
Forecasting functions for river differential prediction.

This module provides functions for generating predictions using trained models,
including single forecasts and ensemble predictions from multiple rainfall scenarios.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from ..models.lstm import HourlyDecoderModel, MultiHorizonLSTMModel, get_device
from ..processing.features import (
    build_september_encoder_features,
    create_features_with_future_rainfall,
    rain_station_columns,
)
from ..config import PHYSICAL_CONSTRAINTS, get_location_config


def apply_recession_limit(
    predictions: pd.Series,
    max_recession_m_per_day: float = PHYSICAL_CONSTRAINTS['max_recession_m_per_day'],
) -> pd.Series:
    """
    Enforce the physical maximum recession rate on an hourly forecast.

    The river essentially never falls faster than ~3 inches (0.076m) per day
    (Anu's rule). Walking forward through the trajectory, each step is not
    allowed to fall more than max_rate/24 below the (already constrained)
    previous value. Rises are untouched.
    """
    max_drop_per_step = max_recession_m_per_day / 24.0
    values = predictions.to_numpy(dtype=float).copy()
    for i in range(1, len(values)):
        floor = values[i - 1] - max_drop_per_step
        if values[i] < floor:
            values[i] = floor
    return pd.Series(values, index=predictions.index)


def apply_hourly_physics(
    predictions: pd.Series,
    historical_diff: pd.Series,
    future_rain: pd.Series,
    max_recession_m_per_day: Optional[float] = None,
    min_differential: Optional[float] = None,
) -> pd.Series:
    """
    Physical limits on an hourly differential forecast.

    Recession clamp always. Dry hours cannot rise, and a quiet plateau is
    held near the current observation. As soon as rain is falling, the
    learned trajectory is allowed to move (so 0–24h rain response is not
    delayed until a large catchment total has accumulated).
    """
    phys = PHYSICAL_CONSTRAINTS
    rec = (
        phys['max_recession_m_per_day']
        if max_recession_m_per_day is None
        else max_recession_m_per_day
    )
    values = predictions.to_numpy(dtype=float).copy()
    current = float(values[0])
    rain = future_rain.reindex(predictions.index).fillna(0.0).to_numpy(dtype=float)
    rain[0] = 0.0
    rain = np.clip(rain, 0, None)
    rain_3h = np.convolve(rain, np.ones(3), mode='full')[: len(rain)]

    hist = historical_diff.dropna()
    recent = hist.iloc[-48:] if len(hist) else hist
    if len(recent) >= 8:
        net_48h = float(recent.iloc[-1] - recent.iloc[0])
        std_48h = float(recent.std())
    else:
        net_48h, std_48h = 0.0, 0.0
    plateau = (abs(net_48h) < phys['plateau_net_m']) and (std_48h < phys['plateau_std_m'])

    min_7d = float(hist.iloc[-168:].min()) if len(hist) else current
    min_14d = float(hist.iloc[-336:].min()) if len(hist) else current
    struct_floor = -0.15 if min_differential is None else float(min_differential)
    if plateau:
        low_floor = min(current, min_7d) - phys['dry_band_m']
    else:
        low_floor = min(current, min_7d, min_14d) - phys['low_flow_slack_m']
    low_floor = max(low_floor, struct_floor)

    dry_ceiling = current + phys['dry_band_m']
    if len(hist):
        dry_ceiling = max(dry_ceiling, float(hist.iloc[-24:].max()) + 0.005)

    max_drop = rec / 24.0
    dry_hour = phys['dry_hour_mm']
    dry_band = phys['dry_band_m']
    left_init = False

    for i in range(1, len(values)):
        rec_floor = values[i - 1] - max_drop
        if values[i] < rec_floor:
            values[i] = rec_floor
        if values[i] < low_floor:
            values[i] = low_floor
        hour_dry = (rain[i] < dry_hour) and (rain_3h[i] < dry_hour * 2.5)
        if (not hour_dry) or (i > 24):
            left_init = True
        # Only pin to the current observation while it is still dry near t0.
        # After rain (or after 24h) the learned trajectory is allowed to move,
        # otherwise winter floods get snapped back to the start.
        if (not left_init) and hour_dry:
            if values[i] > dry_ceiling:
                values[i] = dry_ceiling
            if plateau:
                lo, hi = current - 0.005, current + dry_band
                if values[i] < lo:
                    values[i] = lo
                if values[i] > hi:
                    values[i] = hi
    return pd.Series(values, index=predictions.index)


def _align_future_rain(
    rainfall_forecast_df: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    station_hint: Optional[List[str]] = None,
) -> pd.Series:
    """Hourly catchment rainfall on future_index (0 if missing)."""
    rain = rainfall_forecast_df.copy()
    if rain.index.tz is None and future_index.tz is not None:
        rain = rain.tz_localize(future_index.tz)
    elif rain.index.tz is not None and future_index.tz is None:
        rain = rain.tz_convert('UTC').tz_localize(None)
    cols = list(rain.columns)
    if station_hint:
        cols = [c for c in cols if c in rain.columns]
    catchment = rain[cols].sum(axis=1) if len(cols) else pd.Series(0.0, index=rain.index)
    catchment = catchment.reindex(future_index).fillna(0.0)
    return catchment


def predict_flow_hourly(
    flow_model: HourlyDecoderModel,
    flow_scaler: MinMaxScaler,
    flow_config: Dict,
    historical_df: pd.DataFrame,
    rainfall_forecast_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.Series:
    """240-hour Farmoor flow forecast from rain (absolute m3/s, not log)."""
    device = next(flow_model.parameters()).device
    sequence_length = flow_config['sequence_length']
    horizon = flow_config.get('horizon', 240)
    feature_columns = flow_config['feature_columns']
    decoder_scaler: MinMaxScaler = flow_config['decoder_scaler']

    lookback = max(720, sequence_length + 24)
    history = historical_df.iloc[-lookback:].copy()
    t0 = history.index[-1]
    flow_col = flow_config.get('target_column', 'flow_m3s_Farmoor')
    current_flow = float(history[flow_col].iloc[-1]) if flow_col in history.columns else 0.0

    encoder_df = build_september_encoder_features(history, target_col=flow_col)
    for col in feature_columns:
        if col not in encoder_df.columns:
            encoder_df[col] = 0.0
    seq = encoder_df[feature_columns].iloc[-sequence_length:].ffill().bfill().fillna(0)
    x = flow_scaler.transform(seq.values)
    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

    future_index = pd.date_range(t0 + pd.Timedelta(hours=1), periods=horizon, freq='1h')
    rain = _align_future_rain(rainfall_forecast_df, future_index)
    cov = rain.to_numpy(dtype=float).reshape(-1, 1)
    cov = decoder_scaler.transform(cov)
    c_t = torch.FloatTensor(cov).unsqueeze(0).to(device)

    flow_model.eval()
    with torch.no_grad():
        delta = flow_model(x_t, c_t).cpu().numpy()[0]

    log_now = np.log1p(max(current_flow, 0.0))
    flow_abs = np.expm1(log_now + delta).clip(min=0.0)
    rain_cume = np.cumsum(np.clip(rain.to_numpy(dtype=float), 0, None))
    mix = np.clip(rain_cume / max(PHYSICAL_CONSTRAINTS['dry_cume_mm'], 1e-3), 0.0, 1.0)
    flow_blend = current_flow * (1.0 - mix) + flow_abs * mix
    series = pd.Series(flow_blend, index=future_index, name=flow_col)
    if verbose:
        print(f"  Flow forecast: {flow_abs.min():.1f} – {flow_abs.max():.1f} m3/s")
    return series


def _predict_hourly_differential(
    model: HourlyDecoderModel,
    scaler: MinMaxScaler,
    model_config: Dict,
    historical_df: pd.DataFrame,
    rainfall_forecast_df: pd.DataFrame,
    predicted_flow: Optional[pd.Series],
    max_recession_m_per_day: Optional[float],
    verbose: bool,
) -> pd.Series:
    device = next(model.parameters()).device
    sequence_length = model_config['sequence_length']
    horizon = model_config.get('horizon', 240)
    feature_columns = model_config['feature_columns']
    decoder_cols = model_config['decoder_columns']
    decoder_scaler: MinMaxScaler = model_config['decoder_scaler']

    lookback = max(720, sequence_length + 24)
    history = historical_df.iloc[-lookback:].copy()
    t0 = history.index[-1]
    current_differential = float(history['differential'].iloc[-1])

    encoder_df = build_september_encoder_features(history, target_col='differential')
    for col in feature_columns:
        if col not in encoder_df.columns:
            encoder_df[col] = 0.0
    seq = encoder_df[feature_columns].iloc[-sequence_length:].ffill().bfill().fillna(0)
    x = scaler.transform(seq.values)
    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

    future_index = pd.date_range(t0 + pd.Timedelta(hours=1), periods=horizon, freq='1h')
    rain = _align_future_rain(rainfall_forecast_df, future_index)
    cov_parts = []
    for name in decoder_cols:
        if name == 'catchment_rainfall_total':
            cov_parts.append(rain.to_numpy(dtype=float))
        elif name == 'flow_log':
            if predicted_flow is None:
                last_flow = float(history['flow_m3s_Farmoor'].iloc[-1]) if 'flow_m3s_Farmoor' in history.columns else 0.0
                flow_vals = np.full(horizon, last_flow, dtype=float)
            else:
                flow_vals = predicted_flow.reindex(future_index).ffill().bfill().fillna(0).to_numpy(dtype=float)
            cov_parts.append(np.log1p(np.clip(flow_vals, 0, None)))
        else:
            cov_parts.append(np.zeros(horizon, dtype=float))
    cov = np.column_stack(cov_parts)
    cov = decoder_scaler.transform(cov)
    c_t = torch.FloatTensor(cov).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        delta = model(x_t, c_t).cpu().numpy()[0]

    values = current_differential + delta
    full_index = pd.date_range(t0, periods=horizon + 1, freq='1h')
    full = pd.Series(np.concatenate([[current_differential], values]), index=full_index)
    if max_recession_m_per_day is not None:
        mode = (model_config or {}).get('physics_mode', 'full')
        if mode == 'off':
            pass
        elif mode == 'recession':
            full = apply_recession_limit(full, max_recession_m_per_day)
        else:
            min_diff = model_config.get('min_differential')
            loc = model_config.get('location')
            if min_diff is None and loc:
                min_diff = get_location_config(loc).min_differential
            full = apply_hourly_physics(
                full,
                historical_diff=history['differential'],
                future_rain=rain,
                max_recession_m_per_day=max_recession_m_per_day,
                min_differential=min_diff,
            )
    if verbose:
        print(f"  Prediction range: {full.min():.3f}m to {full.max():.3f}m")
    return full


def predict_single(
    model: MultiHorizonLSTMModel,
    scaler: MinMaxScaler,
    historical_df: pd.DataFrame,
    rainfall_forecast_df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 120,
    horizons: Optional[List[int]] = None,
    predicts_delta: bool = False,
    max_recession_m_per_day: Optional[float] = None,
    verbose: bool = True,
    model_config: Optional[Dict] = None,
    flow_model: Optional[HourlyDecoderModel] = None,
    flow_scaler: Optional[MinMaxScaler] = None,
    flow_config: Optional[Dict] = None,
) -> pd.Series:
    """
    Generate a 240-hour river differential prediction for ONE rainfall scenario.
    
    This function:
    1. Combines recent historical data with the FULL future rainfall forecast
    2. Computes features ONCE with future rainfall visible
    3. Makes ONE prediction from the current state
    4. Interpolates the sparse horizon predictions into a continuous timeseries
    
    Args:
        model: Trained LSTM model
        scaler: Fitted feature scaler
        historical_df: Historical data with differential and rainfall
        rainfall_forecast_df: Future rainfall forecast DataFrame
        feature_columns: List of feature column names used in training
        sequence_length: LSTM sequence length
        horizons: List of prediction horizons
        predicts_delta: True for June-2026-style models whose raw outputs are
            CHANGES from the current differential (added back here)
        max_recession_m_per_day: If set, clamp the trajectory so it never
            falls faster than this rate (e.g. 0.0762 = 3 inches/day)
        verbose: Whether to print progress
        
    Returns:
        pd.Series: Hourly predictions for next 240 hours
    """
    is_hourly = (
        isinstance(model, HourlyDecoderModel)
        or (model_config or {}).get('architecture') == 'hourly_decoder'
    )
    if is_hourly:
        cfg = model_config or {}
        predicted_flow = None
        if (
            cfg.get('uses_predicted_flow', False)
            and flow_model is not None
            and flow_scaler is not None
            and flow_config is not None
        ):
            predicted_flow = predict_flow_hourly(
                flow_model, flow_scaler, flow_config,
                historical_df, rainfall_forecast_df, verbose=verbose,
            )
        return _predict_hourly_differential(
            model=model,
            scaler=scaler,
            model_config=cfg,
            historical_df=historical_df,
            rainfall_forecast_df=rainfall_forecast_df,
            predicted_flow=predicted_flow,
            max_recession_m_per_day=max_recession_m_per_day,
            verbose=verbose,
        )

    device = next(model.parameters()).device
    
    if horizons is None:
        horizons = [
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            30, 36, 42, 48,
            72, 96, 120, 144, 168, 192, 216, 240
        ]
    
    # === KEY OPTIMIZATION: Only use last 720 hours (not full dataset) ===
    lookback_hours = 720
    recent_history = historical_df.iloc[-lookback_hours:].copy()
    
    # Get current time and differential
    forecast_start_time = recent_history.index[-1]
    current_differential = recent_history['differential'].iloc[-1]
    
    if verbose:
        print(f"  Current time: {forecast_start_time}")
        print(f"  Current differential: {current_differential:.3f}m")
    
    # Ensure timezone alignment
    if rainfall_forecast_df.index.tz is None and recent_history.index.tz is not None:
        rainfall_forecast_df = rainfall_forecast_df.tz_localize(recent_history.index.tz)
    elif rainfall_forecast_df.index.tz is not None and recent_history.index.tz is None:
        rainfall_forecast_df = rainfall_forecast_df.tz_convert('UTC').tz_localize(None)
        recent_history = recent_history.copy()
    
    # Create combined dataframe: historical + future rainfall
    # This allows the feature engineering to "see" future rainfall
    combined_df = recent_history.copy()
    
    # Determine the end time we need (forecast_start_time + 240 hours)
    forecast_end_time = forecast_start_time + pd.Timedelta(hours=240)
    
    # Create a continuous hourly index from forecast_start_time to forecast_end_time
    # This ensures no gaps even if rainfall forecast starts later
    required_future_index = pd.date_range(
        start=forecast_start_time + pd.Timedelta(hours=1),  # Start 1 hour after historical
        end=forecast_end_time,
        freq='1h'
    )
    
    # Pad rainfall forecast and extend other columns to future timestamps
    last_values = recent_history.iloc[-1]
    for col in combined_df.columns:
        if col in rainfall_forecast_df.columns:
            # Create a full future timeline for this column
            padded_rainfall = pd.Series(0.0, index=required_future_index, name=col)
            
            # Fill in actual forecast values where available
            available_data = rainfall_forecast_df[[col]].copy()
            available_data = available_data[available_data.index > forecast_start_time]
            if len(available_data) > 0:
                # Align and fill in actual values
                for idx in available_data.index:
                    if idx in padded_rainfall.index:
                        padded_rainfall.loc[idx] = available_data.loc[idx, col]
            
            # Convert to DataFrame and add to combined_df
            future_data = pd.DataFrame({col: padded_rainfall})
            combined_df = pd.concat([combined_df, future_data])
        elif col == 'differential':
            # Forward fill differential for future timestamps
            future_diff = pd.DataFrame(
                {'differential': [current_differential] * len(required_future_index)},
                index=required_future_index
            )
            combined_df = pd.concat([combined_df, future_diff])
        elif col.startswith('flow_m3s_'):
            # Flow: no forecast available; use persistence (last known value)
            last_flow = last_values.get(col, np.nan)
            if pd.notna(last_flow):
                future_flow = pd.DataFrame(
                    {col: [last_flow] * len(required_future_index)},
                    index=required_future_index
                )
                combined_df = pd.concat([combined_df, future_flow])
    
    # Remove any duplicate indices
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()
    
    # Create features with the FULL future rainfall visible
    df_with_features = create_features_with_future_rainfall(
        combined_df,
        rainfall_forecast_df,
        differential_column='differential'
    )
    
    # Get features at the "current" time (end of historical data)
    current_time_idx = df_with_features.index.get_loc(forecast_start_time)
    
    # Get only the features used during training (in same order)
    # Add missing columns with 0
    for col in feature_columns:
        if col not in df_with_features.columns:
            df_with_features[col] = 0
    
    X_forecast = df_with_features[feature_columns].copy()
    X_forecast = X_forecast.ffill().bfill().fillna(0)
    
    # Get the input sequence (last `sequence_length` hours before current time)
    sequence_start = current_time_idx - sequence_length + 1
    sequence_end = current_time_idx + 1
    sequence_data = X_forecast.iloc[sequence_start:sequence_end]
    
    if len(sequence_data) < sequence_length:
        raise ValueError(
            f"Not enough historical data. Need {sequence_length} hours, got {len(sequence_data)}"
        )
    
    # Scale and convert to tensor
    sequence_scaled = scaler.transform(sequence_data)
    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predictions = model(sequence_tensor).cpu().numpy()[0]

    if predicts_delta:
        # Model outputs are changes from the current differential; the
        # forecast is anchored at the observed value by construction.
        predictions = current_differential + predictions
    
    # Create sparse prediction series
    horizon_times = [forecast_start_time + pd.Timedelta(hours=h) for h in horizons]
    sparse_predictions = pd.Series(predictions, index=horizon_times)
    
    # Add current value at t=0 for interpolation
    sparse_predictions[forecast_start_time] = current_differential
    sparse_predictions = sparse_predictions.sort_index()
    
    # Interpolate to hourly resolution
    full_timeline = pd.date_range(
        start=forecast_start_time,
        periods=241,
        freq='1h'
    )
    
    full_predictions = sparse_predictions.reindex(full_timeline)
    full_predictions = full_predictions.interpolate(method='linear')
    full_predictions = full_predictions.ffill().bfill()

    # Physical constraint: the river cannot drop faster than the maximum
    # recession rate, however dry the rainfall forecast is.
    if max_recession_m_per_day is not None:
        full_predictions = apply_recession_limit(
            full_predictions, max_recession_m_per_day
        )
    
    if verbose:
        print(f"  Prediction range: {full_predictions.min():.3f}m to {full_predictions.max():.3f}m")
    
    return full_predictions


def predict_ensemble(
    model: MultiHorizonLSTMModel,
    scaler: MinMaxScaler,
    historical_df: pd.DataFrame,
    rainfall_ensemble_df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 120,
    horizons: Optional[List[int]] = None,
    station_names: Optional[List[str]] = None,
    n_members: int = 20,
    predicts_delta: bool = False,
    max_recession_m_per_day: Optional[float] = None,
    verbose: bool = True,
    model_config: Optional[Dict] = None,
    flow_model: Optional[HourlyDecoderModel] = None,
    flow_scaler: Optional[MinMaxScaler] = None,
    flow_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Generate ensemble river flow predictions from multiple rainfall scenarios.
    
    This function:
    1. Extracts each rainfall ensemble member
    2. Runs prediction for each member (with that member's rainfall)
    3. Returns all predictions for statistical analysis
    
    Args:
        model: Trained LSTM model
        scaler: Fitted feature scaler
        historical_df: Historical data with differential and rainfall
        rainfall_ensemble_df: DataFrame with all ensemble members
                              (columns like: Osney_member_0, Osney_member_1, ...)
        feature_columns: List of feature column names used in training
        sequence_length: LSTM sequence length
        horizons: List of prediction horizons
        station_names: List of rainfall station names
        n_members: Number of ensemble members to use
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Each column is one ensemble member's 240-hour prediction
    """
    if horizons is None:
        horizons = [
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            30, 36, 42, 48,
            72, 96, 120, 144, 168, 192, 216, 240
        ]
    
    if station_names is None:
        # Infer from column names
        station_names = list(set(
            col.rsplit('_member_', 1)[0]
            for col in rainfall_ensemble_df.columns
            if '_member_' in col
        ))
    
    print("="*70)
    print("ENSEMBLE PREDICTION")
    print("="*70)
    print(f"Using {n_members} ensemble members")
    print(f"Stations: {len(station_names)}")
    print(f"Forecast horizons: {len(horizons)} points")
    print("="*70)
    
    ensemble_predictions = {}
    
    for member_idx in tqdm(range(n_members), desc="Processing ensemble members"):
        # Extract this member's rainfall across all stations
        member_columns = [f'{station}_member_{member_idx}' for station in station_names]
        
        # Check columns exist
        missing = [col for col in member_columns if col not in rainfall_ensemble_df.columns]
        if missing:
            if verbose:
                print(f"  Skipping member {member_idx}: missing {len(missing)} columns")
            continue
        
        # Create rainfall df for this member (rename columns to station names)
        member_rainfall = rainfall_ensemble_df[member_columns].copy()
        member_rainfall.columns = station_names
        
        try:
            # Generate prediction for this ensemble member
            prediction = predict_single(
                model=model,
                scaler=scaler,
                historical_df=historical_df,
                rainfall_forecast_df=member_rainfall,
                feature_columns=feature_columns,
                sequence_length=sequence_length,
                horizons=horizons,
                predicts_delta=predicts_delta,
                max_recession_m_per_day=max_recession_m_per_day,
                verbose=False,
                model_config=model_config,
                flow_model=flow_model,
                flow_scaler=flow_scaler,
                flow_config=flow_config,
            )
            
            ensemble_predictions[f'member_{member_idx}'] = prediction
            
        except Exception as e:
            print(f"  Error on member {member_idx}: {e}")
            continue
    
    # Combine all predictions into a DataFrame
    ensemble_df = pd.DataFrame(ensemble_predictions)
    
    print(f"\n✓ Generated {len(ensemble_predictions)} ensemble predictions")
    print(f"  Forecast shape: {ensemble_df.shape}")
    print(f"  Time range: {ensemble_df.index[0]} to {ensemble_df.index[-1]}")
    
    return ensemble_df


def compute_ensemble_statistics(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics from ensemble predictions.
    
    Args:
        ensemble_df: DataFrame with ensemble predictions
        
    Returns:
        DataFrame with mean, median, std, and percentiles
    """
    stats = pd.DataFrame({
        'mean': ensemble_df.mean(axis=1),
        'median': ensemble_df.median(axis=1),
        'std': ensemble_df.std(axis=1),
        'p10': ensemble_df.quantile(0.1, axis=1),
        'p25': ensemble_df.quantile(0.25, axis=1),
        'p75': ensemble_df.quantile(0.75, axis=1),
        'p90': ensemble_df.quantile(0.9, axis=1),
        'min': ensemble_df.min(axis=1),
        'max': ensemble_df.max(axis=1),
    })
    return stats


def differential_to_flag(
    differential: Union[float, pd.Series],
    location: str = 'isis'
) -> Union[str, pd.Series]:
    """
    Convert differential value to flag color.
    
    Flag colors based on differential thresholds:
    - Blue: differential <= 0.15
    - Yellow: 0.15 < differential <= 0.30
    - Orange: 0.30 < differential <= 0.50
    - Red: differential > 0.50
    
    For Wallingford, returns 'white' (no flags).
    
    Args:
        differential: Differential value(s)
        location: Location name (for location-specific thresholds)
        
    Returns:
        Flag color(s) as string or Series
    """
    # For Wallingford, return white (no flags)
    if location.lower() == 'wallingford':
        if isinstance(differential, (int, float)):
            return 'white'
        else:
            return pd.Series('white', index=differential.index)
    
    # Thresholds (same for both locations currently)
    thresholds = {
        'blue': 0.15,
        'yellow': 0.30,
        'orange': 0.50,
    }
    
    if isinstance(differential, (int, float)):
        if differential <= thresholds['blue']:
            return 'blue'
        elif differential <= thresholds['yellow']:
            return 'yellow'
        elif differential <= thresholds['orange']:
            return 'orange'
        else:
            return 'red'
    else:
        # Series
        conditions = [
            differential <= thresholds['blue'],
            differential <= thresholds['yellow'],
            differential <= thresholds['orange'],
        ]
        choices = ['blue', 'yellow', 'orange']
        return pd.Series(
            np.select(conditions, choices, default='red'),
            index=differential.index
        )


def compute_flag_probabilities(ensemble_df: pd.DataFrame, location: str = 'isis') -> pd.DataFrame:
    """
    Compute probability of each flag color from ensemble predictions.
    
    Uses location-specific thresholds to match the 5-flag system:
    green, light_blue, dark_blue, amber, red.
    
    For Wallingford, returns zero probabilities (white flags - no flags).
    
    Args:
        ensemble_df: DataFrame with ensemble predictions
        location: Location name for location-specific thresholds
        
    Returns:
        DataFrame with probability of each flag color at each timestep
    """
    from ..config import get_flag_thresholds
    
    # For Wallingford, return zero probabilities (white flags - no flags)
    if location.lower() == 'wallingford':
        probs = pd.DataFrame(index=ensemble_df.index)
        probs['p_green'] = 0.0
        probs['p_light_blue'] = 0.0
        probs['p_dark_blue'] = 0.0
        probs['p_amber'] = 0.0
        probs['p_red'] = 0.0
        return probs
    
    n_members = len(ensemble_df.columns)
    thresholds = get_flag_thresholds(location)
    
    probs = pd.DataFrame(index=ensemble_df.index)
    
    # Calculate probability for each flag using threshold ranges
    for flag, (lower, upper) in thresholds.items():
        count = ((ensemble_df >= lower) & (ensemble_df < upper)).sum(axis=1)
        probs[f'p_{flag}'] = count / n_members
    
    return probs
