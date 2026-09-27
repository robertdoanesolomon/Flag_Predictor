"""
High-level pipeline functions for the Flag Predictor.

This module provides simple, high-level functions that combine
data loading, processing, training, and prediction steps.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import (
    get_location_config,
    LocationConfig,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    PHYSICAL_CONSTRAINTS,
    RAINFALL_STATION_NAMES,
    SEPTEMBER_2026_TRAINING,
)
from .data.api import (
    fetch_all_api_data,
    calculate_isis_differential,
    calculate_godstow_differential,
    calculate_wallingford_differential,
    get_rainfall_forecast,
    get_rainfall_forecast_ensemble,
)
from .data.loader import load_all_historical_data, load_historical_flow, load_historical_rainfall
from .processing.cleaning import merge_and_clean_data, merge_rain_and_flow
from .processing.features import (
    build_september_encoder_features,
    create_target_and_features,
    rain_station_columns,
)
from .models.lstm import HourlyDecoderModel, MultiHorizonLSTMModel, get_device
from .models.training import train_model, save_model, load_model
from .models.september_train import train_hourly_decoder
from .prediction.forecast import (
    predict_single,
    predict_ensemble,
    compute_ensemble_statistics,
    compute_flag_probabilities,
)


def prepare_training_data(
    location: str = 'isis',
    project_root: Optional[Path] = None,
    delta_targets: bool = True,
    include_api: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare all training data for a location.
    
    This function:
    1. Loads historical differential and rainfall data
    2. Fetches recent API data
    3. Calculates differentials from API data
    4. Merges and cleans all data
    
    Args:
        location: Location name ('isis' or 'godstow')
        project_root: Project root directory
        verbose: Whether to print progress
        
    Returns:
        Tuple of (merged_df, X, y_multi)
    """
    config = get_location_config(location)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Preparing training data for {config.display_name}")
        print(f"{'='*70}")
    
    # Load historical data (flow from Farmoor acts as upstream guide for all locations)
    hist_diff_df, hist_rainfall_df, hist_flow_df, hist_level_df, hist_groundwater_df = load_all_historical_data(
        location=location,
        project_root=project_root,
        use_extra_sources=True,
    )
    
    # Fetch API data (with location-specific rainfall stations and flow)
    if include_api:
        river_levels, api_rainfall, api_flow = fetch_all_api_data(location=location, verbose=verbose)
        if location == 'isis':
            api_diff_df = calculate_isis_differential(river_levels)
        elif location == 'godstow':
            api_diff_df = calculate_godstow_differential(river_levels)
        elif location == 'wallingford':
            api_diff_df = calculate_wallingford_differential(river_levels)
        else:
            api_diff_df = pd.DataFrame(columns=['differential'], index=pd.DatetimeIndex([]))
        if verbose:
            print(f"\n✓ API differential: {api_diff_df.shape}")
    else:
        api_diff_df = None
        api_rainfall = None
        api_flow = None
        if verbose:
            print("\nSkipping API fetch (historical CSVs only)")
    
    # Merge and clean
    merged_df = merge_and_clean_data(
        hist_diff_df=hist_diff_df,
        hist_rainfall_df=hist_rainfall_df,
        api_diff_df=api_diff_df,
        api_rainfall_df=api_rainfall,
        api_flow_df=api_flow,
        hist_flow_df=hist_flow_df,
        hist_level_df=hist_level_df,
        hist_groundwater_df=hist_groundwater_df,
        differential_column='differential',
        min_differential=config.min_differential,
        max_differential=config.max_differential,
        verbose=verbose
    )

    # Optionally drop specific rainfall stations for certain locations.
    # For Godstow we do NOT want to use Bicester or Grimsbury rainfall.
    if location.lower() == "godstow":
        banned_stations = ["Bicester", "Grimsbury"]
        drop_cols = [
            col
            for col in merged_df.columns
            if any(station in col for station in banned_stations)
        ]
        if drop_cols and verbose:
            print(f"\nDropping rainfall stations for godstow: {drop_cols}")
        merged_df = merged_df.drop(columns=drop_cols, errors="ignore")
    
    # Carry the raw EA reading time behind each recent hourly differential point.
    if api_diff_df is not None and "reading_time" in api_diff_df.attrs:
        merged_df.attrs["reading_time"] = api_diff_df.attrs["reading_time"]

    if verbose:
        print(f"\n✓ Merged data: {merged_df.shape}")
        print(f"  Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    # Create features and targets
    # Only rainfall columns go into future_rainfall_df (exclude flow, level, groundwater)
    rainfall_cols = [
        col for col in merged_df.columns
        if col != 'differential'
        and not col.startswith('flow_m3s_')
        and not col.startswith('level_m_')
        and not col.startswith('groundwater_mAOD_')
    ]
    historical_rainfall = merged_df[rainfall_cols].copy()
    
    X, y_multi, mask, horizons = create_target_and_features(
        df_featureless=merged_df,
        future_rainfall_df=historical_rainfall,
        differential_column='differential',
        horizons=config.horizons,
        delta_targets=delta_targets
    )

    # Drop any rows with NaNs in features or targets (e.g. from long rolling
    # windows such as 720h). These typically only affect the very start of
    # the series but can otherwise cause NaN losses during training.
    nan_mask = ~(X.isna().any(axis=1) | y_multi.isna().any(axis=1))
    if not bool(nan_mask.all()):
        dropped = int((~nan_mask).sum())
        if verbose:
            print(f"\n✓ Dropping {dropped} rows with NaNs in features/targets "
                  f"(after feature engineering)")
        # Use positional indexing to avoid any index alignment issues
        X = X.iloc[nan_mask.to_numpy().nonzero()[0]]
        y_multi = y_multi.iloc[nan_mask.to_numpy().nonzero()[0]]
    
    return merged_df, X, y_multi


def train_location_model(
    location: str = 'isis',
    project_root: Optional[Path] = None,
    save_dir: str = 'models',
    verbose: bool = True,
    **training_kwargs
) -> Tuple[MultiHorizonLSTMModel, Dict]:
    """
    Train a model for a specific location.
    
    Args:
        location: Location name ('isis' or 'godstow')
        project_root: Project root directory
        save_dir: Directory to save model
        verbose: Whether to print progress
        **training_kwargs: Additional arguments for train_model()
        
    Returns:
        Tuple of (trained_model, config)
    """
    config = get_location_config(location)
    
    # Prepare data
    merged_df, X, y_multi = prepare_training_data(
        location=location,
        project_root=project_root,
        verbose=verbose
    )
    
    # Merge default config with kwargs
    train_config = {**TRAINING_CONFIG, **training_kwargs}
    
    # Train model
    model, scaler, history, sequence_length, horizons = train_model(
        X=X,
        y_multi=y_multi,
        horizons=config.horizons,
        verbose=verbose,
        **train_config
    )
    
    # Create config for saving
    model_config = {
        'location': location,
        'sequence_length': sequence_length,
        'horizons': horizons,
        'hidden_sizes': MODEL_CONFIG['hidden_sizes'],
        'dropout_rate': MODEL_CONFIG['dropout_rate'],
        'input_size': len(X.columns),
        'feature_columns': list(X.columns),
        'training_history': history,
        # June 2026 model: outputs are deltas from the current differential,
        # and forecasts get a physical recession-rate clamp.
        'predicts_delta': True,
        'max_recession_m_per_day': PHYSICAL_CONSTRAINTS['max_recession_m_per_day'],
    }
    
    # Save model
    save_model(
        model=model,
        scaler=scaler,
        config=model_config,
        save_dir=save_dir,
        name=f'{location}_latest'
    )
    
    # Also save under the June 2026 experiment name
    save_model(
        model=model,
        scaler=scaler,
        config=model_config,
        save_dir=save_dir,
        name=f'experiment_2026_06_{location}'
    )
    
    # Also save as generic 'latest' for backward compatibility
    save_model(
        model=model,
        scaler=scaler,
        config=model_config,
        save_dir=save_dir,
        name='latest'
    )
    
    return model, model_config


def run_forecast(
    location: str = 'isis',
    model_path: Optional[str] = None,
    project_root: Optional[Path] = None,
    ensemble: bool = True,
    n_members: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Run a complete forecast pipeline.
    
    This function:
    1. Loads the trained model
    2. Fetches latest API data
    3. Fetches rainfall forecast
    4. Generates predictions (single or ensemble)
    5. Computes flag probabilities
    
    Args:
        location: Location name ('isis' or 'godstow')
        model_path: Path to model directory (default: 'models')
        project_root: Project root directory
        ensemble: Whether to run ensemble prediction
        n_members: Number of ensemble members
        verbose: Whether to print progress
        
    Returns:
        Dictionary with predictions and statistics
    """
    config = get_location_config(location)
    models_dir = model_path or 'models'
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running forecast for {config.display_name}")
        print(f"{'='*70}")
    
    # Load model (use location-specific latest files when available)
    model, scaler, model_config = load_model(models_dir=models_dir, location=location)
    feature_columns = model_config['feature_columns']
    sequence_length = model_config['sequence_length']
    horizons = model_config['horizons']
    predicts_delta = model_config.get('predicts_delta', False)
    max_recession = model_config.get('max_recession_m_per_day', None)
    
    # Prepare historical data
    merged_df, _, _ = prepare_training_data(
        location=location,
        project_root=project_root,
        verbose=verbose
    )
    
    # Get rainfall forecast (with location-specific stations)
    if ensemble:
        if verbose:
            print("\nFetching ensemble rainfall forecast...")
        rainfall_forecast = get_rainfall_forecast_ensemble(
            location=location,
            n_members=n_members
        )
        
        # Get location-specific station names for prediction
        if location and location.lower() == 'wallingford':
            from ..config import WALLINGFORD_RAINFALL_STATION_NAMES
            station_names = list(RAINFALL_STATION_NAMES) + list(WALLINGFORD_RAINFALL_STATION_NAMES)
        else:
            station_names = RAINFALL_STATION_NAMES
        
        # Run ensemble prediction
        predictions_df = predict_ensemble(
            model=model,
            scaler=scaler,
            historical_df=merged_df,
            rainfall_ensemble_df=rainfall_forecast,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            horizons=horizons,
            station_names=station_names,
            n_members=n_members,
            predicts_delta=predicts_delta,
            max_recession_m_per_day=max_recession,
            verbose=verbose
        )
        
        # Compute statistics
        stats = compute_ensemble_statistics(predictions_df)
        flag_probs = compute_flag_probabilities(predictions_df, location=location)
        
        return {
            'predictions': predictions_df,
            'statistics': stats,
            'flag_probabilities': flag_probs,
            'model_config': model_config,
        }
    
    else:
        if verbose:
            print("\nFetching rainfall forecast...")
        rainfall_forecast = get_rainfall_forecast(location=location)
        
        # Run single prediction
        prediction = predict_single(
            model=model,
            scaler=scaler,
            historical_df=merged_df,
            rainfall_forecast_df=rainfall_forecast,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            horizons=horizons,
            predicts_delta=predicts_delta,
            max_recession_m_per_day=max_recession,
            verbose=verbose
        )
        
        return {
            'prediction': prediction,
            'model_config': model_config,
        }


def prepare_flow_training_data(
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rain + Farmoor flow only (no differential inner join)."""
    if verbose:
        print(f"\n{'='*70}")
        print("Preparing Farmoor flow training data (rain + flow, long record)")
        print(f"{'='*70}")
    rainfall = load_historical_rainfall(
        project_root=project_root, location='isis', verbose=verbose
    )
    flow = load_historical_flow(project_root=project_root)
    if flow is None or flow.empty:
        raise FileNotFoundError(
            "No Farmoor flow CSV found. Run: python -m flag_predictor.data.download_flow"
        )
    return merge_rain_and_flow(rainfall, flow, verbose=verbose)


def _decoder_frame(merged_df: pd.DataFrame, include_flow: bool) -> pd.DataFrame:
    rain_cols = rain_station_columns(merged_df)
    out = pd.DataFrame(index=merged_df.index)
    out['catchment_rainfall_total'] = merged_df[rain_cols].sum(axis=1) if rain_cols else 0.0
    if include_flow:
        flow_col = 'flow_m3s_Farmoor'
        if flow_col not in merged_df.columns:
            raise KeyError("merged data is missing flow_m3s_Farmoor")
        out['flow_log'] = np.log1p(merged_df[flow_col].clip(lower=0))
    return out


def train_september_flow_model(
    project_root: Optional[Path] = None,
    save_dir: str = 'models',
    verbose: bool = True,
) -> Tuple[HourlyDecoderModel, Dict]:
    """Stage 1: hourly rain → Farmoor log-flow decoder."""
    import numpy as np

    merged = prepare_flow_training_data(project_root=project_root, verbose=verbose)
    flow_col = 'flow_m3s_Farmoor'
    encoder_df = build_september_encoder_features(merged, target_col=flow_col)
    decoder_cov = _decoder_frame(merged, include_flow=False)
    target = np.log1p(merged[flow_col].clip(lower=0))
    target.name = 'flow_log'

    model, past_scaler, dec_scaler, extra = train_hourly_decoder(
        encoder_df=encoder_df,
        decoder_cov=decoder_cov,
        target=target,
        target_kind='flow',
        verbose=verbose,
    )
    model_config = {
        **extra,
        'location': 'farmoor',
        'max_recession_m_per_day': None,
        'uses_predicted_flow': False,
        'target_transform': 'log1p',
        'target_column': flow_col,
    }
    # decoder_scaler lives inside extra already
    save_model(model, past_scaler, model_config, save_dir=save_dir, name='experiment_2026_09_farmoor')
    save_model(model, past_scaler, model_config, save_dir=save_dir, name='farmoor_latest')
    return model, model_config


def train_september_location_model(
    location: str,
    project_root: Optional[Path] = None,
    save_dir: str = 'models',
    uses_predicted_flow: bool = True,
    verbose: bool = True,
    name: Optional[str] = None,
    cfg_override: Optional[Dict] = None,
) -> Tuple[HourlyDecoderModel, Dict]:
    """Stage 2: hourly differential decoder, optionally seeing future flow."""
    import numpy as np

    merged_df, _, _ = prepare_training_data(
        location=location,
        project_root=project_root,
        include_api=False,
        verbose=verbose,
    )
    encoder_df = build_september_encoder_features(merged_df, target_col='differential')
    decoder_cov = _decoder_frame(merged_df, include_flow=uses_predicted_flow)
    target = merged_df['differential']

    model, past_scaler, dec_scaler, extra = train_hourly_decoder(
        encoder_df=encoder_df,
        decoder_cov=decoder_cov,
        target=target,
        target_kind='diff',
        verbose=verbose,
        cfg_override=cfg_override,
    )
    loc_cfg = get_location_config(location)
    model_config = {
        **extra,
        'location': location,
        'predicts_delta': True,
        'max_recession_m_per_day': PHYSICAL_CONSTRAINTS['max_recession_m_per_day'],
        'min_differential': loc_cfg.min_differential,
        'uses_predicted_flow': uses_predicted_flow,
        'target_transform': None,
        'target_column': 'differential',
    }
    save_name = name or f'experiment_2026_09_{location}'
    save_model(
        model, past_scaler, model_config, save_dir=save_dir,
        name=save_name,
    )
    save_model(
        model, past_scaler, model_config, save_dir=save_dir,
        name=f'{location}_latest',
    )
    return model, model_config
