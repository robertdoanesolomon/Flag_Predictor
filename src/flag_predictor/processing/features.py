"""
Feature engineering for river differential prediction.

Creates features from rainfall and differential data including:
- Historical rainfall aggregations
- Soil saturation proxies
- River state features
- Future rainfall features
- Cyclical time features
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def create_features_with_future_rainfall(
    df: pd.DataFrame,
    future_rainfall_df: Optional[pd.DataFrame] = None,
    differential_column: str = 'differential',
) -> pd.DataFrame:
    """
    Create comprehensive feature set including future rainfall forecasts.
    
    This function creates:
    - Historical rainfall features (rolling sums, acceleration, intensity)
    - Low-flow regime detection features
    - River differential features (lags, rolling stats, velocity, acceleration)
    - Future rainfall features (if forecast provided)
    - Interaction features
    - Cyclical time features
    
    Args:
        df: DataFrame with historical differential and rainfall data (hourly)
        future_rainfall_df: DataFrame with future rainfall forecasts (optional)
        differential_column: Name of the differential column
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()

    # === Historical Rainfall Features ===
    # Treat ONLY rainfall station columns as rainfall (not flow/level/groundwater)
    rainfall_cols = [
        col
        for col in df.columns
        if col != differential_column
        and not col.startswith("flow_m3s_")
        and not col.startswith("level_m_")
        and not col.startswith("groundwater_mAOD_")
    ]
    df[rainfall_cols] = df[rainfall_cols].fillna(0)
    df["catchment_rainfall_total"] = df[rainfall_cols].sum(axis=1)
    
    # === Low-Flow Regime Features ===
    df = _create_low_flow_features(df, differential_column)
    
    # === Historical Rainfall Aggregations ===
    df = _create_rainfall_features(df)
    
    # === Future Rainfall Features ===
    if future_rainfall_df is not None:
        df = _create_future_rainfall_features(df, future_rainfall_df)
    
    # === River Differential Features ===
    df = _create_differential_features(df, differential_column)
    
    # === Flow Features ===
    flow_cols = [col for col in df.columns if col.startswith('flow_m3s_')]
    if flow_cols:
        df = _create_flow_features(df, flow_cols)
    
    # === Level Features ===
    level_cols = [col for col in df.columns if col.startswith('level_m_')]
    if level_cols:
        df = _create_level_features(df, level_cols)
    
    # === Groundwater Features ===
    groundwater_cols = [col for col in df.columns if col.startswith('groundwater_mAOD_')]
    if groundwater_cols:
        df = _create_groundwater_features(df, groundwater_cols)
    
    # === Interaction Features ===
    df = _create_interaction_features(df, differential_column)
    
    # === Cyclical Features ===
    df['day_of_year'] = df.index.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Hour-of-day features to capture typical rise timing (e.g. 9–12)
    df['hour_of_day'] = df.index.hour
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    return df


def _create_low_flow_features(df: pd.DataFrame, differential_column: str) -> pd.DataFrame:
    """Create features for detecting stable low-flow periods."""
    
    # Hours since significant rainfall (>1mm total)
    significant_rain = (df['catchment_rainfall_total'] > 1.0).astype(int)
    df['hours_since_rain'] = significant_rain.groupby(
        (significant_rain != significant_rain.shift()).cumsum()
    ).cumcount()
    df.loc[significant_rain == 1, 'hours_since_rain'] = 0
    df['hours_since_rain'] = df['hours_since_rain'].clip(upper=720)  # Cap at 30 days
    
    # Is currently in a "dry spell"? (no significant rain for 48+ hours)
    df['is_dry_spell'] = (df['hours_since_rain'] > 48).astype(float)
    
    # Is currently in a "low flow" state? (differential < 0.2m)
    df['is_low_flow'] = (df[differential_column] < 0.2).astype(float)
    
    # Combined: dry spell AND low flow - strong stability signal
    df['stable_low_regime'] = df['is_dry_spell'] * df['is_low_flow']
    
    # Recent rainfall trend: is rainfall decreasing?
    df['rainfall_trend_48h'] = (
        df['catchment_rainfall_total'].rolling(window=24).sum() -
        df['catchment_rainfall_total'].rolling(window=24).sum().shift(24)
    )
    
    # Is drainage ongoing?
    df['is_draining'] = (
        (df[differential_column].diff(6) < -0.01) &
        (df['catchment_rainfall_total'].rolling(window=6).sum() < 1.0)
    ).astype(float)
    
    return df


def _create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical rainfall aggregation features."""
    
    # Rolling sums at different timescales
    for window in [6, 12, 24, 72, 168, 720]:
        df[f'rainfall_rolling_{window}h'] = df['catchment_rainfall_total'].rolling(window=window).sum()
    
    # Intensity ratios
    df['rainfall_intensity_ratio_6_24'] = (
        df['rainfall_rolling_6h'] / (df['rainfall_rolling_24h'] + 0.01)
    )
   # df['rainfall_intensity_ratio_24_168'] = (
   #     df['rainfall_rolling_24h'] / (df['rainfall_rolling_168h'] + 0.01)
   # )
    
    # Rainfall acceleration
    for window in [1, 3, 6, 12, 24, 48]:
        df[f'rainfall_accel_{window}h'] = (
            df['catchment_rainfall_total'].rolling(window=window).sum().diff(window)
        )
    
    # Exponential memory (fast/medium/slow runoff)
    for span, name in [(6, 'fast'), (24, 'medium'), (72, 'slow')]:
        df[f'catchment_rainfall_exp_{name}'] = (
            df['catchment_rainfall_total'].ewm(span=span, adjust=False).mean()
        )
    
    # Antecedent wetness index
    df['antecedent_wetness'] = (
        0.5 * df['rainfall_rolling_24h'] +
        0.3 * df['rainfall_rolling_72h'] +
        0.2 * df['rainfall_rolling_168h']
    )
    
    return df


def _create_future_rainfall_features(
    df: pd.DataFrame,
    future_rainfall_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create features from future rainfall forecasts.

    Provide rainfall at various horizons and windows; the model learns the
    effective catchment lag from data (rain at 6h, 12h, 18h, etc. can all matter).
    """
    # Ensure timezone alignment
    if future_rainfall_df.index.tz is None and df.index.tz is not None:
        future_rainfall_df = future_rainfall_df.tz_localize(df.index.tz)
    elif future_rainfall_df.index.tz is not None and df.index.tz is None:
        df.index = df.index.tz_localize(future_rainfall_df.index.tz)
    
    # Align future rainfall with df index
    future_rainfall_aligned = future_rainfall_df.reindex(df.index)
    catchment_future = future_rainfall_aligned.sum(axis=1)
    
    # Future rainfall at different windows: "rainfall in the next N hours"
    for window in [6, 12, 24, 48, 72, 120, 240]:
        df[f'rainfall_future_{window}h'] = (
            catchment_future.rolling(window=window).sum().shift(-window)
        )
    
    # Horizon-aligned future rainfall
    for h in [2, 4, 8, 10, 14, 16, 18, 20, 22, 30, 36, 42]:
        df[f'rainfall_future_{h}h'] = (
            catchment_future.rolling(window=h).sum().shift(-h)
        )
    
    # Future dry detection
    df['future_is_dry_240h'] = (df['rainfall_future_240h'] < 5.0).astype(float)
    df['future_is_dry_120h'] = (df['rainfall_future_120h'] < 3.0).astype(float)
    df['future_is_dry_72h'] = (df['rainfall_future_72h'] < 2.0).astype(float)
    df['future_is_dry_48h'] = (df['rainfall_future_48h'] < 1.0).astype(float)
    
    # Stability signal
    df['expect_stable_low'] = df['stable_low_regime'] * df['future_is_dry_240h']
    
    # Future rainfall intensity
    df['future_rainfall_intensity_240h'] = df['rainfall_future_240h'] / 240
    df['future_rainfall_intensity_120h'] = df['rainfall_future_120h'] / 120
    df['future_rainfall_intensity_72h'] = df['rainfall_future_72h'] / 72
    
    return df


def _create_differential_features(df: pd.DataFrame, differential_column: str) -> pd.DataFrame:
    """Create features from historical differential values."""
    
    # Lagged values
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'differential_lag_{lag}h'] = df[differential_column].shift(lag)
    
    # Rolling statistics
    for window in [6, 24, 72, 168]:
        df[f'differential_rolling_mean_{window}h'] = (
            df[differential_column].rolling(window=window).mean()
        )
        df[f'differential_rolling_std_{window}h'] = (
            df[differential_column].rolling(window=window).std()
        )
        df[f'differential_rolling_max_{window}h'] = (
            df[differential_column].rolling(window=window).max()
        )
        df[f'differential_rolling_min_{window}h'] = (
            df[differential_column].rolling(window=window).min()
        )
    
    # Range features
    df['differential_range_72h'] = (
        df['differential_rolling_max_72h'] - df['differential_rolling_min_72h']
    )
    df['differential_range_168h'] = (
        df['differential_rolling_max_168h'] - df['differential_rolling_min_168h']
    )
    
    # Rate of change (velocity)
    for i in [1, 3, 6, 12, 24]:
        df[f'differential_velocity_{i}h'] = df[differential_column].diff(periods=i) / i
    
    # Acceleration (second derivative)
    for i in [3, 6, 12, 24]:
        velocity = df[differential_column].diff(periods=i)
        df[f'differential_acceleration_{i}h'] = velocity.diff(periods=i) / i
    
    # Momentum indicator
    df['differential_momentum_12h'] = df[differential_column].rolling(window=12).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.std() + 0.001) if len(x) > 0 else 0,
        raw=False
    )
    
    # Rising flags
    df['is_rising_6h'] = (df['differential_velocity_6h'] > 0.01).astype(float)
    df['is_rising_24h'] = (df['differential_velocity_24h'] > 0.01).astype(float)
    
    # Other features
    df['differential_roc_6h'] = df[differential_column].diff(periods=6)
    df['differential_ewma_6h'] = df[differential_column].ewm(span=6, adjust=False).mean()
    
    return df


def _create_flow_features(df: pd.DataFrame, flow_cols: List[str]) -> pd.DataFrame:
    """
    Create features from flow data (e.g., Farmoor upstream flow).

    Flow obeys different statistics than differential (0-1.5m) and rainfall (mm).
    Flow is typically 20-200+ m³/s, continuous, with different scale and distribution.
    We use scale-invariant and bounded transforms so flow features sit in a
    comparable range and don't dominate the model.

    - Log transform: compresses range, flow often log-normal
    - Percentile features: bounded 0-1, comparable to differential
    - Rate of change as fraction: scale-invariant
    - Travel-time lags: Farmoor is upstream; flow takes 6-24h to reach downstream
    """
    for flow_col in flow_cols:
        station_name = flow_col.replace('flow_m3s_', '')
        flow = df[flow_col]

        # --- Scale-invariant transforms (handle different statistics) ---
        # Log transform: compresses 20-200 m³/s into ~3-5.3, more comparable to other features
        flow_log = np.log1p(flow.clip(lower=0))

        # Normalized within 720h window: bounded 0-1, "how high is flow vs recent range"
        rmin = flow.rolling(720, min_periods=24).min()
        rmax = flow.rolling(720, min_periods=24).max()
        df[f'{flow_col}_norm_720h'] = (
            (flow - rmin) / (rmax - rmin + 1e-6)
        ).clip(0, 1)

        # --- Upstream lag features (travel time to downstream locations) ---
        # Farmoor flow propagates downstream; key lags 6, 12, 18, 24h
        for lag in [6, 12, 18, 24]:
            df[f'{flow_col}_lag_{lag}h'] = flow_log.shift(lag)

        # --- Scale-invariant rate of change ---
        # Fractional change: (flow - flow_6h) / flow_6h instead of m³/s per hour
        for i in [6, 12, 24]:
            lagged = flow.shift(i).replace(0, np.nan)
            df[f'{flow_col}_frac_change_{i}h'] = (
                (flow - lagged) / (lagged + 1.0)
            ).clip(-2, 2)  # Cap extreme ratios

        # Rising / falling as binary (scale-invariant)
        df[f'{flow_col}_is_rising_6h'] = (flow.diff(6) > 0).astype(float)
        df[f'{flow_col}_is_rising_24h'] = (flow.diff(24) > 0).astype(float)

        # --- Rolling stats on log-scale (more stable) ---
        for window in [24, 72, 168]:
            df[f'{flow_col}_log_rolling_mean_{window}h'] = (
                flow_log.rolling(window=window, min_periods=6).mean()
            )

        # Anomaly: current vs 7-day baseline (log space)
        baseline = flow_log.rolling(window=168, min_periods=24).mean()
        df[f'{flow_col}_log_anomaly'] = (flow_log - baseline).clip(-2, 2)
    return df


def _create_level_features(df: pd.DataFrame, level_cols: List[str]) -> pd.DataFrame:
    """Create features from level data (e.g., Hannington Bridge, Shifford Lock, etc.)."""
    
    # For each level column, create similar features
    for level_col in level_cols:
        station_name = level_col.replace('level_m_', '')
        
        # Lagged values
        for lag in [1, 3, 6, 12, 24]:
            df[f'{level_col}_lag_{lag}h'] = df[level_col].shift(lag)
        
        # Rolling statistics
        for window in [6, 24, 72]:
            df[f'{level_col}_rolling_mean_{window}h'] = (
                df[level_col].rolling(window=window).mean()
            )
            df[f'{level_col}_rolling_std_{window}h'] = (
                df[level_col].rolling(window=window).std()
            )
        
        # Rate of change (velocity)
        for i in [1, 3, 6, 12]:
            df[f'{level_col}_velocity_{i}h'] = df[level_col].diff(periods=i) / i
        
        # Rising flag
        df[f'{level_col}_is_rising_6h'] = (df[f'{level_col}_velocity_6h'] > 0).astype(float)
    
    # Cross-station level differences (upstream - downstream relationships)
    if len(level_cols) >= 2:
        # Example: Hannington Bridge is upstream, so differences might be useful
        for i, col1 in enumerate(level_cols):
            for col2 in level_cols[i+1:]:
                station1 = col1.replace('level_m_', '')
                station2 = col2.replace('level_m_', '')
                df[f'level_diff_{station1}_{station2}'] = df[col1] - df[col2]
    
    return df


def _create_groundwater_features(df: pd.DataFrame, groundwater_cols: List[str]) -> pd.DataFrame:
    """Create features from groundwater data (e.g., SP50_72).
    
    Groundwater changes very slowly, so we focus on absolute values and long-term trends,
    not short-term changes.
    """
    
    # For each groundwater column, create minimal features
    for gw_col in groundwater_cols:
        station_name = gw_col.replace('groundwater_mAOD_', '')
        
        # Only a few key lags (groundwater changes slowly)
        for lag in [24, 168]:  # 1 day and 1 week
            df[f'{gw_col}_lag_{lag}h'] = df[gw_col].shift(lag)
        
        # Long-term rolling statistics (groundwater changes very slowly)
        for window in [168, 720]:  # 1 week and 1 month
            df[f'{gw_col}_rolling_mean_{window}h'] = (
                df[gw_col].rolling(window=window).mean()
            )
        
        # That's it - no velocity, acceleration, or rising flags
        # The absolute value and long-term trends are what matter
    
    return df


def _create_interaction_features(df: pd.DataFrame, differential_column: str) -> pd.DataFrame:
    """Create interaction features between rainfall and differential."""
    
    df['rainfall_last_hour'] = df['catchment_rainfall_total']
    df['rainfall_interaction_720h'] = df['rainfall_last_hour'] * df['rainfall_rolling_720h']
    
    # Flood amplification (recent rainfall × current differential)
    for window in [24, 72, 168]:
        df[f'flood_amplification_{window}h'] = (
            df[f'rainfall_rolling_{window}h'] * df[differential_column]
        )
    
    # Future flood risk
    if 'rainfall_future_24h' in df.columns:
        for window in [24, 48, 72]:
            df[f'future_flood_risk_{window}h'] = (
                df[f'rainfall_future_{window}h'] * df[differential_column]
            )
    
    # Rainfall in transit
    df['rainfall_in_transit_12_24h'] = (
        df['catchment_rainfall_total'].rolling(window=12).sum().shift(12)
    )
    df['rainfall_in_transit_6_18h'] = (
        df['catchment_rainfall_total'].rolling(window=12).sum().shift(6)
    )
    
    return df


def create_target_and_features(
    df_featureless: pd.DataFrame,
    future_rainfall_df: pd.DataFrame,
    differential_column: str = 'differential',
    horizons: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[int]]:
    """
    Create multi-horizon targets and feature set for training.
    
    Args:
        df_featureless: DataFrame with differential and rainfall (no features yet)
        future_rainfall_df: DataFrame with future rainfall (use historical for training)
        differential_column: Name of the differential column to predict
        horizons: List of hours ahead to predict
        
    Returns:
        Tuple of (X, y_multi, mask, horizons)
    """
    if horizons is None:
        horizons = [
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            30, 36, 42, 48,
            72, 96, 120, 144, 168, 192, 216, 240
        ]
    
    print(f"\nCreating features with future rainfall...")
    df_with_features = create_features_with_future_rainfall(
        df_featureless, future_rainfall_df, differential_column
    )
    
    print(f"Creating targets for {len(horizons)} horizons...")
    
    # Create target variables (one for each horizon)
    targets = []
    for horizon in horizons:
        target_col = f'target_{horizon}h'
        df_with_features[target_col] = df_with_features[differential_column].shift(-horizon)
        targets.append(target_col)
    
    # Define features (exclude targets, differential, and raw flow/level/groundwater)
    # Raw flow (m³/s) has very different statistics; we use only derived features.
    def is_raw_source(col: str) -> bool:
        if col.startswith('flow_m3s_'):
            return not any(s in col for s in ['_lag_', '_norm_', '_frac_change_', '_is_rising_', '_log_', '_anomaly'])
        if col.startswith('level_m_') and 'level_diff_' not in col:
            return not any(s in col for s in ['_lag_', '_velocity_', '_rolling_', '_is_rising_'])
        if col.startswith('groundwater_mAOD_'):
            return not any(s in col for s in ['_lag_', '_rolling_'])
        return False

    raw_source_cols = [col for col in df_with_features.columns if is_raw_source(col)]
    exclude_cols = (
        targets +
        [differential_column] +
        [col for col in df_with_features.columns if 'target_' in col] +
        raw_source_cols
    )
    features = [col for col in df_with_features.columns if col not in exclude_cols]
    
    X = df_with_features[features]
    y_multi = df_with_features[targets]
    
    # Remove rows where ANY target is NaN
    mask = ~y_multi.isna().any(axis=1)
    X = X[mask]
    y_multi = y_multi[mask]
    
    print(f"\n{'='*70}")
    print(f"Multi-Horizon Model Setup:")
    print(f"{'='*70}")
    print(f"Number of features: {len(features)}")
    print(f"Number of horizons: {len(horizons)}")
    print(f"Horizons: {horizons}")
    print(f"Training samples: {len(X)}")
    print(f"Date range: {X.index.min()} to {X.index.max()}")
    print(f"{'='*70}")
    
    return X, y_multi, mask, horizons
