"""
Data cleaning and merging utilities.

Functions for combining historical and API data, resampling to hourly frequency,
and cleaning spurious values.
"""

import numpy as np
import pandas as pd
from typing import Optional


def merge_and_clean_data(
    hist_diff_df: pd.DataFrame,
    hist_rainfall_df: pd.DataFrame,
    api_diff_df: Optional[pd.DataFrame] = None,
    api_rainfall_df: Optional[pd.DataFrame] = None,
    hist_flow_df: Optional[pd.DataFrame] = None,
    hist_level_df: Optional[pd.DataFrame] = None,
    hist_groundwater_df: Optional[pd.DataFrame] = None,
    differential_column: str = "differential",
    min_differential: Optional[float] = None,
    max_differential: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge historical and API data, resample to hourly frequency, and clean.
    
    This function:
    1. Normalizes rainfall column names
    2. Joins historical differential with historical rainfall, flow, level, groundwater
    3. Merges API data (API data takes precedence for overlapping dates)
    4. Resamples to hourly frequency
    5. Cleans spurious values (negative rainfall, outliers, spikes)
    
    Args:
        hist_diff_df: Historical differential data
        hist_rainfall_df: Historical rainfall data
        api_diff_df: Recent differential from API (optional)
        api_rainfall_df: Recent rainfall from API (optional)
        hist_flow_df: Historical flow data (optional)
        hist_level_df: Historical level data (optional)
        hist_groundwater_df: Historical groundwater data (optional)
        differential_column: Name of differential column
        min_differential: Minimum valid differential value (default: -0.1)
        max_differential: Maximum valid differential value (default: 1.5)
        verbose: Whether to print progress
        
    Returns:
        Cleaned DataFrame with hourly frequency
    """
    rainfall_df = hist_rainfall_df.copy()

    # Normalize rainfall column names (extract station name from full column name)
    # and keep track of which columns are rainfall so we can treat them correctly
    rainfall_column_map = {}
    for col in rainfall_df.columns:
        if "mm_" in col:
            parts = col.split("mm_")
            if len(parts) > 1:
                after_mm = parts[1]
                if "-" in after_mm:
                    # Format: rainfall_mm_Osney-Lock-rainfall-15min-Qualified -> Osney
                    station_name = after_mm.split("-")[0]
                    rainfall_column_map[col] = station_name
                else:
                    # Format: rainfall_mm_Abingdon -> Abingdon (simple name)
                    station_name = after_mm
                    rainfall_column_map[col] = station_name

    if rainfall_column_map:
        rainfall_df = rainfall_df.rename(columns=rainfall_column_map)
        rainfall_station_names = list(rainfall_column_map.values())
    else:
        # Fall back to treating all non-differential columns as rainfall
        rainfall_station_names = [c for c in rainfall_df.columns if c != differential_column]
    
    # Ensure timezone consistency
    if rainfall_df.index.tz is None:
        rainfall_df = rainfall_df.tz_localize('UTC')
    
    # Join historical data
    df = hist_diff_df.join(rainfall_df, how="inner")
    
    # Add flow, level, and groundwater data if available
    if hist_flow_df is not None and not hist_flow_df.empty:
        # Ensure timezone consistency
        flow_df = hist_flow_df.copy()
        if flow_df.index.tz is None:
            flow_df = flow_df.tz_localize('UTC')
        df = df.join(flow_df, how='left')
    
    if hist_level_df is not None and not hist_level_df.empty:
        # Ensure timezone consistency
        level_df = hist_level_df.copy()
        if level_df.index.tz is None:
            level_df = level_df.tz_localize('UTC')
        df = df.join(level_df, how='left')
    
    if hist_groundwater_df is not None and not hist_groundwater_df.empty:
        # Ensure timezone consistency
        groundwater_df = hist_groundwater_df.copy()
        if groundwater_df.index.tz is None:
            groundwater_df = groundwater_df.tz_localize('UTC')
        df = df.join(groundwater_df, how='left')
    
    # Merge with API data (API data takes precedence for overlapping dates)
    if api_rainfall_df is not None:
        df = df.combine_first(api_rainfall_df)
    if api_diff_df is not None:
        df = df.combine_first(api_diff_df)
    
    # Resample to hourly frequency
    aggregation_rules = {}
    for col in df.columns:
        if col == differential_column:
            aggregation_rules[col] = "mean"  # Average differential
        elif col in rainfall_station_names:
            aggregation_rules[col] = "sum"  # Sum rainfall
        elif col.startswith("flow") or col.startswith("level") or col.startswith("groundwater"):
            aggregation_rules[col] = "mean"  # Average flow/level/groundwater
        else:
            aggregation_rules[col] = "mean"  # Default to mean
    
    df_hourly = df.resample('1h').agg(aggregation_rules)
    df = df_hourly.copy()
    
    # Clean data - remove NaN differentials
    df = df.dropna(subset=[differential_column])
    
    # Fill NaN values selectively:
    # - Rainfall: fill with 0 (no rain)
    # - Flow, level, groundwater: forward fill then backward fill (preserve actual values)
    # - Other columns: fill with 0
    rainfall_cols = [col for col in df.columns if col in rainfall_station_names]
    flow_cols = [col for col in df.columns if col.startswith("flow")]
    level_cols = [col for col in df.columns if col.startswith("level")]
    groundwater_cols = [col for col in df.columns if col.startswith("groundwater")]
    
    # Fill rainfall with 0
    for col in rainfall_cols:
        df[col] = df[col].fillna(0)
    
    # Fill flow, level, groundwater with forward fill then backward fill
    # This preserves actual measurements rather than filling with 0
    for col in flow_cols + level_cols + groundwater_cols:
        if col in df.columns:
            non_null_before = df[col].notna().sum()
            df[col] = df[col].ffill().bfill()
            non_null_after = df[col].notna().sum()
            if verbose and col.startswith('groundwater'):
                print(f"  Groundwater {col}: {non_null_before} -> {non_null_after} non-null, range: {df[col].min():.3f} to {df[col].max():.3f}")
    
    # Fill any remaining columns with 0
    remaining_cols = [col for col in df.columns if col not in rainfall_cols + flow_cols + level_cols + groundwater_cols + [differential_column]]
    for col in remaining_cols:
        df[col] = df[col].fillna(0)
    
    # Clean spurious rainfall values (ONLY for rainfall columns)
    rainfall_cols_clean = rainfall_cols
    if verbose:
        print(f"Cleaning rainfall data for {len(rainfall_cols_clean)} stations...")
    
    for col in rainfall_cols_clean:
        # Remove negative rainfall (sensor errors)
        negative_count = (df[col] < 0).sum()
        if negative_count > 0 and verbose:
            print(f"  {col}: Removing {negative_count} negative values")
        df.loc[df[col] < 0, col] = 0
        
        # Cap extreme rainfall values (likely sensor errors)
        # Typical max is around 50mm/hour in UK
        high_count = (df[col] > 50).sum()
        if high_count > 0 and verbose:
            print(f"  {col}: Capping {high_count} values >50mm/hour")
        df.loc[df[col] > 50, col] = 50
    
    # Clean spurious flow/level values (remove negatives, but don't cap - these are real measurements)
    flow_cols_clean = [col for col in df.columns if col.startswith('flow')]
    level_cols_clean = [col for col in df.columns if col.startswith('level')]
    for col in flow_cols_clean + level_cols_clean:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0 and verbose:
            print(f"  {col}: Removing {negative_count} negative values")
        df.loc[df[col] < 0, col] = 0
    
    if verbose:
        print("✓ Rainfall cleaning complete")
    
    # Clean differential values with location-specific thresholds
    df = _clean_differential(
        df, 
        differential_column, 
        min_val=min_differential if min_differential is not None else -0.1,
        max_val=max_differential if max_differential is not None else 1.5,
        verbose=verbose
    )
    
    return df


def _clean_differential(
    df: pd.DataFrame,
    differential_column: str = 'differential',
    min_val: float = -0.1,
    max_val: float = 1.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Clean differential values by removing outliers and spikes.
    
    Args:
        df: DataFrame with differential column
        differential_column: Column name
        min_val: Minimum valid value
        max_val: Maximum valid value
        verbose: Whether to print progress
        
    Returns:
        Cleaned DataFrame
    """
    # Remove extreme values
    original_len = len(df)
    df = df[(df[differential_column] > min_val) & (df[differential_column] <= max_val)].copy()
    removed = original_len - len(df)
    if removed > 0 and verbose:
        print(f"  Removed {removed} extreme differential values")
    
    # Remove spikes: large change followed by opposite large change
    diff_series = df[differential_column]
    changes = diff_series.diff()
    next_changes = changes.shift(-1)
    
    spike_mask = (
        (np.abs(changes) > 0.5) & 
        (np.abs(next_changes) > 0.5) & 
        (np.sign(changes) != np.sign(next_changes))
    )
    
    if spike_mask.sum() > 0:
        if verbose:
            print(f"  Removing {spike_mask.sum()} spike values in differential")
        df.loc[spike_mask, differential_column] = np.nan
        df[differential_column] = df[differential_column].interpolate(method='linear')
    
    # Remove statistical outliers using rolling window
    window = 6
    rolling_mean = df[differential_column].rolling(window=window, center=True).mean()
    rolling_std = df[differential_column].rolling(window=window, center=True).std()
    outliers = np.abs(df[differential_column] - rolling_mean) > (3 * rolling_std)
    
    if outliers.sum() > 0 and verbose:
        print(f"  Smoothing {outliers.sum()} statistical outliers")
    
    if outliers.sum() > 0:
        df.loc[outliers, differential_column] = df[differential_column].rolling(
            window=window, center=True
        ).median()[outliers]
        df[differential_column] = df[differential_column].rolling(
            window=3, center=True
        ).mean().fillna(df[differential_column])
    
    return df
