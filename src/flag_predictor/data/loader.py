"""
Data loading utilities for historical CSV data.

Functions for loading historical differential, rainfall, flow, level, and groundwater data from CSV files.
"""

import os
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..config import (
    LocationConfig, 
    get_location_config,
    WALLINGFORD_RAINFALL_STATION_NAMES
)


def load_historical_differential(
    config: LocationConfig,
    project_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load historical differential data from CSV file.
    
    Args:
        config: LocationConfig object with file path and column info
        project_root: Root directory of project (defaults to auto-detect)
        
    Returns:
        DataFrame with 'differential' column and timestamp index
    """
    if project_root is None:
        # Try to find project root
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    file_path = project_root / config.historical_data_file
    
    if not file_path.exists():
        raise FileNotFoundError(f"Historical data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Select and rename the differential column
    df = df[[config.differential_column]]
    df = df.rename(columns={config.differential_column: 'differential'})
    
    print(f"✓ Loaded {config.display_name} historical data: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def load_historical_rainfall(
    rainfall_dir: str = 'data/rainfall_training_data/',
    project_root: Optional[Path] = None,
    location: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load historical rainfall data from CSV files.
    
    Args:
        rainfall_dir: Directory containing rainfall CSV files
        project_root: Root directory of project
        location: Location name ('wallingford' includes additional stations, others exclude them)
        
    Returns:
        DataFrame with rainfall stations as columns (filtered by location)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    rainfall_path = project_root / rainfall_dir
    csv_files = glob.glob(str(rainfall_path / '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {rainfall_path}")
    
    # Define station filters by location
    wallingford_stations = set(WALLINGFORD_RAINFALL_STATION_NAMES)
    godstow_excluded_stations = {'Bicester', 'Grimsbury'}
    
    if location:
        location_lower = location.lower()
        if location_lower != 'wallingford':
            if verbose:
                print(f"Filtering out Wallingford-specific stations for {location}...")
        if location_lower == 'godstow':
            if verbose:
                print(f"Filtering out Bicester and Grimsbury for {location}...")
    
    rainfall_dfs = {}
    print(f"Loading historical rainfall data from {len(csv_files)} files...")
    
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Skip Wallingford-specific stations if location is not 'wallingford'
        if location and location.lower() != 'wallingford':
            if file_name in wallingford_stations:
                if verbose:
                    print(f"  ⊘ Skipping {file_name} (Wallingford-specific station)")
                continue
        
        # Skip Bicester and Grimsbury for Godstow
        if location and location.lower() == 'godstow':
            if file_name in godstow_excluded_stations:
                if verbose:
                    print(f"  ⊘ Skipping {file_name} (excluded for Godstow)")
                continue
        
        try:
            df = pd.read_csv(csv_file, dtype=str, on_bad_lines='warn')
        except Exception as e:
            print(f"  ✗ Error reading {file_name}: {e}")
            continue
        
        if 'dateTime' not in df.columns or 'value' not in df.columns:
            continue
        
        df = df[['dateTime', 'value']]
        df = df.rename(columns={
            'dateTime': 'timestamp',
            'value': f'rainfall_mm_{file_name}'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[f'rainfall_mm_{file_name}'] = pd.to_numeric(
            df[f'rainfall_mm_{file_name}'],
            errors='coerce'
        ).fillna(0)
        df = df.set_index('timestamp')
        rainfall_dfs[file_name] = df
    
    if not rainfall_dfs:
        raise ValueError("No valid rainfall data found")
    
    combined = pd.concat(rainfall_dfs.values(), axis=1)
    print(f"\n✓ Combined historical rainfall: {combined.shape}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined


def load_historical_flow(
    flow_dir: str = 'data/flow_training_data/',
    project_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load all historical flow data from CSV files.
    
    Args:
        flow_dir: Directory containing flow CSV files
        project_root: Root directory of project
        
    Returns:
        DataFrame with all flow stations as columns
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    flow_path = project_root / flow_dir
    csv_files = glob.glob(str(flow_path / '*.csv'))
    
    if not csv_files:
        print(f"⚠ No flow CSV files found in {flow_path}")
        return pd.DataFrame()
    
    flow_dfs = {}
    print(f"Loading historical flow data from {len(csv_files)} files...")
    
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        # Extract station name (e.g., "Farmoor-flow-15min-Qualified" -> "Farmoor")
        station_name = file_name.split('-flow')[0]
        
        try:
            df = pd.read_csv(csv_file, dtype=str, on_bad_lines='warn')
        except Exception as e:
            print(f"  ✗ Error reading {file_name}: {e}")
            continue
        
        if 'dateTime' not in df.columns or 'value' not in df.columns:
            continue
        
        df = df[['dateTime', 'value']]
        df = df.rename(columns={
            'dateTime': 'timestamp',
            'value': f'flow_m3s_{station_name}'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[f'flow_m3s_{station_name}'] = pd.to_numeric(
            df[f'flow_m3s_{station_name}'],
            errors='coerce'
        )
        df = df.set_index('timestamp')
        flow_dfs[station_name] = df
    
    if not flow_dfs:
        print("  ⚠ No valid flow data found")
        return pd.DataFrame()
    
    combined = pd.concat(flow_dfs.values(), axis=1)
    print(f"✓ Combined historical flow: {combined.shape}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined


def load_historical_level(
    level_dir: str = 'data/level_training_data/',
    project_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load all historical level data from CSV files.
    
    Args:
        level_dir: Directory containing level CSV files
        project_root: Root directory of project
        
    Returns:
        DataFrame with all level stations as columns
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    level_path = project_root / level_dir
    csv_files = glob.glob(str(level_path / '*.csv'))
    
    if not csv_files:
        print(f"⚠ No level CSV files found in {level_path}")
        return pd.DataFrame()
    
    level_dfs = {}
    print(f"Loading historical level data from {len(csv_files)} files...")
    
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        # Extract station name (e.g., "Hannington-Bridge-level-15min-Qualified" -> "Hannington-Bridge")
        station_name = file_name.split('-level')[0]
        
        try:
            df = pd.read_csv(csv_file, dtype=str, on_bad_lines='warn')
        except Exception as e:
            print(f"  ✗ Error reading {file_name}: {e}")
            continue
        
        if 'dateTime' not in df.columns or 'value' not in df.columns:
            continue
        
        df = df[['dateTime', 'value']]
        df = df.rename(columns={
            'dateTime': 'timestamp',
            'value': f'level_m_{station_name}'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[f'level_m_{station_name}'] = pd.to_numeric(
            df[f'level_m_{station_name}'],
            errors='coerce'
        )
        df = df.set_index('timestamp')
        level_dfs[station_name] = df
    
    if not level_dfs:
        print("  ⚠ No valid level data found")
        return pd.DataFrame()
    
    combined = pd.concat(level_dfs.values(), axis=1)
    print(f"✓ Combined historical level: {combined.shape}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined


def load_historical_groundwater(
    groundwater_dir: str = 'data/groundwater_training_data/',
    project_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load all historical groundwater data from CSV files.
    
    Args:
        groundwater_dir: Directory containing groundwater CSV files
        project_root: Root directory of project
        
    Returns:
        DataFrame with all groundwater stations as columns
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    groundwater_path = project_root / groundwater_dir
    csv_files = glob.glob(str(groundwater_path / '*.csv'))
    
    if not csv_files:
        print(f"⚠ No groundwater CSV files found in {groundwater_path}")
        return pd.DataFrame()
    
    groundwater_dfs = {}
    print(f"Loading historical groundwater data from {len(csv_files)} files...")
    
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        # Extract station name (e.g., "SP50_72-groundwater-15min-Qualified" -> "SP50_72")
        station_name = file_name.split('-groundwater')[0]
        
        try:
            df = pd.read_csv(csv_file, dtype=str, on_bad_lines='warn')
        except Exception as e:
            print(f"  ✗ Error reading {file_name}: {e}")
            continue
        
        if 'dateTime' not in df.columns or 'value' not in df.columns:
            continue
        
        df = df[['dateTime', 'value']]
        df = df.rename(columns={
            'dateTime': 'timestamp',
            'value': f'groundwater_mAOD_{station_name}'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[f'groundwater_mAOD_{station_name}'] = pd.to_numeric(
            df[f'groundwater_mAOD_{station_name}'],
            errors='coerce'
        )
        df = df.set_index('timestamp')
        groundwater_dfs[station_name] = df
    
    if not groundwater_dfs:
        print("  ⚠ No valid groundwater data found")
        return pd.DataFrame()
    
    combined = pd.concat(groundwater_dfs.values(), axis=1)
    print(f"✓ Combined historical groundwater: {combined.shape}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined


def load_all_historical_data(
    location: str = "isis",
    project_root: Optional[Path] = None,
    use_extra_sources: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all historical data for a location.
    
    Args:
        location: Location name ('isis' or 'godstow')
        project_root: Root directory of project
        
    Returns:
        Tuple of (differential_df, rainfall_df, flow_df, level_df, groundwater_df)
        All dataframes are aligned to end on the last common timestamp.
    """
    config = get_location_config(location)
    
    differential_df = load_historical_differential(config, project_root)
    rainfall_df = load_historical_rainfall(project_root=project_root, location=location)

    if use_extra_sources:
        flow_df = load_historical_flow(project_root=project_root)
        level_df = load_historical_level(project_root=project_root)
        groundwater_df = load_historical_groundwater(project_root=project_root)
    else:
        # Empty frames mean "no extra sources" downstream
        flow_df = pd.DataFrame()
        level_df = pd.DataFrame()
        groundwater_df = pd.DataFrame()
    
    # Ensure all dataframes have consistent timezones (UTC) before alignment
    # This is critical for intersection to work correctly
    if differential_df.index.tz is None:
        differential_df = differential_df.tz_localize('UTC')
    if rainfall_df.index.tz is None:
        rainfall_df = rainfall_df.tz_localize('UTC')
    if not flow_df.empty and flow_df.index.tz is None:
        flow_df = flow_df.tz_localize('UTC')
    if not level_df.empty and level_df.index.tz is None:
        level_df = level_df.tz_localize('UTC')
    if not groundwater_df.empty and groundwater_df.index.tz is None:
        groundwater_df = groundwater_df.tz_localize('UTC')
    
    # Align all dataframes to end on the last common timestamp
    all_indices = [differential_df.index, rainfall_df.index]
    if not flow_df.empty:
        all_indices.append(flow_df.index)
    if not level_df.empty:
        all_indices.append(level_df.index)
    if not groundwater_df.empty:
        all_indices.append(groundwater_df.index)
    
    # Find common timestamps
    common_times = all_indices[0]
    for idx in all_indices[1:]:
        common_times = common_times.intersection(idx)
    
    if not common_times.empty:
        common_end = common_times.max()
        differential_df = differential_df.loc[:common_end]
        rainfall_df = rainfall_df.loc[:common_end]
        if not flow_df.empty:
            flow_df = flow_df.loc[:common_end]
        if not level_df.empty:
            level_df = level_df.loc[:common_end]
        if not groundwater_df.empty:
            groundwater_df = groundwater_df.loc[:common_end]
    
    return differential_df, rainfall_df, flow_df, level_df, groundwater_df
