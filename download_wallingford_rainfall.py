#!/usr/bin/env python3
"""
Download rainfall data for Wallingford-specific stations.

This script downloads historical rainfall data for the 6 Wallingford-specific stations
using the Flood Monitoring API (for recent data) and attempts to find Hydrology API
measure IDs for full historical downloads.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from flag_predictor.data.api import fetch_rainfall_data

# Wallingford-specific stations with TP IDs
WALLINGFORD_STATIONS = {
    'Stanford': '260221TP',
    'Abingdon': '261021TP',
    'Wheatley': '263541TP',
    'Benson': '264254TP',
    'Aylesbury': '261923TP',
    'Cleeve': '264845TP',
}

FLOOD_MONITORING_BASE = "http://environment.data.gov.uk/flood-monitoring/id/measures"
HYDROLOGY_BASE = "https://environment.data.gov.uk/hydrology/id/measures"


def download_from_flood_monitoring_api(tp_id: str, station_name: str, output_dir: Path, days_back: int = 365):
    """
    Download recent rainfall data from Flood Monitoring API.
    
    This gets the last N days of data, which is useful for training.
    For full historical data, you'll need to find the Hydrology API UUID.
    """
    url = f"{FLOOD_MONITORING_BASE}/{tp_id}-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000"
    
    print(f"Downloading {station_name} ({tp_id}) from Flood Monitoring API...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'items' not in data or not data['items']:
            print(f"  ✗ No data found")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(data['items'])
        if 'dateTime' not in df.columns or 'value' not in df.columns:
            print(f"  ✗ Invalid data format")
            return False
        
        df = df[['dateTime', 'value']]
        df.rename(columns={'dateTime': 'dateTime', 'value': 'value'}, inplace=True)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        
        # Filter to last N days if specified
        if days_back:
            cutoff_date = pd.Timestamp.now(tz=df['dateTime'].dt.tz) - pd.Timedelta(days=days_back)
            df = df[df['dateTime'] >= cutoff_date]
        
        # Save as CSV (matching format of other rainfall data)
        # Use capitalized name to match config station names
        output_file = output_dir / f"{station_name}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(df)} records to {output_file}")
        print(f"    Date range: {df['dateTime'].min()} to {df['dateTime'].max()}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def find_hydrology_measure_id(tp_id: str, station_name: str):
    """
    Attempt to find the Hydrology API measure ID for a TP ID.
    
    This searches the Hydrology API for measures matching the TP ID.
    """
    print(f"\nLooking up Hydrology API measure ID for {station_name} ({tp_id})...")
    
    # Try searching the Hydrology API
    # Note: This is a simplified search - the actual API might require different parameters
    search_url = f"{HYDROLOGY_BASE}?label={tp_id}"
    
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                measure_id = data['items'][0].get('@id', '').split('/')[-1]
                print(f"  Found: {measure_id}")
                return measure_id
    except Exception as e:
        print(f"  Search failed: {e}")
    
    print(f"  Could not find Hydrology API measure ID automatically")
    print(f"  You may need to find it manually from:")
    print(f"  https://environment.data.gov.uk/hydrology/id/measures")
    return None


def main():
    """Download all Wallingford-specific rainfall stations."""
    project_root = Path(__file__).parent
    output_dir = project_root / 'data' / 'rainfall_training_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading Wallingford-Specific Rainfall Stations")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nNote: This downloads recent data (last 365 days) from Flood Monitoring API.")
    print("For full historical data, you'll need to find the Hydrology API measure IDs.")
    print("=" * 70)
    
    results = {}
    for station_name, tp_id in WALLINGFORD_STATIONS.items():
        success = download_from_flood_monitoring_api(tp_id, station_name, output_dir, days_back=365)
        results[station_name] = success
        
        # Also try to find Hydrology API measure ID
        hydrology_id = find_hydrology_measure_id(tp_id, station_name)
        if hydrology_id:
            print(f"  → Update download_rainfall.py with: '{station_name}': '{hydrology_id}'")
    
    print("\n" + "=" * 70)
    print("Download Summary:")
    print("=" * 70)
    for station_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {station_name}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Check the downloaded CSV files in:", output_dir)
    print("2. For full historical data, find Hydrology API measure IDs and update")
    print("   src/flag_predictor/data/download_rainfall.py")
    print("3. Then run: python -m flag_predictor.data.download_rainfall --station <name>")


if __name__ == '__main__':
    main()
