#!/usr/bin/env python3
"""
Download all Wallingford-specific rainfall stations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from flag_predictor.data.download_rainfall import download_station_rainfall, RAINFALL_HYDROLOGY_MEASURES
import time

# Wallingford-specific stations
WALLINGFORD_STATIONS = [
    'Stanford',
    'Abingdon',
    'Wheatley',
    'Benson',
    'Aylesbury',
    'Cleeve',
]

def main():
    """Download all Wallingford stations."""
    print("=" * 70)
    print("Downloading Wallingford-Specific Rainfall Stations")
    print("=" * 70)
    print(f"\nStations to download: {', '.join(WALLINGFORD_STATIONS)}")
    print("\nNote: Existing files will be skipped to avoid overwriting")
    print("=" * 70)
    
    results = {}
    for i, station_name in enumerate(WALLINGFORD_STATIONS):
        if station_name not in RAINFALL_HYDROLOGY_MEASURES:
            print(f"\n✗ {station_name}: Not found in RAINFALL_HYDROLOGY_MEASURES")
            results[station_name] = False
            continue
        
        measure_id = RAINFALL_HYDROLOGY_MEASURES[station_name]
        print(f"\n[{i+1}/{len(WALLINGFORD_STATIONS)}] Downloading {station_name}...")
        print(f"  Measure ID: {measure_id}")
        
        success = download_station_rainfall(
            station_name=station_name,
            measure_id=measure_id,
            min_date='2017-02-02',  # Same start date as Osney-Lock
            verbose=True
        )
        
        results[station_name] = success
        
        # Be polite to the API - delay between requests
        if i < len(WALLINGFORD_STATIONS) - 1:
            print(f"  Waiting 2 seconds before next download...")
            time.sleep(2)
    
    print("\n" + "=" * 70)
    print("Download Summary:")
    print("=" * 70)
    for station_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {station_name}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\n{successful}/{len(WALLINGFORD_STATIONS)} stations downloaded successfully")
    print("=" * 70)

if __name__ == '__main__':
    main()
