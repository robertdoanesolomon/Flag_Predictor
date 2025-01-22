#!/usr/bin/env python3
"""
List rainfall stations used for each location (ISIS, Godstow, Wallingford).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from flag_predictor.config import (
    RAINFALL_STATION_NAMES,
    RAINFALL_STATION_COORDINATES,
    WALLINGFORD_RAINFALL_STATION_NAMES,
    WALLINGFORD_RAINFALL_STATION_COORDINATES,
)

def get_stations_for_location(location: str):
    """Get list of rainfall stations for a given location."""
    location_lower = location.lower()
    
    # Start with default stations
    stations = list(RAINFALL_STATION_NAMES)
    
    if location_lower == 'wallingford':
        # Add Wallingford-specific stations
        stations.extend(WALLINGFORD_RAINFALL_STATION_NAMES)
    elif location_lower == 'godstow':
        # Remove Bicester and Grimsbury for Godstow
        stations = [s for s in stations if s not in {'Bicester', 'Grimsbury'}]
    # For ISIS, use all default stations (no changes)
    
    return sorted(stations)

def main():
    """List stations for each location."""
    locations = ['isis', 'godstow', 'wallingford']
    
    print("=" * 80)
    print("RAINFALL STATIONS BY LOCATION")
    print("=" * 80)
    
    for location in locations:
        stations = get_stations_for_location(location)
        location_display = location.upper()
        
        print(f"\n{location_display} ({len(stations)} stations):")
        print("-" * 80)
        
        # Group stations for better readability
        for i, station in enumerate(stations, 1):
            # Check if it's a Wallingford-specific station
            is_wallingford_only = station in WALLINGFORD_RAINFALL_STATION_NAMES
            marker = " [Wallingford-only]" if is_wallingford_only else ""
            print(f"  {i:2d}. {station}{marker}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    isis_stations = set(get_stations_for_location('isis'))
    godstow_stations = set(get_stations_for_location('godstow'))
    wallingford_stations = set(get_stations_for_location('wallingford'))
    
    print(f"\nISIS:        {len(isis_stations)} stations")
    print(f"Godstow:     {len(godstow_stations)} stations (excludes Bicester, Grimsbury)")
    print(f"Wallingford: {len(wallingford_stations)} stations (includes 6 Wallingford-specific)")
    
    print(f"\nStations only in ISIS:        {sorted(isis_stations - godstow_stations - wallingford_stations)}")
    print(f"Stations only in Godstow:     {sorted(godstow_stations - isis_stations - wallingford_stations)}")
    print(f"Stations only in Wallingford: {sorted(wallingford_stations - isis_stations - godstow_stations)}")
    print(f"Stations in all locations:    {sorted(isis_stations & godstow_stations & wallingford_stations)}")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
