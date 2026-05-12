#!/usr/bin/env python3
"""
Find Hydrology API measure IDs (UUIDs) for rainfall stations from their TP IDs.

This script queries the Hydrology API to find the measure IDs needed for
downloading full historical data.
"""

import requests
import json

# Wallingford-specific stations with TP IDs
WALLINGFORD_STATIONS = {
    'Stanford': '260221TP',
    'Abingdon': '261021TP',
    'Wheatley': '263541TP',
    'Benson': '264254TP',
    'Aylesbury': '261923TP',
    'Cleeve': '264845TP',
}

HYDROLOGY_BASE = "https://environment.data.gov.uk/hydrology/id/measures"
FLOOD_MONITORING_BASE = "http://environment.data.gov.uk/flood-monitoring/id/measures"


def find_hydrology_id_from_station(station_ref: str):
    """
    Find Hydrology API measure ID from Flood Monitoring station reference.
    
    Args:
        station_ref: Station reference (e.g., '260221TP')
        
    Returns:
        Hydrology API measure ID (UUID) if found, None otherwise
    """
    print(f"\nLooking up {station_ref}...")
    
    # Step 1: Get station info from Flood Monitoring API
    try:
        station_url = f"{FLOOD_MONITORING_BASE.replace('/measures', '/stations')}/{station_ref}"
        response = requests.get(station_url, timeout=10)
        if response.status_code == 200:
            station_data = response.json()
            print(f"  Found station: {station_data.get('items', {}).get('label', 'N/A')}")
            
            # Step 2: Try to find Hydrology API measure by searching
            # The Hydrology API might have the station reference in metadata
            search_url = f"{HYDROLOGY_BASE}?label={station_ref}"
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    for item in data['items']:
                        measure_id = item.get('@id', '').split('/')[-1]
                        print(f"  Found measure ID: {measure_id}")
                        return measure_id
            
            # Step 3: Try searching by station reference in different ways
            # The Hydrology API uses different search parameters
            for search_term in [station_ref, station_ref.replace('TP', '')]:
                search_url = f"{HYDROLOGY_BASE}?notation={search_term}"
                response = requests.get(search_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'items' in data and len(data['items']) > 0:
                        for item in data['items']:
                            measure_id = item.get('@id', '').split('/')[-1]
                            if 'rainfall' in measure_id.lower():
                                print(f"  Found measure ID: {measure_id}")
                                return measure_id
            
            # Step 4: Try to get from the measure endpoint directly
            # Some stations might work with TP ID in format: {uuid}_{TP_ID}
            # We can try querying the Hydrology API with the TP ID
            measure_url = f"{HYDROLOGY_BASE}/{station_ref}-rainfall-t-900-mm-qualified"
            response = requests.get(measure_url, timeout=10)
            if response.status_code == 200:
                # If this works, the measure ID format might be just the TP ID
                print(f"  Measure accessible with TP ID format")
                return station_ref
            
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"  Could not find Hydrology API measure ID")
    return None


def main():
    """Find Hydrology API measure IDs for all Wallingford stations."""
    print("=" * 70)
    print("Finding Hydrology API Measure IDs for Wallingford Stations")
    print("=" * 70)
    
    results = {}
    for station_name, tp_id in WALLINGFORD_STATIONS.items():
        hydrology_id = find_hydrology_id_from_station(tp_id)
        results[station_name] = (tp_id, hydrology_id)
    
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print("\nUpdate download_rainfall.py with:")
    print()
    for station_name, (tp_id, hydrology_id) in results.items():
        if hydrology_id:
            print(f"    '{station_name}': '{hydrology_id}_{tp_id}',")
        else:
            print(f"    '{station_name}': '{tp_id}',  # TODO: Find UUID")
    
    print("\n" + "=" * 70)
    print("Alternative: Try using TP ID directly")
    print("=" * 70)
    print("Some stations might work with just the TP ID. Try running:")
    print("  python -m flag_predictor.data.download_rainfall --station Stanford")
    print("If it works, the format is just the TP ID.")


if __name__ == '__main__':
    main()
