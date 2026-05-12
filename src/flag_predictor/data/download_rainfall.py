"""
Download historical qualified rainfall data from Environment Agency Hydrology API.

Uses the CSV endpoint with date filtering:
https://environment.data.gov.uk/hydrology/id/measures/{measure-id}-rainfall-t-900-mm-qualified/readings.csv?min-date=YYYY-MM-DD
"""

import requests
from pathlib import Path
from typing import Dict, Optional
import time

# Mapping of station names to Hydrology API measure IDs (station UUIDs)
RAINFALL_HYDROLOGY_MEASURES = {
    'Bicester': '5466e4ec-fcb6-47f5-bb1f-fd1a54e108da',
    'Bourton': 'bf217aec-f418-42a6-9b0f-a3de46d7c592',
    'Byfield': 'fb3f99de-1e0a-4b91-9071-41bfe45b12f6',
    'Chipping-Norton': 'bcdb1cad-f070-49dc-9eb6-1512273b1119',
    'Eynsham': 'de8bb5b8-45a5-4282-8028-6268e5af1d30',
    'Grimsbury': '0eba4d80-4108-49bc-9285-074e9776e57d',
    'Osney-Lock': '5086b349-e04b-4f8e-b8f1-fa92b21078d0_256230TP',
    'Rapsgate': 'eba07158-07a9-4649-aeda-739af9b74772',
    'Shorncote': '6a43c101-4972-4491-8e67-3fa99a097cf1_248332TP',
    'St-Johns': '94fd6f10-20e1-4b72-915c-5364725b14d3',
    'Stowell-Park': '2e68c090-ed7e-4146-8cf8-18ee56d2ba27',
    'Swindon': 'ecc40362-51bd-44e5-9dd9-c117d26fdec5',
    'Worsham': 'bed0ef93-6faa-4f98-af56-bfb3fed3ed61_253861TP',
    # Wallingford-specific stations
    'Stanford': 'da800212-08da-488a-9bf3-afcad2aec991_260221TP',
    'Abingdon': 'fe3efb87-062c-4880-9dc3-7dd8f335a6d1',
    'Wheatley': '7eb546a1-8663-4d07-a9cc-e7b5f6e13006',
    'Benson': '1ad8ff76-625d-4ff1-82bd-9efcc5d41b2f_264254TP',
    'Aylesbury': 'd8333bc4-b9f8-4984-bdc6-77909c81e473',
    'Cleeve': '11070879-1ccb-4a52-bb60-5cf74a549431',
}

BASE_URL = "https://environment.data.gov.uk/hydrology/id/measures"


def download_station_rainfall(
    station_name: str,
    measure_id: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> bool:
    """
    Download qualified rainfall data for a single station as CSV.
    
    Args:
        station_name: Name of the rainfall station
        measure_id: Hydrology API measure ID (station UUID)
        min_date: Minimum date in YYYY-MM-DD format (filters at API level)
        max_date: Maximum date in YYYY-MM-DD format (optional)
        output_dir: Directory to save CSV file (defaults to data/rainfall_training_data/)
        project_root: Project root directory
        verbose: Whether to print progress
        
    Returns:
        True if successful, False otherwise
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    if output_dir is None:
        output_dir = project_root / 'data' / 'rainfall_training_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build CSV API URL
    # API limits to 100,000 rows per request, so we need pagination
    # Format: {measure_id}-rainfall-t-900-mm-qualified/readings.csv
    base_url = f"{BASE_URL}/{measure_id}-rainfall-t-900-mm-qualified/readings.csv"
    
    if verbose:
        print(f"Downloading {station_name}...")
        if min_date:
            print(f"  Filtering from {min_date} onwards (using API min-date parameter)")
        if max_date:
            print(f"  Filtering until {max_date} (using API max-date parameter)")
    
    try:
        output_file = output_dir / f"{station_name}-rainfall-15min-Qualified.csv"
        
        # Check if file already exists - never overwrite
        if output_file.exists():
            if verbose:
                print(f"  ⚠ File already exists: {output_file}")
                print(f"  Skipping download to avoid overwriting existing data")
            return True
        
        # Download with pagination (100,000 rows per page)
        limit = 100000
        offset = 0
        page = 1
        total_rows = 0
        
        with open(output_file, 'wb') as f:
            while True:
                if verbose:
                    print(f"  Downloading page {page} (offset {offset:,})...", end=" ", flush=True)
                
                # Build URL with pagination and optional date filtering
                url = f"{base_url}?_limit={limit}&_offset={offset}"
                if min_date:
                    url += f"&min-date={min_date}"
                if max_date:
                    url += f"&max-date={max_date}"
                
                response = requests.get(url, timeout=300, stream=True)
                response.raise_for_status()
                
                # Read the response
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                
                # Check if we got any data
                if not content or len(content.strip()) == 0:
                    break
                
                # Decode to check number of lines
                text_content = content.decode('utf-8')
                lines = text_content.strip().split('\n')
                
                # First page: write everything including header
                if offset == 0:
                    f.write(content)
                    rows_in_page = len(lines) - 1  # Subtract header
                else:
                    # Subsequent pages: skip header (first line) and write rest
                    if len(lines) > 1:
                        # Write all lines except the first (header)
                        remaining_content = '\n'.join(lines[1:]) + '\n'
                        f.write(remaining_content.encode('utf-8'))
                        rows_in_page = len(lines) - 1
                    else:
                        rows_in_page = 0
                
                total_rows += rows_in_page
                
                if verbose:
                    print(f"✓ {rows_in_page:,} rows")
                
                # If we got fewer rows than the limit, we're done
                if rows_in_page < limit:
                    break
                
                # Move to next page
                offset += limit
                page += 1
                
                # Small delay between pages
                time.sleep(0.5)
        
        # Check file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        if verbose:
            print(f"  ✓ Complete: {total_rows:,} total rows")
            print(f"  ✓ Saved to {output_file}")
            print(f"  ✓ File size: {file_size_mb:.2f} MB")
        
        return True
        
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"  ✗ Error downloading {station_name}: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"  ✗ Unexpected error for {station_name}: {e}")
        return False


def download_all_rainfall(
    stations: Optional[Dict[str, str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    delay: float = 2.0,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Download qualified rainfall data for all stations.
    
    Args:
        stations: Dictionary mapping station names to measure IDs
                 (defaults to RAINFALL_HYDROLOGY_MEASURES)
        min_date: Minimum date in YYYY-MM-DD format (filters at API level)
        max_date: Maximum date in YYYY-MM-DD format (optional)
        output_dir: Directory to save CSV files
        project_root: Project root directory
        delay: Delay between requests in seconds (to be polite to API)
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping station names to success status
    """
    if stations is None:
        stations = RAINFALL_HYDROLOGY_MEASURES
    
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    if output_dir is None:
        output_dir = project_root / 'data' / 'rainfall_training_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Downloading historical qualified rainfall data")
        print(f"{'='*70}")
        if min_date:
            print(f"Minimum date: {min_date}")
        if max_date:
            print(f"Maximum date: {max_date}")
        print(f"Output directory: {output_dir}")
        print(f"Stations: {len(stations)}")
        print()
    
    results = {}
    
    for i, (station_name, measure_id) in enumerate(stations.items(), 1):
        if verbose:
            print(f"[{i}/{len(stations)}] ", end="")
        
        success = download_station_rainfall(
            station_name=station_name,
            measure_id=measure_id,
            min_date=min_date,
            max_date=max_date,
            output_dir=output_dir,
            project_root=project_root,
            verbose=verbose
        )
        
        results[station_name] = success
        
        # Be polite to the API
        if delay > 0 and i < len(stations):
            time.sleep(delay)
    
    if verbose:
        successful = sum(1 for v in results.values() if v)
        print(f"\n{'='*70}")
        print(f"Download complete: {successful}/{len(stations)} stations successful")
        print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download historical qualified rainfall data from Environment Agency Hydrology API'
    )
    parser.add_argument(
        '--min-date',
        type=str,
        help='Minimum date in YYYY-MM-DD format (filters at API level - efficient!)'
    )
    parser.add_argument(
        '--max-date',
        type=str,
        help='Maximum date in YYYY-MM-DD format (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for CSV files (defaults to data/rainfall_training_data/)'
    )
    parser.add_argument(
        '--station',
        type=str,
        help='Download data for a single station only'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between API requests in seconds (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    if args.station:
        # Download single station
        if args.station not in RAINFALL_HYDROLOGY_MEASURES:
            print(f"Error: Unknown station '{args.station}'")
            print(f"Available stations: {', '.join(RAINFALL_HYDROLOGY_MEASURES.keys())}")
            exit(1)
        
        measure_id = RAINFALL_HYDROLOGY_MEASURES[args.station]
        download_station_rainfall(
            station_name=args.station,
            measure_id=measure_id,
            min_date=args.min_date,
            max_date=args.max_date,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            verbose=True
        )
    else:
        # Download all stations
        download_all_rainfall(
            min_date=args.min_date,
            max_date=args.max_date,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            delay=args.delay,
            verbose=True
        )
