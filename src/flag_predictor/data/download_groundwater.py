"""
Download historical qualified groundwater level data from Environment Agency Hydrology API.

Uses the CSV endpoint with date filtering:
https://environment.data.gov.uk/hydrology/id/measures/{measure-id}/readings.csv?min-date=YYYY-MM-DD
"""

import requests
from pathlib import Path
from typing import Dict, Optional
import time

# Mapping of groundwater station names to Hydrology API measure IDs (full notation)
# Format: {station-uuid}-gw-logged-i-subdaily-mAOD-qualified for 15-minute logged groundwater
GROUNDWATER_HYDROLOGY_MEASURES = {
    'SP50_72': 'ca3d0164-ee1d-444e-bc6c-04d600498b61-gw-logged-i-subdaily-mAOD-qualified',
}

BASE_URL = "https://environment.data.gov.uk/hydrology/id/measures"


def download_station_groundwater(
    station_name: str,
    measure_id: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> bool:
    """
    Download qualified groundwater data for a single station as CSV.
    
    Args:
        station_name: Name of the groundwater station
        measure_id: Hydrology API measure ID (full notation)
        min_date: Minimum date in YYYY-MM-DD format (filters at API level)
        max_date: Maximum date in YYYY-MM-DD format (optional)
        output_dir: Directory to save CSV file (defaults to data/groundwater_training_data/)
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
        output_dir = project_root / 'data' / 'groundwater_training_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build CSV API URL for groundwater data
    # Groundwater endpoint pattern: {measure_id}/readings.csv
    base_url = f"{BASE_URL}/{measure_id}/readings.csv"
    
    if verbose:
        print(f"Downloading {station_name} groundwater...")
        if min_date:
            print(f"  Filtering from {min_date} onwards (using API min-date parameter)")
        if max_date:
            print(f"  Filtering until {max_date} (using API max-date parameter)")
    
    try:
        output_file = output_dir / f"{station_name}-groundwater-15min-Qualified.csv"
        
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


def download_all_groundwater(
    stations: Optional[Dict[str, str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    delay: float = 2.0,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Download qualified groundwater data for all stations.
    
    Args:
        stations: Dictionary mapping station names to measure IDs
                 (defaults to GROUNDWATER_HYDROLOGY_MEASURES)
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
        stations = GROUNDWATER_HYDROLOGY_MEASURES
    
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    if output_dir is None:
        output_dir = project_root / 'data' / 'groundwater_training_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Downloading historical qualified groundwater data")
        print(f"{'='*70}")
        if min_date:
            print(f"Minimum date: {min_date}")
        if max_date:
            print(f"Maximum date: {max_date}")
        print(f"Output directory: {output_dir}")
        print(f"Stations: {len(stations)}")
        print()
    
    results = {}
    for name, measure_id in stations.items():
        if measure_id is None:
            if verbose:
                print(f"⚠ Skipping {name}: measure ID not set")
            results[name] = False
            continue
            
        success = download_station_groundwater(
            station_name=name,
            measure_id=measure_id,
            min_date=min_date,
            max_date=max_date,
            output_dir=output_dir,
            project_root=project_root,
            verbose=verbose
        )
        results[name] = success
        time.sleep(delay)
    
    if verbose:
        print(f"\n{'='*70}")
        print("Download Summary:")
        print(f"{'='*70}")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download historical qualified groundwater data from Environment Agency Hydrology API')
    parser.add_argument('--min-date', type=str, help='Minimum date in YYYY-MM-DD format (optional)')
    parser.add_argument('--max-date', type=str, help='Maximum date in YYYY-MM-DD format (optional)')
    parser.add_argument('--output-dir', type=str, help='Output directory for CSV files (defaults to data/groundwater_training_data/)')
    parser.add_argument('--station', type=str, help='Download data for a single station only')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between API requests in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    if args.station:
        if args.station not in GROUNDWATER_HYDROLOGY_MEASURES:
            print(f"Error: Unknown station '{args.station}'. Available: {', '.join(GROUNDWATER_HYDROLOGY_MEASURES.keys())}")
            exit(1)
        stations_to_download = {args.station: GROUNDWATER_HYDROLOGY_MEASURES[args.station]}
    else:
        stations_to_download = GROUNDWATER_HYDROLOGY_MEASURES
    
    download_all_groundwater(
        stations=stations_to_download,
        min_date=args.min_date,
        max_date=args.max_date,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        delay=args.delay,
        verbose=True
    )
