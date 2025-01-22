"""
Clear old cache and regenerate forecasts with new format.
"""

import shutil
from pathlib import Path
from forecast_scheduler import update_all_forecasts

CACHE_DIR = Path(__file__).parent / 'forecast_cache'

if __name__ == '__main__':
    print("="*80)
    print("Clearing old cache and regenerating forecasts...")
    print("="*80)
    
    # Clear cache
    if CACHE_DIR.exists():
        print(f"\nDeleting old cache in {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
        print("✓ Cache cleared")
    else:
        print("\nNo cache directory found (this is OK)")
    
    # Regenerate
    print("\nGenerating new forecasts with updated format...")
    print("This will take a few minutes...\n")
    
    update_all_forecasts(verbose=True)
    
    print("\n" + "="*80)
    print("✓ Cache regenerated successfully!")
    print("="*80)
    print("\nYou can now visit the website and it should work.")
