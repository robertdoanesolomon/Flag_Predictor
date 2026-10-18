"""Print the date (YYYY-MM-DD) of the newest reading across the EA rainfall CSVs in a folder.

Used by CI to decide whether the cached long rainfall record is stale.
Prints 1970-01-01 if there is nothing readable.
"""

import glob
import sys

import pandas as pd


def newest(folder: str) -> str:
    stamps = []
    for path in glob.glob(f'{folder}/*.csv'):
        try:
            last = pd.read_csv(path, usecols=['dateTime'])['dateTime'].dropna()
            if len(last):
                stamps.append(pd.to_datetime(last).max())
        except Exception:
            continue
    return max(stamps).strftime('%Y-%m-%d') if stamps else '1970-01-01'


if __name__ == '__main__':
    print(newest(sys.argv[1] if len(sys.argv) > 1 else 'data/rainfall_training_data'))
