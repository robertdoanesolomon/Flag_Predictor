"""
The official OURCs flags (https://ourcs.co.uk/information/flags/).

* ``fetch_status(river)``: the flag flying now, from the OURCs status API
  (``/api/flags/status/<river>/``: status letter, text, notices, set time (UTC)).
* ``fetch_history(river)``: every flag since December 2018 from the river's
  flag page (status, start, end in UK time), as UTC intervals.

The flags are set by people, not by the gauges: on the Isis by counting the
"bucks" at Iffley weir (Code of Conduct 2.8.g), at Godstow from the lock
differential (2.7.c), so the differential bands on this site are only an
approximation of the flag.
"""

from __future__ import annotations

import html
import re
from typing import Optional

import pandas as pd

RIVERS = ('isis', 'godstow')
STATUS_URL = 'https://ourcs.co.uk/api/flags/status/{river}/'
HISTORY_URL = 'https://ourcs.co.uk/information/flags/{river}/'
USER_AGENT = 'FlagPredictor (github.com/robertdoanesolomon/Flag_Predictor)'
LOCAL_TZ = 'Europe/London'

# Flag names on the OURCs pages -> the keys used on this site. Before late 2021
# the Isis had a single "Blue" flag between Green and Amber.
FLAG_KEYS = {
    'green': 'green', 'light blue': 'light_blue', 'dark blue': 'dark_blue', 'blue': 'blue',
    'amber': 'amber', 'red': 'red', 'black': 'black', 'grey': 'grey', 'gray': 'grey',
}
STATUS_LETTERS = {'G': 'green', 'LB': 'light_blue', 'DB': 'dark_blue', 'B': 'blue', 'A': 'amber',
                  'R': 'red', 'K': 'black', 'BL': 'black', 'X': 'grey', 'GR': 'grey', 'GY': 'grey'}
# Order of severity (grey = not monitored, no position).
SEVERITY = ['green', 'light_blue', 'blue', 'dark_blue', 'amber', 'red', 'black']


def flag_key(text: str) -> str:
    return FLAG_KEYS.get(str(text).strip().lower(), str(text).strip().lower().replace(' ', '_'))


def _get(url: str, timeout: int = 30):
    import requests
    r = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r


def fetch_status(river: str, timeout: int = 30) -> Optional[dict]:
    """Current official flag: {'river', 'flag', 'label', 'notices', 'set_at' (UTC ISO)} or None.

    Who set the flag is deliberately not kept: no OURCs officer's name goes on the website."""
    try:
        j = _get(STATUS_URL.format(river=river), timeout).json()
    except Exception as exc:
        print(f"  [ourcs] status for {river} unavailable ({exc})")
        return None
    label = j.get('status_text') or ''
    key = flag_key(label) if label else STATUS_LETTERS.get(str(j.get('status', '')).upper(), 'unknown')
    set_at = pd.Timestamp(j['set_date']).tz_convert('UTC') if j.get('set_date') else None
    return {'river': river, 'flag': key, 'label': label or key.replace('_', ' ').title(),
            'notices': list(j.get('notices') or []),
            'set_at': set_at.isoformat() if set_at is not None else None}


_ROW = re.compile(r'<tr>\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>([^<]*)</td>\s*</tr>', re.S)


def parse_history_html(page: str) -> pd.DataFrame:
    """The 'History' table of a river's flag page -> DataFrame(flag, label, start, end), UTC, oldest first."""
    i = page.find('>History<')
    body = page[i:] if i >= 0 else page
    rows = []
    for status, start, end in _ROW.findall(body):
        status, start, end = (html.unescape(x).strip() for x in (status, start, end))
        if not re.match(r'\d\d/\d\d/\d\d \d\d:\d\d', start):
            continue
        rows.append((status, start, end))
    df = pd.DataFrame(rows, columns=['label', 'start', 'end'])

    def to_utc(s):
        t = pd.to_datetime(s, format='%d/%m/%y %H:%M', errors='coerce')
        return t.dt.tz_localize(LOCAL_TZ, ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert('UTC')

    df['start'] = to_utc(df['start'])
    df['end'] = to_utc(df['end'].replace('', None))
    df['flag'] = df['label'].map(flag_key)
    df = df.dropna(subset=['start']).sort_values('start').reset_index(drop=True)
    return df[['flag', 'label', 'start', 'end']]


def fetch_history(river: str, timeout: int = 30) -> Optional[pd.DataFrame]:
    try:
        return parse_history_html(_get(HISTORY_URL.format(river=river), timeout).text)
    except Exception as exc:
        print(f"  [ourcs] history for {river} unavailable ({exc})")
        return None


def flag_series(history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Official flag key in force at each time of `index` (UTC), from a history table."""
    h = history.sort_values('start')
    pos = h['start'].searchsorted(index, side='right') - 1
    out = pd.Series(pd.NA, index=index, dtype='object')
    ok = pos >= 0
    out[ok] = h['flag'].to_numpy()[pos[ok]]
    return out
