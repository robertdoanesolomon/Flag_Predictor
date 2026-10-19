"""
Which flag alerts to send after a forecast run (pure logic, no network).

Per stretch (Isis, Godstow), from the preview bundle (docs/preview/forecast_data.json):

* (the official flag has changed: sent within a minute by the Cloudflare Worker, push/worker)
* the official flag is likely to change: chance >= LIKELY today, or by the end of tomorrow
                                                               -> "Flag likely down to Amber by tomorrow (70%) ..."
* Isis only: Black likely by the end of tomorrow (>= LIKELY), or, when Black, likely back to Red.

Each "likely" alert is sent once per flag; it can be sent again only after its chance has
dropped below RESET (so a forecast hovering around the threshold does not alert every hour), or
once the flag has changed. Nothing is sent from a stale forecast (gauges offline) or an old bundle.
The state between runs is a small JSON object (kept in Cloudflare KV by send_push.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

RIVERS = ('isis', 'godstow')
NAMES = {'isis': 'Isis', 'godstow': 'Godstow'}
LIKELY = 0.6
RESET = 0.3
MAX_BUNDLE_AGE_H = 3.0
SITE_URL = 'https://flagpredictor.com/preview/'
UK = ZoneInfo('Europe/London')
EMOJI = {'green': '🟢', 'light_blue': '🩵', 'blue': '🔵', 'dark_blue': '🔵', 'amber': '🟠', 'red': '🔴',
         'black': '⚫', 'grey': '⚪'}
LABELS = {'green': 'Green', 'light_blue': 'Light Blue', 'dark_blue': 'Dark Blue', 'amber': 'Amber',
          'red': 'Red', 'black': 'Black', 'grey': 'Grey'}


def _label(key: Optional[str]) -> str:
    return LABELS.get(key or '', (key or '?').replace('_', ' ').title())


def _pct(p: float) -> str:
    if p >= 0.95:
        return 'over 95%'
    return f'{int(round(p * 20)) * 5}%'


def _ts(iso: str) -> datetime:
    t = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _when(iso: Optional[str], now: datetime) -> str:
    """' Probably mid-morning tomorrow.' from a UTC time, in UK time."""
    if not iso:
        return ''
    t = _ts(iso).astimezone(UK)
    h = t.hour
    part = ('early morning' if 6 <= h < 9 else 'mid-morning' if 9 <= h < 12 else 'early afternoon' if 12 <= h < 15
            else 'late afternoon' if 15 <= h < 18 else 'in the evening' if 18 <= h < 22 else 'overnight')
    day = 'today' if t.date() == now.astimezone(UK).date() else 'tomorrow'
    return f' Probably {part} {day}.'


def _msg(river: str, body: str, flag: Optional[str] = None) -> dict:
    """An alert; ``flag`` (the flag it is about) puts its colour emoji in front."""
    if flag in EMOJI:
        body = f'{EMOJI[flag]} {body}'
    return {'river': river, 'title': f'{NAMES[river]} Flag', 'body': body, 'url': SITE_URL, 'tag': f'flag-{river}'}


def decide(bundle: dict, state: dict, now: Optional[datetime] = None) -> Tuple[List[dict], dict]:
    """(messages, new state). state: {river: {'flag': key, 'sent': [alert keys]}}."""
    now = now or datetime.now(timezone.utc)
    new_state = {r: dict(v) for r, v in (state or {}).items()}
    msgs: List[dict] = []
    gen = bundle.get('generated_at')
    if not gen or (now - _ts(gen)).total_seconds() / 3600 > MAX_BUNDLE_AGE_H:
        return msgs, new_state
    for river in RIVERS:
        loc = (bundle.get('locations') or {}).get(river)
        off = (loc or {}).get('official_flag')
        if not off or not off.get('flag'):
            continue
        flag = off['flag']
        prev = (state or {}).get(river) or {}
        sent = set(prev.get('sent', []))
        if prev.get('flag') and prev['flag'] != flag:
            sent = set()             # the change itself is alerted by the Worker; start afresh
        if loc.get('stale') or off.get('forecast_stale'):
            new_state[river] = {'flag': flag, 'sent': sorted(sent)}
            continue

        oo = loc.get('official_outlook') or {}
        win = oo.get('windows') or {}
        for d in ('up', 'down'):
            info = oo.get(d)
            if not info:
                continue
            p_today = (win.get('today') or {}).get(d, 0.0) or 0.0
            p_tom = (win.get('tomorrow') or {}).get(d, 0.0) or 0.0
            target = info.get('flag')
            k_today, k_tom = f'{d}:{target}:today', f'{d}:{target}:tomorrow'
            if p_tom < RESET:
                sent -= {k for k in sent if k.startswith(f'{d}:')}
                continue
            when = _when((info.get('when') or {}).get('median_time'), now)
            if p_today >= LIKELY and k_today not in sent:
                msgs.append(_msg(river, f'Official Flag likely {d} to {_label(target)} today ({_pct(p_today)}).{when}', target))
                sent |= {k_today, k_tom}
            elif p_tom >= LIKELY and k_tom not in sent:
                msgs.append(_msg(river, f'Official Flag likely {d} to {_label(target)} by the end of tomorrow '
                                        f'({_pct(p_tom)}).{when}', target))
                sent.add(k_tom)

        bk = loc.get('black') if river == 'isis' else None
        if bk and bk.get('windows'):
            p_tom = bk['windows'].get('tomorrow', 0.0) or 0.0
            is_black = bk.get('black_now') or flag == 'black'
            key = 'black:to_red' if is_black else 'black:to_black'
            other = 'black:to_black' if is_black else 'black:to_red'
            sent.discard(other)
            if p_tom < RESET:
                sent.discard(key)
            elif p_tom >= LIKELY and key not in sent:
                when = _when((bk.get('when') or {}).get('median_time'), now)
                text = (f'Black Flag likely back to Red by the end of tomorrow ({_pct(p_tom)}).{when}' if is_black
                        else f'Black Flag likely by the end of tomorrow ({_pct(p_tom)}): Iffley Lock is at '
                             f"{bk.get('level_m', 0):.2f} m.{when}")
                msgs.append(_msg(river, text, 'red' if is_black else 'black'))
                sent.add(key)
        new_state[river] = {'flag': flag, 'sent': sorted(sent)}
    return msgs, new_state
