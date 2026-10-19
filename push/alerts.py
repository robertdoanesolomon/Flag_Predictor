"""
Which flag alerts to send after a forecast run (pure logic, no network).

Per stretch (Isis, Godstow), from the preview bundle (docs/preview/forecast_data.json):

* (the official flag has changed: sent within a minute by the Cloudflare Worker, push/worker)
* the official flag is likely to change: chance >= LIKELY today, or by the end of tomorrow
                                                               -> "Flag likely down to Amber by tomorrow (70%) ..."
* escalation: after an "up" alert, a higher flag becomes >= LIKELY (going up can skip a flag)
                                                               -> "Official Flag now likely up to Red by tomorrow (70%)"
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
SEVERITY = ['green', 'light_blue', 'blue', 'dark_blue', 'amber', 'red', 'black']
CLEAR = 0.75     # name the destination only when this share of the outcomes that change agree on it
MAX_BUNDLE_AGE_H = 3.0
SITE_URL = 'https://flagpredictor.com/preview/'
UK = ZoneInfo('Europe/London')
EMOJI = {'green': '🟢', 'light_blue': '🔷', 'blue': '🔵', 'dark_blue': '🔵', 'amber': '🟠', 'red': '🔴',
         'black': '⚫', 'grey': '⚪'}
LABELS = {'green': 'Green', 'light_blue': 'Light Blue', 'dark_blue': 'Dark Blue', 'amber': 'Amber',
          'red': 'Red', 'black': 'Black', 'grey': 'Grey'}


def _label(key: Optional[str]) -> str:
    return LABELS.get(key or '', (key or '?').replace('_', ' ').title())


def _word(p: float) -> str:
    """Same words as the website, set a little cautiously: very likely 90%+, likely 70%+, else possibly."""
    r = round(p * 20) / 20
    return 'very likely' if r >= 0.9 else 'likely' if r >= 0.7 else 'possibly'


def _if_so(when: str, p: float) -> str:
    """' Probably mid-morning.' -> ' If so, probably mid-morning.' when the change is only possible."""
    return when.replace(' Probably', ' If so, probably') if when and _word(p) == 'possibly' else when


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


ARROW = {'up': '⬆️', 'down': '⬇️'}


def _msg(river: str, body: str, flag: Optional[str] = None, direction: Optional[str] = None) -> dict:
    """An alert; an arrow for the direction and the colour emoji of the flag it is about go in front."""
    if flag in EMOJI:
        body = f'{EMOJI[flag]} {body}'
    if direction in ARROW:
        body = f'{ARROW[direction]} {body}'
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
        up_top = prev.get('up_top')          # highest flag an "up" alert has said is likely
        if prev.get('flag') and prev['flag'] != flag:
            sent, up_top = set(), None       # the change itself is alerted by the Worker; start afresh
        if loc.get('stale') or off.get('forecast_stale'):
            new_state[river] = {'flag': flag, 'sent': sorted(sent), 'up_top': up_top}
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
            split = {k: v for k, v in (info.get('split') or {}).items() if v > 0}
            total = sum(split.values())
            clear = not split or max(split.values()) / total >= CLEAR
            k_today, k_tom = f'{d}:today', f'{d}:tomorrow'
            if p_tom < RESET:
                sent -= {k for k in sent if k.startswith(f'{d}:')}
                if d == 'up':
                    up_top = None
                continue
            when = _when((info.get('when') or {}).get('median_time'), now)
            # chance of getting at least as far as each flag (going up can skip a flag)
            reach = {k: sum(v for k2, v in split.items() if SEVERITY.index(k2) >= SEVERITY.index(k))
                     for k in split if k in SEVERITY}
            likely_top = max((k for k, v in reach.items() if v >= LIKELY), key=SEVERITY.index, default=None)
            if clear:
                where, emoji = f' to {_label(target)}', target
                detail = ''
            else:                     # e.g. up from Dark Blue: Amber or straight to Red
                where, emoji = '', None
                parts = sorted(split.items(), key=lambda kv: -kv[1])
                detail = ': ' + ', '.join(f'{_label(k)} {_pct(v)}' for k, v in parts if v >= 0.05)
            if p_today >= LIKELY and k_today not in sent:
                verb = f'{_word(p_today)} {d}' if clear else f'{_word(p_today)} to go {d}'
                msgs.append(_msg(river, f'Official Flag {verb}{where} today ({_pct(p_today)}){detail}.{_if_so(when, p_today)}', emoji, d))
                sent |= {k_today, k_tom}
                if d == 'up':
                    up_top = likely_top or target
            elif p_tom >= LIKELY and k_tom not in sent:
                verb = f'{_word(p_tom)} {d}' if clear else f'{_word(p_tom)} to go {d}'
                msgs.append(_msg(river, f'Official Flag {verb}{where} by the end of tomorrow ({_pct(p_tom)}){detail}.{_if_so(when, p_tom)}',
                                 emoji, d))
                sent.add(k_tom)
                if d == 'up':
                    up_top = likely_top or target
            elif (d == 'up' and up_top in SEVERITY and likely_top
                  and SEVERITY.index(likely_top) > SEVERITY.index(up_top) and k_tom in sent):
                # escalation: after an "up" alert, a higher flag has become 60%+ likely
                pr = reach[likely_top]
                msgs.append(_msg(river, f'Official Flag now {_word(pr)} up to {_label(likely_top)} by the end of tomorrow '
                                        f'({_pct(pr)}).{_if_so(when, pr)}', likely_top, 'up'))
                up_top = likely_top

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
                text = (f'Black Flag {_word(p_tom)} back to Red by the end of tomorrow ({_pct(p_tom)}).{_if_so(when, p_tom)}' if is_black
                        else f'Black Flag {_word(p_tom)} by the end of tomorrow ({_pct(p_tom)}): Iffley Lock is at '
                             f"{bk.get('level_m', 0):.2f} m.{_if_so(when, p_tom)}")
                msgs.append(_msg(river, text, 'red' if is_black else 'black', 'down' if is_black else 'up'))
                sent.add(key)
        new_state[river] = {'flag': flag, 'sent': sorted(sent), 'up_top': up_top}
    return msgs, new_state
