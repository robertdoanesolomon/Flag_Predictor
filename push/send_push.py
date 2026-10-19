#!/usr/bin/env python
"""
Send flag alerts after a forecast run (called from .github/workflows/update_figures.yml).

    python push/send_push.py                 # decide (push/alerts.py) and send
    python push/send_push.py --dry-run       # print what would be sent, change nothing
    python push/send_push.py --test [isis|godstow]   # send a test notification to everyone (or one stretch)

Reads docs/preview/forecast_data.json; sign-ups, the VAPID private key and the alert state come
from the Cloudflare KV namespace (CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID). Without those it
does nothing. Dead subscriptions (the push service answers 404 / 410) are removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alerts import NAMES, RIVERS, SITE_URL, decide  # noqa: E402
from cloudflare import Cloudflare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'docs' / 'preview' / 'forecast_data.json'
STATE_KEY = 'state:alerts'
TTL_S = 6 * 3600          # a phone that is off gets the alert when it comes back, up to 6 h later


def load_subscriptions(cf: Cloudflare) -> list:
    subs = []
    for key in cf.keys('sub:'):
        raw = cf.get(key)
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        rec['_key'] = key
        subs.append(rec)
    return subs


def send(cf: Cloudflare, msgs: list, subs: list) -> dict:
    from py_vapid import Vapid
    from pywebpush import WebPushException, webpush
    vapid = Vapid.from_pem(cf.get('config:vapid_private').encode())
    claims = {'sub': 'https://flagpredictor.com'}
    sent = failed = removed = 0
    gone = set()
    for m in msgs:
        payload = json.dumps({k: m[k] for k in ('title', 'body', 'url', 'tag')})
        for s in subs:
            if s['_key'] in gone or m['river'] not in s.get('rivers', []):
                continue
            try:
                webpush(s['subscription'], data=payload, vapid_private_key=vapid, vapid_claims=dict(claims),
                        ttl=TTL_S, headers={'Urgency': 'high'}, timeout=20)
                sent += 1
            except WebPushException as exc:
                code = getattr(exc.response, 'status_code', None)
                if code in (404, 410):
                    cf.delete(s['_key'])
                    gone.add(s['_key'])
                    removed += 1
                else:
                    failed += 1
                    print(f'  push failed ({code}): {str(exc)[:160]}')
            except Exception as exc:  # network trouble: try the others
                failed += 1
                print(f'  push failed: {type(exc).__name__}: {str(exc)[:160]}')
    return {'sent': sent, 'failed': failed, 'removed': removed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--test', nargs='?', const='all', choices=['all', *RIVERS])
    args = ap.parse_args()
    cf = Cloudflare()
    if not cf.configured:
        print('[alerts] Cloudflare not configured; no alerts sent')
        return
    if not cf.find_namespace():
        print('[alerts] no alerts namespace yet (run the "Deploy flag alerts" workflow); no alerts sent')
        return
    subs = load_subscriptions(cf)
    by_river = {r: sum(r in s.get('rivers', []) for s in subs) for r in RIVERS}
    print(f'[alerts] {len(subs)} sign-ups {by_river}')

    if args.test:
        rivers = RIVERS if args.test == 'all' else (args.test,)
        msgs = [{'river': r, 'title': f'{NAMES[r]} Flag', 'tag': f'test-{r}', 'url': SITE_URL,
                 'body': f'Test alert: {NAMES[r]} Flag alerts are working.'} for r in rivers]
        print('[alerts] sending test:', json.dumps(msgs))
        print('[alerts]', send(cf, msgs, subs))
        return

    if not BUNDLE.exists():
        print('[alerts] no preview bundle; nothing to do')
        return
    bundle = json.loads(BUNDLE.read_text())
    state = json.loads(cf.get(STATE_KEY) or '{}')
    msgs, new_state = decide(bundle, state)
    for m in msgs:
        print(f"[alerts] {m['title']}: {m['body']}")
    if args.dry_run:
        print('[alerts] dry run: new state', json.dumps(new_state))
        return
    if msgs and subs:
        print('[alerts]', send(cf, msgs, subs))
    elif not msgs:
        print('[alerts] nothing to send')
    cf.put(STATE_KEY, json.dumps(new_state))


if __name__ == '__main__':
    main()
