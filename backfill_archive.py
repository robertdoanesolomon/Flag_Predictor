#!/usr/bin/env python
"""
Backfill docs/archive from the git history of docs/forecast_data.json.

Every CI run since the hybrid went live (2026-09-27) committed the forecast it
published, so the history already holds the operational forecasts. For each
6-hour slot this takes the first published forecast and the observations that
came with it.

    python backfill_archive.py --since 2026-09-27T15:20:00Z

Needs the GitHub CLI (gh) authenticated for the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.prediction.archive import (  # noqa: E402
    build_now, merge_observed, rebuild_index, slot_id, write_issue,
)

REPO = 'robertdoanesolomon/flag_predictor'


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def list_commits(since: str, until: str | None) -> list[tuple[str, str]]:
    q = f'repos/{REPO}/commits?path=docs/forecast_data.json&sha=main&since={since}&per_page=100'
    if until:
        q += f'&until={until}'
    rows = []
    for page in range(1, 200):   # explicit pages (the API's own next-links are not always usable)
        out = gh('api', f'{q}&page={page}', '--jq', '.[] | [.sha, .commit.committer.date] | @tsv')
        lines = [line.split('\t') for line in out.strip().splitlines() if line.strip()]
        rows.extend(lines)
        if len(lines) < 100:
            break
    return sorted(((d, s) for s, d in rows))


def fetch_bundle(sha: str) -> dict:
    raw = gh('api', '-H', 'Accept: application/vnd.github.raw',
             f'repos/{REPO}/contents/docs/forecast_data.json?ref={sha}')
    return json.loads(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-27T15:20:00Z')
    ap.add_argument('--until', default=None)
    ap.add_argument('--docs', default='docs')
    ap.add_argument('--model', default='hybrid_c1_ps (rain spread only)')
    ap.add_argument('--location-model', default='hybrid_c1_ps',
                    help='model name recorded per location (the main page showed the hybrid from 2026-09-27)')
    args = ap.parse_args()
    archive_dir = ROOT / args.docs / 'archive'

    commits = list_commits(args.since, args.until)
    print(f'{len(commits)} commits touching forecast_data.json')
    by_slot: dict[str, tuple[str, str]] = {}
    for date, sha in commits:
        by_slot.setdefault(slot_id(pd.Timestamp(date)), (date, sha))
    print(f'{len(by_slot)} six-hour slots')
    for slot, (date, sha) in sorted(by_slot.items()):
        try:
            bundle = fetch_bundle(sha)
        except subprocess.CalledProcessError as exc:
            print(f'  {slot}: fetch failed ({exc.stderr.strip()[:80]})')
            continue
        payloads = bundle.get('locations', {})
        for p in payloads.values():
            p.setdefault('model', args.location_model)
        issued = pd.Timestamp(bundle.get('generated_at', date))
        if slot_id(issued) != slot:          # commit landed in a later slot than the run
            slot_used = slot_id(issued)
        else:
            slot_used = slot
        merge_observed(archive_dir, payloads)
        wrote = write_issue(archive_dir, slot_used, payloads, args.model, issued)
        print(f'  {slot_used}: {sha[:8]} generated {issued} {"written" if wrote else "exists"}')
    entries = rebuild_index(archive_dir)
    from flag_predictor.prediction.archive import _load_json, observed_series
    stored = _load_json(archive_dir / 'observed.json', {})
    build_now(archive_dir, {loc: observed_series(v) for loc, v in stored.items()})
    print(f'index: {len(entries)} issues')


if __name__ == '__main__':
    main()
