"""
Train the September 2026 flag predictor.

Stage 1: rain → Farmoor flow (hourly decoder, long record).
Kill switch: if that model does not beat flow persistence on 24–240h MAE
over the frozen backtest window, location models train without predicted flow.

Stage 2: per-location hourly differential decoder.

Usage:
    python train_september_models.py [farmoor isis godstow wallingford]
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.pipeline import (  # noqa: E402
    train_september_flow_model,
    train_september_location_model,
)


def main():
    requested = sys.argv[1:] or ['farmoor', 'isis', 'godstow', 'wallingford']
    save_dir = str(PROJECT_ROOT / 'models')
    results = {}
    uses_predicted_flow = True
    flag_path = PROJECT_ROOT / 'models' / 'september_stage1.json'

    if 'farmoor' in requested:
        print(f"\n{'=' * 80}\nTRAINING SEPTEMBER 2026 FLOW MODEL (Farmoor)\n{'=' * 80}", flush=True)
        start = time.time()
        try:
            _, cfg = train_september_flow_model(
                project_root=PROJECT_ROOT, save_dir=save_dir, verbose=True
            )
            results['farmoor'] = {
                'status': 'ok',
                'minutes': (time.time() - start) / 60,
                'best_val_mae': min(cfg['training_history']['val_mae']),
                'epochs': len(cfg['training_history']['val_mae']),
            }
        except Exception:
            results['farmoor'] = {'status': 'failed'}
            traceback.print_exc()
            sys.exit(1)

        # Kill-switch backtest (imported lazily so training can run first)
        from backtest_flow_vs_persistence import run_flow_persistence_backtest  # noqa: E402
        beat, summary = run_flow_persistence_backtest(PROJECT_ROOT)
        uses_predicted_flow = bool(beat)
        results['farmoor']['beats_persistence'] = beat
        results['farmoor']['flow_backtest'] = summary
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text(json.dumps({
            'uses_predicted_flow': uses_predicted_flow,
            'summary': summary,
        }, indent=2, default=str))
        print(
            f"\nKill switch: uses_predicted_flow={uses_predicted_flow} "
            f"(stage-1 24–240h MAE vs persistence)",
            flush=True,
        )
        (PROJECT_ROOT / 'training_log_september.txt').write_text(
            'September 2026 training log\n'
            + '=' * 40 + '\n'
            + f"farmoor: {results['farmoor']}\n"
        )
    elif flag_path.exists():
        uses_predicted_flow = json.loads(flag_path.read_text()).get(
            'uses_predicted_flow', True
        )

    for location in [x for x in requested if x != 'farmoor']:
        print(f"\n{'=' * 80}")
        print(f"TRAINING SEPTEMBER 2026 MODEL: {location.upper()} "
              f"(predicted flow={uses_predicted_flow})")
        print(f"{'=' * 80}", flush=True)
        start = time.time()
        try:
            _, cfg = train_september_location_model(
                location=location,
                project_root=PROJECT_ROOT,
                save_dir=save_dir,
                uses_predicted_flow=uses_predicted_flow,
                verbose=True,
            )
            results[location] = {
                'status': 'ok',
                'minutes': (time.time() - start) / 60,
                'best_val_mae': min(cfg['training_history']['val_mae']),
                'epochs': len(cfg['training_history']['val_mae']),
                'uses_predicted_flow': uses_predicted_flow,
            }
        except Exception:
            results[location] = {'status': 'failed'}
            traceback.print_exc()

    print(f"\n{'=' * 80}\nTRAINING SUMMARY\n{'=' * 80}")
    for loc, res in results.items():
        print(f"  {loc}: {res}")
    log_path = PROJECT_ROOT / 'training_log_september.txt'
    lines = ['September 2026 training log', '=' * 40]
    for loc, res in results.items():
        mae = res.get('best_val_mae', 'n/a')
        lines.append(
            f"{loc}: status={res.get('status')}  best_val_mae={mae}  "
            f"epochs={res.get('epochs', 'n/a')}  minutes={res.get('minutes', 'n/a')}"
        )
        if loc == 'farmoor':
            lines.append(f"  beats_persistence={res.get('beats_persistence')}")
            lines.append(f"  flow_backtest={res.get('flow_backtest')}")
        if 'uses_predicted_flow' in res:
            lines.append(f"  uses_predicted_flow={res['uses_predicted_flow']}")
    log_path.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {log_path}")
    if any(r.get('status') != 'ok' for r in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
