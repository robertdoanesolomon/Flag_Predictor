"""
Train the June 2026 flag predictor models.

Improvements over the May 2026 / experiment_2026_01 models:
1. Delta targets: the model predicts the CHANGE in differential from the
   current value, so forecasts are anchored at the observed level and the
   "weird jump" at the start of predictions disappears.
2. Physical recession constraint: a soft loss penalty plus a hard clamp at
   inference stop the model predicting the river dropping faster than
   ~3 inches/day (Anu's rule).
3. Winter upweighting: samples from Nov-Mar (when structural error is worst)
   count 1.5x in the loss.
4. Future-rain noise augmentation: future-rainfall features are perturbed
   during training so the model does not over-trust the (perfect, historical)
   rainfall it sees in training but never gets at forecast time.
5. Fixed a 1-hour misalignment between input sequences and targets.

Saves models as:
    models/multihorizon_model_experiment_2026_06_{location}.pth (+ scaler/config)
    models/multihorizon_model_{location}_latest.pth (+ scaler/config)

Usage:
    python train_june_models.py [isis godstow wallingford]
"""

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flag_predictor.pipeline import train_location_model  # noqa: E402


def main():
    locations = sys.argv[1:] or ['isis', 'godstow', 'wallingford']
    results = {}

    for location in locations:
        print(f"\n{'=' * 80}")
        print(f"TRAINING JUNE 2026 MODEL: {location.upper()}")
        print(f"{'=' * 80}", flush=True)
        start = time.time()
        try:
            model, model_config = train_location_model(
                location=location,
                project_root=PROJECT_ROOT,
                save_dir=str(PROJECT_ROOT / 'models'),
                verbose=True,
            )
            elapsed = time.time() - start
            history = model_config['training_history']
            results[location] = {
                'status': 'ok',
                'minutes': elapsed / 60,
                'best_val_mae': min(history['val_mae']),
                'epochs': len(history['val_mae']),
            }
            print(f"\n[DONE] {location}: {elapsed/60:.1f} min, "
                  f"best val MAE {min(history['val_mae']):.4f}", flush=True)
        except Exception:
            results[location] = {'status': 'failed'}
            print(f"\n[FAILED] {location}:", flush=True)
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")
    for loc, res in results.items():
        print(f"  {loc}: {res}")
    if any(r['status'] != 'ok' for r in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
