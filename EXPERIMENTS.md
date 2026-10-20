# Physics-redesign experiments

Branch `physics-redesign`. Goal: 10-day differential forecasts that are as
accurate as possible (MAE by horizon) and physically realistic by
construction, not by clamps applied afterwards. The live September 2026 model
is the baseline; nothing here touches `main`, `docs/`, `.github/` or the live
model files.

## Summary

**Recommendation: replace the September model with the physics hybrid
(`hybrid_c1_ps`, five seeds averaged).** Not deployed; see Deployment notes.

On the frozen test window (weekly starts, Nov 2024 – Jan 2026, observed
rain), against the live September model:

- **MAE falls by 36–44%:** 0.048 → 0.031 m (Isis), 0.066 → 0.040 m (Godstow),
  0.107 → 0.060 m (Wallingford).
- **Winter MAE falls by 39–46%.**
- **Right flag colour:** 83% → 89% of forecast hours at Isis, 87% → 92% at
  Godstow. Two or more colours wrong: 4.9% → 2.4% at Isis.
- **The two problems asked about are gone by construction:**
  - Dry-weather climbs of > 1 cm: 16–44% of forecasts → 0–2%. The 2% is one
    Godstow forecast draining rain from earlier in the week.
  - Kinks: the sharpest falls from 13–14 to 0.6–1.3 mm/h², and direction
    reversals from 5–7 to under 1 per forecast.
- **More rain never lowers the forecast** (live: 5–7% of hours at Isis and
  Godstow).
- **Recessions now fall as fast as the real river does,** instead of being
  clamped to 3 in/day.

Where it doesn't win: the first 24 h at Wallingford (0.0267 vs 0.0256 m).

The fixed LSTM (candidate A) is more accurate than the live model and has the
best first-day MAE. But without structural constraints it still climbs in dry
weather and wiggles, and every attempt to train realism into it with
penalties collapsed it to a flat forecast.

## How to reproduce

```bash
python evaluate_candidates.py all --window val  --candidates persistence,sept_full,sept_raw
python evaluate_candidates.py all --window test --candidates sept_full,redesign:<name>
python train_physics_models.py hybrid all --name <name> [--cfg '{"epochs": 25}']
```

Per-start rows and trajectories land in `figures/eval/`; the summary table
for each window is `figures/eval/summary_{window}.csv`.

## Evaluation (step 1)

`src/flag_predictor/evaluation.py`, driven by `evaluate_candidates.py`.

- **Windows.** `test`: weekly starts 2024-11-01 → 2026-01-05 (the frozen May/June/Sept
  window, 57 Isis / 58 Godstow / 58 Wallingford starts with >200 observed hours).
  `val`: weekly starts through 2023 (validation year). Model choices are
  made on `val`; `test` is reported at the end.
- **Accuracy.** MAE over hours 1–240, 1–24, 25–72, 73–240, bias, winter-only MAE.
- **Production-like inputs.** Rain is the observed rain ("clairvoyant"), as in all
  earlier backtests, but stage 2 of the September model gets stage-1
  *predicted* Farmoor flow, never observed future flow.
- **Realism, on each candidate's raw output:**
  - `dry_climb_cm`: total climb (cm) during forecast hours with < 2 mm
    mean-station rain in the preceding 72 h (observed past + forecast).
    `dry_climb_any` is the share of forecasts climbing > 1 cm while dry.
    The 72 h / 2 mm rule comes from the data: the rain → dD/dt cross-correlation
    peaks at 6 h and stays above a quarter of its peak out to 48–60 h at all
    three locations.
  - `roughness_mm` (mean |second difference|, mm/h²), `max_kink_mm`,
    `reversals` (direction changes per forecast).
  - `recession_viol_frac`: share of hours whose next-24 h fall beats
    3 in/day by > 10%, next to `obs_recession_frac` for the observed river over
    the same windows.
- **Rain-scaling test.** Rerun each start with future rain × 0, 0.5, 1, 1.5.
  `mono_viol_frac`: share of hours where more rain gives a forecast > 5 mm
  lower. `zero_rain_late_climb_cm`: with no future rain, the largest climb
  after hour 72 (water already in transit should have arrived by then).
- **Robustness.** Four perturbed-rain runs per start (±12/24 h timing shifts
  and 6-hourly lognormal noise, σ = 0.6), reported as `mae_perturbed`.

## Data findings

These affect every earlier model, not only the September one.

1. **Rain vanishes during differential gaps.** The merged frame is built by an
   inner join on the differential, so hours without a differential reading have
   no rain or flow either. That's 14% of hours at Isis and 67% of 2019. The
   September trainer then splices across these gaps as if the hours were
   contiguous, so some 10-day targets are misaligned.
2. **Rain is undercounted where the differential is hourly.** The differential
   is recorded hourly in 2017–2018 (Isis) and 2017–2019 (Godstow), and
   15-minutely afterwards. The inner join keeps only the rain reading that
   lands on the hour, so those years train on about a quarter of the real rain.
   Over the common hours, the old frame has 16–23% less rain than the gauges.
3. **The differential is a staircase.** At Isis it moves in abrupt
   5–10 cm steps with flat stretches between, while Farmoor flow is smooth
   (sluice/weir operation). Hour-to-hour noise is about 3 mm median, and 1% of
   hours move more than 4.5 cm. Some "kinks" and "rises without rain" in the
   observations are these steps, which no model can time. A realistic
   forecast is a smooth curve through them.
4. **The 3 in/day recession rule is not a hard physical limit for these
   gauges.** The observed differential falls faster than 7.6 cm in 24 h in
   4.4% (Isis), 5.4% (Godstow) and 11.5% (Wallingford) of all hours, and in
   18–30% of hours when D > 0.4 m. A hard clamp holds recessions up.
5. **Differential tracks Farmoor flow closely** (Pearson 0.955 / 0.980 / 0.974,
   Spearman 0.91 / 0.95 / 0.98 for Isis / Godstow / Wallingford). This is the
   basis for the rating-curve redesign.
6. **About 3% of sizeable rises (> 10 cm / 24 h) have < 2 mm of catchment rain
   in the previous 10 days.** They are almost certainly weir operations or
   gauge artefacts, and they teach a model that dry-weather rises happen.

Fixes in the new data path (`load_merged`, `physics_data.py`):

- Rain and flow come from the full gauge records on a continuous hourly grid.
- A station that isn't reporting stays missing rather than zero.
- The differential keeps its gaps.
- Gaps of up to 24 h are interpolated for the encoder only, never beyond t0.
- Missing target hours are masked in the loss.

## Step 2: diagnosis of the September model

The September model was run four ways on the test window, all with stage-1
predicted flow:

- `sept_full`: live (clamps plus flow hold).
- `sept_recession`: recession clamp only.
- `sept_off_blend`: no clamps, but Farmoor flow still held at its last value
  until 2 mm of rain.
- `sept_raw`: no clamps and no flow hold.

Table: `figures/eval/summary_test.csv`.

| test window | Isis full / raw | Godstow full / raw | Wallingford full / raw |
|---|---|---|---|
| MAE (m) | 0.0484 / 0.0459 | 0.0662 / 0.0589 | 0.1068 / 0.1075 |
| MAE 1–24 h | 0.0174 / 0.0193 | 0.0210 / 0.0258 | 0.0256 / 0.0268 |
| bias (m) | +0.017 / +0.010 | +0.010 / −0.007 | +0.030 / +0.010 |
| climbs > 1 cm while dry | 44% / 54% | 22% / 32% | 16% / 52% |
| max kink (mm/h²) | 13.2 / 19.5 | 12.8 / 19.6 | 14.3 / 23.2 |
| reversals per forecast | 7.2 / 7.9 | 5.3 / 6.7 | 5.8 / 7.7 |
| more rain → lower forecast | 5.3% / 0.1% | 6.6% / 0.0% | 0.4% / 0.0% |
| 24 h falls > 3 in/day (observed: 5.3 / 6.0 / 11.4%) | 0% / 2.3% | 0% / 3.0% | 0% / 4.5% |

Where the problems come from:

1. **Kinks and jitter mostly come from the LSTM itself; the clamps only trim
   them.** With clamps off, the kinks are sharper and more frequent.
   - Two-thirds to three-quarters of kinks over 5 mm/h² fall after hour 27,
     spread over the forecast: the free-running decoder wiggling as it
     follows hourly rain.
   - The next-biggest group is in hours 1–3. The raw model jumps 7–13 mm
     (median) in its first hour. In the live forecast that first step sits
     at exactly 3.2 mm, the recession clamp's hourly limit, so the clamp is
     hiding a start jump rather than fixing it.
   - The 24 h plateau-pin release adds a few kinks at hours 23–26.
2. **Dry-weather climbs are learned, not a clamp artefact.** The raw model
   climbs more than 1 cm in dry weather in 32–54% of forecasts. The clamps
   reduce this but don't remove it. Contributing causes:
   - Stage 2 is trained on the observed future Farmoor flow, so it learns to
     follow flow. At forecast time it gets stage 1's prediction instead.
   - The dry-weather loss terms look only at rain in the same hour. So they
     also penalise water still arriving days after a storm, blurring the
     link between rain and rises.
   - The training data undercounts rain in 2017–2019 and loses it during
     gaps (see Data findings), and contains weir-operation rises with no rain.
3. **Rain non-monotonicity is caused by the Farmoor flow hold.** With no rain,
   forecast flow is frozen at its last value. A little rain blends it toward
   stage 1's (usually receding) forecast, so a little rain lowers the forecast.
   Without the hold (`sept_raw`) it essentially never happens (0–0.1%); with
   the hold it happens in 5–11% of hours.
4. **The clamps cost accuracy after day one.**
   - Good: they help the first 24 h, by capping the start jump.
   - Bad: they raise overall MAE at Isis and Godstow, raise winter MAE, and
     add an upward bias. The recession clamp allows 0% fast falls; the real
     river has them in 5–11% of hours, 18–30% at high water.
5. **Train/inference feature mismatch.** At forecast time the live code builds
   features from only the last 720 rows. The 720 h rolling rain feature can
   therefore only be computed for the final hour, and it's back-filled across
   the whole 100 h encoder window, which training never saw.
   `sept_raw_longhist` measures the effect (below).

## Other checks

- **History truncation (finding 5 in step 2) is minor.** Giving the September
  model 1,100 h of history instead of 720 h (`sept_raw_longhist`) changes test
  MAE by < 1 mm (Isis 0.0462 vs 0.0459; Godstow 0.0594 vs 0.0589).
- **September without Farmoor flow is worse at Wallingford.** The saved
  no-flow variant scores test MAE 0.139 raw, against 0.108 with flow.
- **A second test window (Feb–Sep 2026) isn't feasible.**
  - The historical differentials come from the flags.jamesonlee.com
    archive, which ends 2026-01-20.
  - The EA flood-monitoring API keeps only about four weeks of level readings
    (earliest available 2026-08-29).
  - The EA Hydrology API has one relative level series per lock, not the
    upstream and downstream stage the differential formulas use.
  - That leaves at most two forecast starts (5–17 September, dry late
    summer), too few to be worth it.

## Step 3: candidates

Model selection uses the 2023 validation year: the trainer's validation MAE
over about 1,300–1,700 windows per location (stride 4 h), plus the harness on
weekly 2023 starts for realism. The test window is only scored for the final
comparison.

### Candidate A: the two-stage LSTM, fixed properly (`lstm_v2*`)

Same September encoder features and hourly LSTM decoder, but:

- no clamps
- trained on the corrected hourly data
- mean-station rain and season as decoder inputs
- realism taught through the loss instead of clamps: dry-climb penalty on
  the 72 h / 2 mm rule, curvature penalty, paired "more rain must not lower
  the forecast" penalty

Variants:

- **`lstm_v2`** (rain only; penalty weights 2 / 0.5 / 5 plus a recession
  penalty). Collapsed to a near-flat forecast: validation MAE stuck at 0.114.
  The penalties are easiest to satisfy by predicting no change. Stopped.
- **`lstm_v2a`** (rain only, no penalties, batch 32). Learns slowly: best
  validation MAE 0.0725 at epoch 23 (Isis). Without future Farmoor flow the
  decoder has to learn the whole rain response itself.
- **`lstm_v2f` / `lstm_v2fp`.** Stage-2 decoder also gets stage-1 *predicted*
  Farmoor flow, precomputed for every training window with the September
  Farmoor model, flow hold off. `stage1_flow.py`'s batch forecast matches
  `predict_flow_hourly` exactly. This removes the train/forecast mismatch.
  `fp` adds gentle penalties (dry climb 0.5, curvature 0.05, rain
  monotonicity 1.0, no recession penalty).

### Candidate B: physically structured hybrid (`hybrid_*`)

`src/flag_predictor/models/physics_models.py`. An LSTM encoder reads the same
100 h of past features but only sets the *initial state* of a small
differentiable rainfall-runoff model:

- soil wetness
- quick / medium (3-store cascade) / slow (groundwater) stores
- a channel store

Future rain drives that model hour by hour. A learned monotonic rating curve
maps channel outflow Q to the differential, and is initialised from the
observed Farmoor-flow → D quantile map. Guaranteed by construction, with no
clamps:

- The forecast starts exactly at the observed D(t0).
- The quick/medium stores can hold at most the last 24 h / 168 h of rain, and
  groundwater starts at or below the current outflow. So with no recent and
  no future rain, Q and D can only hold or fall.
- More rain never lowers the forecast.
- Trajectories are reservoir-smooth.

Variants (validation MAE from the trainer, Isis / Godstow / Wallingford):

| name | change | Isis | Godstow | Wallingford |
|---|---|---|---|---|
| `hybrid_v1` | baseline structure | 0.0524 | 0.0533 | 0.0758 |
| `hybrid_ema` | + EMA of weights (0.995) | 0.0542 | 0.0530 | 0.0761 |
| `hybrid_ps` | + EMA, encoder also modulates k_quick, k_medium, β and the runoff split per forecast | 0.0527 | 0.0499 | 0.0724 |
| `hybrid_aux` | + EMA, latent Q pulled toward observed Farmoor flow (weight 0.02) | 0.0579 | 0.0538 | 0.0742 |
| `hybrid_h128` | + EMA, encoder 128 hidden, dropout 0.3 | 0.0540 | 0.0519 | 0.0743 |
| `hybrid_c1` | C1 (smooth) rating curve | 0.0508 | 0.0513 | 0.0748 |
| `hybrid_c1_st` | + learned rain-gauge weights | 0.0516 | 0.0547 | 0.0715 |
| `hybrid_c1_ps` | C1 curve + per-forecast parameters (no EMA) | **0.0506** | 0.0506 | **0.0719** |
| `hybrid_c1_ps_st` | + gauge weights | 0.0521 | **0.0505** | **0.0719** |
| `hybrid_w` | v1 with first-day loss weights 5 / 2.5 / 1 | 0.0557 | 0.0520 | 0.0767 |
| `hybrid_v1_s1`, `_s2` | v1, seeds 1 and 2 | 0.0519, 0.0508 | 0.0547, 0.0548 | 0.0735, 0.0789 |

**Seed noise.** Three seeds of the same v1 setup span 0.0508–0.0524 (Isis),
0.0533–0.0548 (Godstow) and 0.0735–0.0789 (Wallingford). That's as large as
most differences between variants. So the final model averages several seeds,
and single-run gaps under about 2 mm (Wallingford: 5 mm) aren't read as real.

**Chosen setup: `hybrid_c1_ps`.** Best or joint best at all three locations.
Gauge weights and first-day weighting didn't give a consistent gain, so they
were left out.

**Final model: five seeds of `hybrid_c1_ps`, averaged** (`ensemble:hybrid_c1_ps+…_s4`).
Averaging keeps every structural guarantee: a mean of trajectories that start
at D(t0), never climb without water and rise with rain has the same
properties. Validation MAE of the individual seeds:

| seed | Isis | Godstow | Wallingford |
|---|---|---|---|
| 0 | 0.0506 | 0.0506 | 0.0719 |
| 1 | 0.0491 | 0.0529 | 0.0755 |
| 2 | 0.0518 | 0.0530 | 0.0724 |
| 3 | 0.0497 | 0.0501 | 0.0735 |
| 4 | 0.0503 | 0.0508 | 0.0724 |

**Training on imperfect rain** (`hybrid_c1_ps_rn`). In operation the future
rain is a forecast. This variant trains with the future rain perturbed:
6-hourly lognormal noise σ = 0.5, plus a random ±18 h timing shift.
Validation MAE with the observed rain gets worse (0.0540 / 0.0569 / 0.0747),
as expected. As a five-seed ensemble, compared with the plain five-seed
ensemble on the weekly 2023 starts (Isis / Godstow / Wallingford):

| | MAE, observed rain | MAE, perturbed rain |
|---|---|---|
| plain ensemble | **0.0423 / 0.0432 / 0.0617** | 0.0569 / **0.0619** / 0.1067 |
| noisy-rain ensemble | 0.0427 / 0.0499 / 0.0634 | **0.0543** / 0.0627 / **0.0999** |

It trades accuracy with good rain for robustness to bad rain at Isis and
Wallingford only. Kept as an option (`hybrid_c1_ps_rn*`). The right choice
depends on how wrong real rain forecasts are, which needs an archive of past
forecasts to backtest against.

Notes:

- **Piecewise-linear rating curve (v1).** A recession that sweeps log Q past a
  knot shows a visible corner, e.g. hours 80–100 and ~165 in
  `figures/eval/examples_val_isis.png`. The C1 curve (slope linear between
  knots, exact quadratic inverse) removes them. It's also more accurate at
  all three locations, so it becomes the base.
- **EMA** doesn't help on its own. **Per-forecast parameters** help relative
  to their EMA base at all three locations. **The Farmoor-flow auxiliary
  loss** helps Wallingford a little and hurts Isis. **Gauge weights** help
  Wallingford (tributaries below Farmoor) and hurt Godstow.
- **LSTM with predicted flow** (`lstm_v2f`): validation MAE 0.0543 / 0.0540 /
  0.0843 (Isis / Godstow / Wallingford), much closer to the hybrid. With the
  gentle penalties trained from scratch (`lstm_v2fp`) it collapses to the flat
  solution again (0.1143).
- **Curriculum** (`lstm_v2fc`): start from the trained `lstm_v2f`, then
  fine-tune with the gentle penalties at lr 3e-4. It still collapses to flat
  (0.1141) within the first epoch. With these loss terms the flat forecast is
  a strong attractor.
- **Candidate A on the weekly 2023 starts (Isis).**
  - Accuracy: `lstm_v2f` MAE 0.048, and the best first-day MAE of any model
    (0.0138, level with persistence).
  - Realism: still unrealistic without constraints. It climbs > 1 cm in dry
    weather in 13% of forecasts, max kink 8.3 mm/h², 8.6 reversals per
    forecast, rain non-monotonic in 0.6% of hours, and a late climb after
    removing all rain.
  - It's better than the September model on every realism measure, but far
    from the hybrid.

### What the hybrid learned

Global parameters of `hybrid_c1_ps`, seed 0. The encoder also nudges the
quick/medium time constants, β and the split per forecast.

| | Isis | Godstow | Wallingford |
|---|---|---|---|
| quick store time constant (h) | 9.0 | 10.3 | 9.0 |
| medium store time constant, ×3 in cascade (h) | 18.5 | 19.5 | 18.7 |
| groundwater time constant (h) | 475 | 448 | 335 |
| channel time constant (h) | 22 | 29 | 19 |
| runoff exponent β | 2.15 | 2.11 | 2.05 |
| soil capacity (mm) | 76 | 77 | 103 |
| evapotranspiration, winter base / midsummer extra (mm/h) | 0.012 / 0.106 | 0.007 / 0.157 | 0.012 / 0.117 |
| runoff split quick / medium / groundwater | 18 / 58 / 24% | 18 / 58 / 25% | 17 / 59 / 25% |

The three locations were trained independently yet agree closely, which
suggests the parameters are physically identifiable rather than curve-fitting.

- **Travel times.** About 2.3 days mean travel on the medium path; groundwater
  recession over 14–20 days (Farmoor flow halved over about three weeks in the
  Dec 2024 recession).
- **Evapotranspiration.** It peaks at roughly 2.6–3.8 mm/day in midsummer and
  sits around 0.2–0.3 mm/day in winter, plausible values for southern England
  that came out of rain and river data alone.

## Final results (step 3, test window)

Frozen window: weekly starts 2024-11-01 → 2026-01-05, observed rain, all
clamps off except in `sept_full`. 57 / 60 / 61 starts at Isis / Godstow /
Wallingford. `figures/eval/summary_test.csv` has every column.

### Accuracy (MAE, m)

| | Isis | Godstow | Wallingford |
|---|---|---|---|
| Persistence | 0.0863 | 0.1125 | 0.1847 |
| September, live (clamped) | 0.0484 | 0.0662 | 0.1068 |
| September, raw output | 0.0459 | 0.0589 | 0.1075 |
| A: LSTM + predicted flow (`lstm_v2f`) | 0.0420 | 0.0423 | 0.0664 |
| B: hybrid, first version (`hybrid_v1`) | 0.0358 | 0.0416 | 0.0651 |
| B: hybrid, final setup, one seed (`hybrid_c1_ps`) | 0.0310 | 0.0431 | 0.0635 |
| **B: hybrid, final setup, 5 seeds averaged** | **0.0312** | **0.0404** | **0.0599** |

MAE by lead time, live September vs final hybrid:

| | Isis | Godstow | Wallingford |
|---|---|---|---|
| 1–24 h | 0.0174 → **0.0146** | 0.0210 → **0.0170** | **0.0256** → 0.0267 |
| 25–72 h | 0.0303 → **0.0261** | 0.0429 → **0.0253** | 0.0612 → **0.0449** |
| 73–240 h | 0.0582 → **0.0351** | 0.0794 → **0.0481** | 0.1313 → **0.0690** |
| winter only | 0.0936 → **0.0575** | 0.1154 → **0.0624** | 0.1883 → **0.1063** |
| with perturbed rain | 0.0569 → **0.0401** | 0.0778 → **0.0558** | 0.1266 → **0.0799** |
| bias | +0.017 → +0.008 | +0.010 → +0.005 | +0.030 → +0.017 |

On the first day, the LSTM (0.0117 / 0.0183 / 0.0202) and persistence (0.0146
at Isis) are the ones to beat. The hybrid matches persistence at Isis and
beats it at Godstow and Wallingford.

### Realism (on each model's raw output)

| | live September | final hybrid | LSTM (A) |
|---|---|---|---|
| forecasts climbing > 1 cm in dry weather (I / G / W) | 44 / 22 / 16% | **0 / 2 / 0%** | 16 / 17 / 28% |
| sharpest kink (mm/h²) | 13.2 / 12.8 / 14.3 | **0.6 / 0.6 / 1.3** | 6.8 / 6.0 / 9.4 |
| direction reversals per forecast | 7.2 / 5.3 / 5.8 | **0.6 / 0.8 / 0.8** | 4.5 / 6.9 / 7.9 |
| hours where more rain gives a lower forecast | 5.3 / 6.6 / 0.4% | **0 / 0 / 0%** | 0.2 / 0 / 0.2% |
| with no future rain, climb after hour 72 (cm) | 0 / 0 / 0 (clamped) | **0 / 0 / 0** | 0.16 / 0.22 / 0.34 |
| 24 h falls > 3 in/day (observed: 5.3 / 6.0 / 11.4%) | 0 / 0 / 0% | 2.7 / 5.6 / 9.9% | 2.4 / 3.5 / 9.7% |

The hybrid's single "dry climb" at Godstow is water from rain 3–7 days before
t0 draining through the medium stores. That's physically legitimate, and the
72 h rule counts it as dry.

### Flag colour (Isis and Godstow)

Share of observed forecast hours with the right flag colour
(`flag_accuracy.py`, `figures/eval/flag_accuracy_test.csv`).

| | Isis | Isis, first day | Isis, ≥ 2 colours wrong | Godstow | Godstow, first day |
|---|---|---|---|---|---|
| Persistence | 80.0% | 95.5% | 12.7% | 80.4% | 96.6% |
| September, live | 83.3% | 93.3% | 4.9% | 87.3% | 95.1% |
| A: LSTM | 86.7% | 96.7% | 4.0% | 91.7% | 95.6% |
| **Final hybrid** | **89.0%** | **97.3%** | **2.4%** | **92.4%** | **96.6%** |

The 2023 validation year shows the same ordering (Isis 88.1% vs 79.3%;
Godstow 93.2% vs 89.0%).

### Plots

`figures/eval/examples_{test,val}_{isis,godstow,wallingford}.png`. Each shows
the three biggest storms and three driest spells in the window. Under each, the
same start is rerun with all future rain removed. The live model shows its
hour-24 plateau-release jumps (e.g. Isis 2025-01-24: 0.22 → 0.30 m with no
rain) and flatlines on its floors (Wallingford 2025-02-28 stuck at 0.92 m
while the river fell to 0.4 m). The LSTM drifts upward in dry spells. The
hybrid only recedes.

### Known weaknesses of the hybrid

- **First-day MAE at Wallingford** is 1 mm worse than the live model. The
  LSTM is better still on the first day everywhere. The staircase makes "no
  change for a few hours" hard to beat, and the hybrid always moves
  smoothly.
- **The biggest flood peaks are under-predicted** (Wallingford Jan 2025:
  1.6 vs 1.9 m), though by less than the other models.
- **More sensitive to rain-forecast error than the raw September model** in
  relative terms. With perturbed rain its MAE rises 29–38%, against 18–19%
  for September. It's still lower in absolute terms at every location.
  Training on perturbed rain (`hybrid_c1_ps_rn`) helps with bad rain at Isis
  and Wallingford but costs accuracy with good rain (Candidate B above).
- **All testing uses observed rain,** with synthetic perturbations as a proxy.
  There's no archive of past forecast rain to backtest against, beyond one
  10-day window in February 2026 that has no observed differential.

## Deployment notes (not done)

Swapping the hybrid into `generate_all_location_figures.py` needs more than
loading different weights:

1. **A continuous hourly history.** The live script builds history with the
   September merge (differential rows only), so there's a hole between the
   end of the differential archive (2026-01-20) and the start of the EA API's
   ~4-week window. The hybrid's encoder needs about 820 h of continuous rain
   (its 720 h rolling rain feature) and about 270 h of differential.
   - Rain: CI already downloads the qualified EA rain CSVs up to near-present
     (see the workflow). Building rain and flow from those plus the API, as
     `evaluation.load_merged` does, closes the gap.
   - Differential: the API's 4 weeks are enough.
2. **Ensemble prediction.** Per member, call
   `candidates.ensemble_predictor([...5 seeds...], location)` with that
   member's station rain. The past-only encoder inputs are computed once per
   start and shared across members and seeds, so 50 members × 5 seeds takes a
   few seconds per location.
3. **Weights to commit:** `models/redesign_hybrid_c1_ps*_{location}.{pt,pkl}`
   (15 weight files of about 300 kB, plus their small configs).
4. **Drop the clamps.** None of `apply_hourly_physics` / `apply_recession_limit`
   / the flow blend applies; the model doesn't use stage-1 flow at all.

---

# October 2026: continuous state, flow pre-training, model-error spread

Branch `forecast-improvements`. Three ideas from the review of the September
hybrid: (1) archive the live forecasts, (2) track the catchment state
continuously instead of inferring it from 100 h of features, (3) pre-train the
hydrology on the long Farmoor flow record. Plus a model-error spread for the
forecast band.

## Hydrology pre-trained on Farmoor flow (idea 3)

`train_continuous_models.py pretrain`. The reservoir model of the hybrid
(soil bucket → quick / 3-cascade medium / groundwater → channel), plus a small
direct-runoff share `f_direct` that ignores soil wetness, driven only by
mean-station gauge rain from 2000 and fitted to Farmoor flow (mm/h over the
1609 km² catchment) by multiple shooting (12-week chunks simulated in parallel,
each starting where the previous one ended on the previous epoch). Train:
2001 – Nov 2022 and Jan – Oct 2024; validation 2023; test from Nov 2024.

| core | ET | train NSE / log-NSE | 2023 NSE / log-NSE | test NSE / log-NSE | bias (test) |
|---|---|---|---|---|---|
| `cs_cos` | seasonal cosine | 0.90 / 0.88 | **0.92 / 0.96** | 0.91 / 0.91 | +1 % |
| `cs_pet` | k_et × ERA5 ET0 | 0.88 / 0.88 | 0.87 / 0.92 | 0.92 / 0.89 | −8 % |

A rain-only simulation reproduces Farmoor flow well, including the first
rises after the dry summer of 2025 (`figures/core_cs_cos_flow.csv.gz`). Real
ET0 (ERA5 via Open-Meteo, fetched by the one-off workflow on the
`data/era5-et` branch) did not help: the learned crop factor is ≈ 1.0 and the
fit is no better than with the seasonal cosine, so it was not used further.

Learned parameters (`cs_cos`): quick 24 h, medium 25 h ×3, groundwater 480 h,
channel 35 h, β 4.4, soil capacity 259 mm, f_direct 2 %, runoff split
quick / medium / groundwater 8 / 10 / 82 %. Fitted to *flow volume* the
catchment is groundwater-dominated (consistent with the high baseflow of the
Cotswold limestone tributaries); fitted to the *differential* the September hybrid put only
24 % through groundwater: the differential is far more sensitive to the quick
response than the volume is.

## Continuous state (idea 2)

Three ways of using the continuous simulation in the forecast, all with the
same windows, encoder features and loss as the September hybrid
(validation MAE, 2023, seed 0; September hybrid seed 0: 0.0506 / 0.0506 / 0.0719,
its five seeds span 0.0491–0.0518 / 0.0501–0.0530 / 0.0719–0.0755):

| variant | what | Isis | Godstow | Wallingford |
|---|---|---|---|---|
| `hybrid_cs_cos` | forecast starts from the simulated state (encoder only nudges it); flow-trained hydrology frozen | 0.0533 | 0.0573 | 0.0849 |
| `hybrid_cs_ft` | same, hydrology fine-tuned on the differential (re-simulated every epoch) | 0.0485 | 0.0525 | 0.0806 |
| `hybrid_cs_cap` | + quick/medium stores capped by 72 h / 336 h rain instead of 24 h / 168 h | 0.0515 | 0.0542 | – |
| `hybrid_cs_old` | starts from the simulated state, hydrology = the September hybrid's | 0.0504 | 0.0597 | 0.0845 |
| `hybrid_simfeat` | September hybrid unchanged, simulated soil wetness, flow, groundwater outflow, fast-store water and sim/observed flow ratio added as encoder inputs | **0.0476** | **0.0502** | **0.0686** |

Test window, every 6 h (MAE, m, seed 0, observed rain; `compare_batch.py`):

| | Isis | Godstow | Wallingford | autumn 2025 Isis / Godstow / Wallingford |
|---|---|---|---|---|
| September hybrid | 0.0299 | 0.0472 | 0.0659 | 0.046 / 0.073 / 0.110 |
| `hybrid_cs_ft` | 0.0379 | 0.0462 | 0.0713 | 0.080 / 0.085 / 0.098 |
| `hybrid_cs_old` | 0.0352 | 0.0498 | 0.0772 | 0.064 / 0.084 / 0.164 |
| `hybrid_simfeat` | 0.0293 | 0.0477 | 0.0629 | 0.053 / 0.078 / 0.104 |

Starting the forecast *from* the simulated state is worse than letting the
encoder infer the state, at every location on the test window. The river
itself integrates the catchment's memory: the current differential, its
recession and Farmoor flow already tell the encoder most of what the soil and
groundwater stores hold, and pinning the start to a rain-only simulation
imports the simulation's own errors (e.g. wherever gauge rain under- or
over-catches). Giving the simulation to the encoder as extra information
(`hybrid_simfeat`) is the variant that can only help, and it is level with or
slightly better than the September hybrid for seed 0.

### Five seeds of `hybrid_simfeat` against the deployed five-seed hybrid

Validation MAE of the extra seeds (Isis / Godstow / Wallingford): s1 0.0530 /
0.0555 / 0.0724, s2 0.0478 / 0.0519 / –, s3 0.0491 / 0.0536 / 0.0761,
s4 0.0507 / 0.0538 / 0.0701; seed 0 was a lucky draw. Ensembles of five,
every 6 h (`compare_batch.py`), with 95 % intervals from a paired block
bootstrap over whole weeks:

| | 2023 (validation) | Nov 2024 – Jan 2026 (test) |
|---|---|---|
| Isis | 0.0468 → 0.0456 (−2.8 %, CI −6.0 to +0.3) | 0.0298 → 0.0277 (**−7.0 %, CI −13.4 to −0.4**) |
| Godstow | 0.0481 → 0.0503 (+4.4 %, CI −3.0 to +12.5) | 0.0432 → 0.0426 (−1.3 %, CI −6.0 to +4.2) |
| Wallingford | 0.0661 → 0.0658 (−0.4 %, CI −6.5 to +6.6) | 0.0585 → 0.0588 (+0.5 %, CI −4.5 to +7.5) |

First storms after the 2025 drought (starts 20 Oct – 15 Dec 2025): Isis
0.049 → 0.042, Godstow 0.069 → 0.065, Wallingford 0.107 → 0.103.

The weekly-start harness at Isis (`evaluate_candidates.py`, realism included):

| Isis | MAE | 1–24 h | winter | perturbed rain | right flag | ≥ 2 colours off | max kink (mm/h²) |
|---|---|---|---|---|---|---|---|
| 2023: hybrid ×5 | 0.0423 | 0.0169 | 0.0486 | 0.0569 | 88.1 % | 1.8 % | 0.80 |
| 2023: simfeat ×5 | **0.0388** | **0.0155** | **0.0430** | **0.0544** | 87.7 % | **1.3 %** | 0.69 |
| test: hybrid ×5 | 0.0312 | 0.0146 | 0.0575 | 0.0401 | 89.0 % | 2.4 % | 0.61 |
| test: simfeat ×5 | **0.0281** | **0.0132** | **0.0514** | **0.0378** | **90.3 %** | **1.7 %** | 0.54 |

Same architecture, so the same structural guarantees (no dry climbs, rain
monotonicity 0 %, no late climb without rain).

**Decision (17 Oct): `hybrid_simfeat` for the Isis only** (chosen on 2023,
confirmed on the test window). **Revised 18 Oct: one model for all three
locations, so the plain hybrid runs everywhere; `hybrid_simfeat` is kept but
switched off (`SIMFEAT_MODELS = {}`).** The original reasoning, for reference: Godstow is worse on 2023 and level on test; Wallingford is
level on both, so both keep the simpler plain hybrid. Live, the Isis
simulation needs ~760 days of gauge rain; CI keeps the long EA rain CSVs and
refreshes them weekly when stale; with < 90 % coverage the Isis falls back to
the plain hybrid. Model-error library for it: `models/error_library_hybrid_simfeat_isis.npz`
(CRPS 0.0270 vs point MAE 0.0341 in cross-validation).

## Model-error spread

`build_error_library.py`. For each location, the deployed ensemble is re-run
every 6 h over 2023 and the test window (≈ 2,400–2,800 starts) with observed
rain. Each start's 240 h error trajectory is stored with six descriptors
(level, 24 h trend, log rain over the next 3 and 10 days, log rain over the
past week, season). A forecast member borrows 10 trajectories from its k
nearest past starts (weighted standardised distance; level counts double).
Whole trajectories are borrowed, so errors keep their persistence.

Cross-validation between the two periods (errors from one build bands for
the other), hybrid ×5:

| k | Isis CRPS raw / calibrated | Godstow | Wallingford | raw share in 10–90 % band |
|---|---|---|---|---|
| 40 | 0.0303 / 0.0436 | 0.0371 / 0.0460 | 0.0504 / 0.0620 | 64–67 % |
| 100 | 0.0288 / 0.0318 | 0.0351 / 0.0368 | 0.0477 / 0.0502 | 69–71 % |
| 200 | **0.0280 / 0.0288** | **0.0343 / 0.0349** | **0.0471 / 0.0479** | 72–74 % |
| 500 (Isis) | 0.0285 / 0.0288 | | | 77 % |
| 1100 (Isis) | 0.0304 / 0.0313 | | | 76 % |

Point MAE of the same starts: 0.0359 / 0.0449 / 0.0614. Conditioning helps a
little (k = 200 beats near-unconditional sampling) and the raw bands are a
little too narrow across periods, because each period alone has seen fewer
kinds of weather; a per-lead factor (1.15–1.2 at 24 h, 1.25–1.4 at 240 h),
fitted so 80 % of held-out observations fall in the 10–90 % band, is stored
with each library.

Why it matters: in the 81 archived operational forecasts (27 Sep – 17 Oct
2026, a dry spell) the observation was inside the rain-only 10–90 % spread
in 0–21 % of hours over the first three days (Isis 0–1 %), because the rain
members barely disagree in dry weather while the river still moves by a
centimetre or two (weirs, model error).

## Lock-keeper step layer (flag-change box)

Question from the site's user: if the flag is green, the river is heading up
and the forecast takes it into amber soon, the forecast should show the
*jump* (keepers move the gates mostly 08:00–16:00) instead of a smooth ramp
that crosses the line overnight.

What the 15-minute record says (steps ≥ 3 cm in 15 min, 4 cm at Wallingford,
held 2 h): 56–79 % of steps fall 08:00–14:00 UK time; their direction follows
the 24 h trend (Isis 73 % up when rising / 89 % down when falling;
Wallingford 95 % / 95 %); stretches jump within about an hour of each other.
At hourly resolution (as the live site sees it) about half of the *up*
moves at Isis and Godstow are 08:00–14:00 but only 34–40 % of observed
up-*flag-changes* fall 10:00–16:00 (rain-driven rises cross a line at any
hour), against 65–70 % of down-changes.

### Versions

1. **Rules by hand** (mockups): jump towards the smooth curve in keeper hours.
   Helped dry falls, made rises worse.
2. **Hazard given the true future** (v1): logistic hazard fitted on the
   observed record, conditioned on where the river *really* was 24 h later,
   then driven by forecast outcomes. Wrong in principle: the hazard learned
   "if the river will be 25 cm higher tomorrow, move now", and the smooth
   outcome rises early, so jumps came up to a day early. Brier worse than the
   smooth outcomes; even with the true future as input it was poor.
3. **Hazard given the forecast** (deployed): rows are (forecast start, hour
   1–72) on the training years with the hybrid's own forecasts (rain that
   fell), starting each hour from where the river really was (teacher
   forcing). Inputs: pending move to the forecast at +0/6/12/24 h, time of
   day, weekend, past 24 h trend, rain over the past day, hours since the last
   jump, forecast hour, and jumps by this / a neighbouring stretch in the 3 h
   before the start. Jump size: fitted on the pending moves plus a factor drawn
   from past jumps (median 0.8, 10–90 % 0.4–1.7: keepers often overshoot).
   Drift between jumps: ridge regression on the pending moves. Outcomes keep
   their borrowed model error smoothed over 25 h as the target, and the
   error's hour-scale wobble (minus a 5 h running median) is added back after
   stepping: without the wobble the stepped outcomes never brush a flag line
   the way the real hourly readings do, and same-day changes came out too low.

Hazard check (log loss, 2023 / test; constant rate in brackets): Isis up
0.046 (0.055) / 0.021 (0.032), Godstow up 0.089 (0.103) / 0.052 (0.069),
Wallingford up 0.040 (0.057) / 0.029 (0.043); down similar.

### Backtest (2023 and Nov 2024 – Jan 2026, starts every 3 h, rain that fell)

Out of sample for the hybrid, the step layer and the error library (errors
borrowed from the other period). A flag change counts when the new flag holds
for 3 hours.

First 48 h, cm (smooth outcomes → stepped):

| | CRPS | median MAE |
|---|---|---|
| Isis | 1.63 → **1.52** | 2.14 → **2.03** |
| Godstow | 1.97 → **1.91** | 2.62 → **2.56** |
| Wallingford | 2.54 → **2.48** | 3.31 → **3.25** |

Flag change held 3 h, Brier (smooth → stepped; none of the differences is
outside a weekly block-bootstrap 95 % interval on its own):

| | up today | down today | up by end of tomorrow | down by end of tomorrow |
|---|---|---|---|---|
| Isis | 0.0173 → 0.0160 | 0.0214 → 0.0201 | 0.0440 → 0.0428 | 0.0310 → 0.0299 |
| Godstow | 0.0109 → 0.0112 | 0.0100 → 0.0099 | 0.0324 → 0.0323 | 0.0220 → 0.0225 |

When the smooth outcomes said a rise was ≥ 30 % likely by the end of tomorrow
(Isis, 417 starts), they gave 47 % for *today* against 31 % observed; stepped
36 % (Brier 0.163 → 0.147). By the end of tomorrow: 84 % smooth, 74 % stepped,
68 % observed (0.180 → 0.160). Pooled over Isis and Godstow, both windows and
both directions, the stepped probabilities are better calibrated:

| forecast | 0.35–0.6 | 0.6–0.8 | 0.8–1 |
|---|---|---|---|
| smooth: observed share | 0.37 | 0.52 | 0.84 |
| stepped: observed share | 0.42 | 0.65 | 0.89 |

Timing of the first change by the end of tomorrow (share of outcomes within
3 h of the real change): falls Isis 31 % → 41 %, Godstow 29 % → 40 %; rises
Isis 28 % → 31 %, Godstow 24 % → 25 %.

Neighbour / own jump in the 3 h before the start: once the forecast says the
river is moving, it adds almost nothing (hazard log loss over the first 6 h
0.0185 → 0.0184 Isis up, 0.0210 → 0.0207 Isis down, similar at Godstow; the
fitted coefficients are slightly negative). Raw counts make it look strong
(daytime: own up-jump in the next 3 h 13 % after a neighbour jumped vs 2 %
otherwise) but given a forecast rise of ≥ 5 cm it is 39 % vs 27 %, and the
forecast already carries most of that. Kept, since it costs nothing. Farmoor
flow is not used: its live feed (EA hydrology API) could not be checked for
delay from here.

What it does not fix: where the river ends up. In the two big February 2025
rises the hybrid (with the rain that fell) stopped 10–25 cm short; the step
layer puts the jump on the right morning (Isis 10 Feb: smooth crossing
08:00, stepped 11:00, real 11:30; Godstow 24 Feb: 14:00 / 11:00 / 11:15) but
jumps to the level the outcomes say.

## Official OURCs flag vs the bands (preview page)

Data: the OURCs flag history pages (every change since Dec 2018, UK time; saved
without names as `data/flags/history_<river>.csv`), the 15-minute differential
(extended to Oct 2026 from flags.jamesonlee.com, identical on the overlap) and
the Iffley Lock upstream stage since 2017 (EA hydrology API, station 1501TH).
The Isis flag is set by counting bucks at Iffley weir; Godstow's from this
differential only since Sep 2024.

Agreement (share of monitored time the flag equals the band): Isis 81 % (since
Nov 2021, when Light/Dark Blue started), Godstow 87 % (since 2019).

Flag changes (single steps), differential at the change, median:

| Isis | going up | coming down |
|---|---|---|
| Green ↔ Light Blue (line 0.215) | 0.231 | 0.165 |
| Light ↔ Dark Blue (0.33) | 0.332 | 0.267 |
| Dark Blue ↔ Amber (0.44) | 0.422 | 0.381 |
| Amber ↔ Red (0.535) | 0.597 (0.563 over 2019–2026) | 0.499 |

Godstow: Green → Amber 0.49, Amber → Red 0.82; Red → Amber 0.61, Amber → Green
0.38. Going up the flag usually follows the band within an hour or two (median
0.5 h Isis, 1.5 h Godstow); coming down it lags (median 3.8 h / 5.8 h after the
band, 5–8 cm below the line). 93 % of changes are between 08:00 and 20:00.

Bands kept as they are: the Isis Amber/Red line at 0.535 matches the flag best
over 2019–2026 (92 % of Amber/Red time; 0.56: 89 %, 0.60: 85 %).

**Black (Isis)** is not a differential band: the differential saturates near
0.95 (Black-flag hours: 10–90 % range 0.81–0.99, overlapping Red). On the
Iffley upstream stage the flag went Black at 0.36–0.47 m and back to Red at
0.03–0.25 m. Rule: Black once Iffley ≥ 0.42 m, Red again below 0.22 m —
matches the flag in 97.8 % of Red/Black hours since 2019 (92 % of Black hours
caught, 1.4 % false Black). No single threshold does as well (≥ 0.45 m: never
wrong but catches only 37 % of Black hours).

**Official-flag model** (`official_flag.py`, `fit_official_flag.py`): hourly
logistic hazard of a change up / down, pooled over both rivers with a river
term. Inputs: how far past the next line (up) or below the flag's own line
(down) the river is, hours it has been there, daylight (08–20) and 10–17,
weekend, hours since the flag was set, 6 h trend. A change up goes to the band
the river is in; a change down one flag. Leave-one-season-out log loss vs a
constant rate: Isis up 0.0189 (0.0216), down 0.0309 (0.0427); Godstow up
0.0089 (0.0173), down 0.0204 (0.0300).

Backtest of the official-flag box (starts every 3 h; Isis 2023 + Nov 2024 – Jan
2026, Godstow Nov 2024 – Jan 2026; flag model refitted without the period it is
tested on; rain that fell). Event: the official flag has changed by the end of
today / tomorrow (UK). Brier, flag model vs the flag-band outlook used as a
forecast of the official flag (climatology in brackets):

| | up today | down today | up by tomorrow | down by tomorrow |
|---|---|---|---|---|
| Isis | **0.024** vs 0.036 (0.038) | 0.037 vs 0.042 (0.052) | **0.040** vs 0.051 (0.081) | 0.065 vs 0.068 (0.114) |
| Godstow | **0.013** vs 0.021 (0.030) | **0.022** vs 0.032 (0.037) | **0.021** vs 0.032 (0.064) | **0.033** vs 0.049 (0.084) |

Bold: weekly block-bootstrap 95 % interval of the difference excludes zero. The
gain is mostly when the flag and band disagree at the start (Isis down by
tomorrow 0.146 vs 0.282; Godstow 0.173 vs 0.666). Going up the flag model is
well calibrated (Isis: forecast 0.6–0.8 → 69 %, > 0.8 → 87 %); coming down it
is too cautious (Isis forecast 0.2–0.35 → 46 % happened, 0.35–0.6 → 69 %), so
a recalibration of the "down" hazard is the obvious next step.

## Gauge outages (Isis gauges offline from 13 Oct 2026)

When a gauge stops, the stretch's forecast freezes at its last reading and the
page says so ("Stale forecast — sensor offline"). On `/preview` the flag-change
box is replaced by a note, and no sync note or official-flag outlook is worked
out, because both would compare an old differential with the flag flying now.
A Black note needs an Iffley reading less than 4 h old.

When the gauge comes back, the production history has a hole of several days.
The hybrid's encoder (100 h window) only bridges holes of up to a day, so the
rest is NaN. Test: 40 random starts Nov 2024 – Jan 2026, Isis, rain that fell,
with a 3–12-day hole ending 1–24 h before the start. Mean absolute error (m)
over the first 48 h / 10 days. No hole: 0.0186 / 0.0349.

| hole, back for | left as NaN | straight line | shaped by Godstow |
|---|---|---|---|
| 7 days, 1 h | 0.0298 / 0.0429 | 0.0210 / 0.0446 | **0.0196** / 0.0437 |
| 12 days, 1 h | 0.0333 / 0.0502 | 0.0224 / 0.0508 | **0.0207** / 0.0486 |
| 7 days, 24 h | **0.0185** / **0.0327** | 0.0205 / 0.0400 | 0.0198 / 0.0388 |
| 12 days, 24 h | **0.0188** / **0.0328** | 0.0201 / 0.0387 | 0.0197 / 0.0392 |

"Shaped by Godstow" means: during the hole, the Isis differential = b × the
Godstow differential, plus a residual drawn straight between the readings
either side. b is the 2019–2026 slope of 24 h changes (0.67), pulled towards
the slope fitted on the two weeks before the hole when Godstow moved enough
then to show it. It reconstructs the hole about twice as well as a straight
line for 7–12-day holes: mean error 2.2 vs 3.8 cm and 3.1 vs 6.5 cm at the
Isis, with similar gains at Godstow (from the Isis) and Wallingford (from the
Isis).

So `hybrid_live.live_frame` bridges an outage of up to 14 days, shaped by the
neighbour, for the first 12 h after the gauge is back. After that the hole is
left to the encoder as before. In one example (Isis, 12 Feb 2025, a 6-day hole
ending 5 h before the start), the chance of Red→Amber by the end of tomorrow
was 96 % unbridged, 85 % bridged and 92 % with no hole.

The page's history line now breaks at missing hours rather than drawing a
straight line across them.

## Official flag coming down: recalibration (issue #4)

The official-flag model was too cautious about flags coming down. Over the box
backtest starts, the chance of coming down by the end of tomorrow came out as
follows:

| River | Forecast | Happened |
|---|---|---|
| Isis | 0.20–0.35 | 46 % |
| Isis | 0.60–0.80 | 77 % |
| Godstow | 0.20–0.35 | 70 % |

Running the flag model along the observed river (no forecast error) gives the
same pattern, so the hazard is the cause, not the forecast paths:

| River | Forecast | Happened |
|---|---|---|
| Isis | 0.20–0.35 | 46 % |
| Isis | 0.60–0.80 | 91 % |
| Godstow | 0.35–0.60 | 91 % |

Structural changes to the hourly "down" hazard barely moved the season-CV log
loss. The changes tried were daylight hours below the line, a hinge 5 cm below
the line, a morning term and river-specific run terms. Log loss went from
0.0309 to 0.0301 at the Isis and from 0.0204 to 0.0195 at Godstow.

So the chance of coming down is recalibrated instead. The recalibration is a
Platt fit on the window probability, logit q = 1.114 + 1.171 logit p, one
function for today and by tomorrow (`DOWN_RECALIBRATION` in `official_flag.py`).
Cross-fitted by 3-month blocks, the Brier score of the flag coming down is:

| | Before | After | 95 % interval of the change (weekly blocks) |
|---|---|---|---|
| Isis, today | 0.0371 | 0.0354 | [−0.0068, +0.0040] |
| Isis, by tomorrow | 0.0652 | 0.0585 | [−0.020, +0.006] |
| Godstow, today | 0.0217 | 0.0194 | [−0.0055, +0.0010] |
| Godstow, by tomorrow | 0.0333 | 0.0242 | [−0.018, −0.001] |

The same recalibration does not help the chance of flags going up, so that is
unchanged.

## Black flag forecast (issue #5)

The target is the Iffley Lock upstream level (EA 1501TH) 1–48 h ahead, with the
hysteresis rule run along it: Black at ≥ 0.42 m, back to Red below 0.22 m. The
OURCs history has only 5 Black periods, so the level is modelled, not the flag.

**What predicts Iffley** (hours with Iffley > 0.15 m, fitted on 2017–2022,
tested on 2023 – Jan 2026). The persistence spread is 6 cm at 24 h. With the
observed future differentials, a regression explains 34 % of the 24 h change.
With the hybrid's forecast differentials it explains nothing (R² −0.03),
because the observed changes are partly Iffley itself, which is in the Isis
formula. Iffley is also fed by the Cherwell, which no differential sees. What
does carry over is the following:
- Iffley now, and its 6 h and 24 h change.
- The Isis and Godstow differentials now.
- The rain of the last 24 h and 72 h, and the rain over the lead.

**Model** (`black_flag.py`): one linear model per lead on those inputs. There is
a variant without the differentials for when a gauge is offline. Each rain
member gives a mean path, plus a random walk with the model's spread at each
lead (2.5 cm at 6 h, 5 cm at 24 h, 6 cm at 48 h).

**Test** (2023 – Jan 2026, starts every 3 h, rain that fell). Brier against the
test-period rate:

| | 24 h | 48 h |
|---|---|---|
| Iffley reaches 0.42 m | 0.0025 vs 0.0046 (40 events) | 0.0068 vs 0.0092 (80) |
| Official flag Red → Black, from Red | 0.0094 vs 0.0148 (21) | 0.0247 vs 0.0345 (50) |
| Official flag Black → Red, from Black | 0.042 vs 0.060 (14) | 0.068 vs 0.140 (37) |

Reliability, Red → Black within 24 h:

| Forecast | Happened | Starts |
|---|---|---|
| 0.05–0.20 | 16 % | 31 |
| 0.20–0.50 | 70 % | 10 |
| > 0.50 | 94 % | 16 |

**Past cases:**
- **12 Feb 2024, 07:00:** Iffley at 0.44 m, so the card shows Black about 6.5 h
  before OURCs raised it.
- **13 Jan 2024, 07:00:** the flag was Black and Iffley 0.32 m and falling. The
  card gave a 58 % chance of Red by the end of tomorrow; the flag came down at
  13:37 the next day.
- **25 Nov 2024 (Storm Bert): missed.** Iffley had been drawn down to 0.01 m (a
  12 cm drop in a day) and then rose to 0.76 m within two days.

**On the page:** a Black card appears only when Black is close. That means the
flag is Black, Iffley is within 6 cm of 0.42 m, or there is at least a 10 %
chance of Black by the end of tomorrow.

## Gauges offline: estimate from the neighbouring stretch (Oct 2026)

While a stretch's gauges are offline (last reading more than 4 h old) and its
neighbour's are not, the live forecast no longer freezes at the last reading.
`hybrid_live.outage_nowcast` extends the history hourly to the neighbour's last
reading with d(t) = d_last + b × (n(t) − n(t_last)) (Isis from Godstow, b ≈ 0.67;
Godstow from the Isis, b ≈ 0.96; b pulled towards the slope of the two weeks
before, as in `fill_outage`), carries the gauge rain and Farmoor flow on, and
runs the hybrid from there. Every outcome is shifted by a draw of the estimate's
error, so the likely range and the flag chances carry it. The page marks the
stretch "Gauges offline — estimated from …", draws the estimated part of the
history dotted, and says "about" before the differential. After five days the
estimate is marked low confidence and no "likely" alerts go out for that
stretch; alerts before then say they are estimated. Both offline, or more than
14 days: the stale forecast as before. The rain history on the page carries on
to now from the EA rain gauges either way.

`backtest_outage_nowcast.py` (2019–2026). Part 1, the estimate itself at
≈1,300 random hours per stretch: mean error 0.019 / 0.030 / 0.046 / 0.062 m
after 1 / 3 / 7 / 14 days down (Isis), against 0.029 / 0.069 / 0.120 / 0.170 m
for the last reading held; Godstow similar. Errors are heavy-tailed and about
1.4–1.6 × larger in the middle of the range. The sd model
(`NOWCAST_ERROR`: a √days, × a mid factor) puts 78–88 % of outcomes inside
±1.28 sd (target 80 %).

Part 2, the hybrid forecast with rain that fell, ~35 starts per stretch in
2023 and Nov 2024 – Jan 2026 (mean error, m, first 48 h / 10 days; right band
in the first 48 h):

| | days down | from the true level | from the estimate | stale forecast |
|---|---|---|---|---|
| Isis | 2 | 0.019 / 0.038, 91 % | 0.027 / 0.041, 89 % | 0.030 / 0.048, 87 % |
| Isis | 5 | 0.020 / 0.040, 92 % | 0.039 / 0.047, 88 % | 0.036 / 0.070, 86 % |
| Isis | 10 | 0.017 / 0.038, 92 % | 0.045 / 0.056, 85 % | 0.051 / 0.113, 81 % |
| Godstow | 2 | 0.023 / 0.042, 99 % | 0.039 / 0.052, 98 % | 0.040 / 0.051, 98 % |
| Godstow | 5 | 0.022 / 0.042, 99 % | 0.042 / 0.056, 99 % | 0.043 / 0.075, 95 % |
| Godstow | 10 | 0.022 / 0.039, 99 % | 0.044 / 0.049, 97 % | 0.063 / 0.113, 93 % |

"Stale forecast" is the forecast from the last reading, read off at the same
hours, and is flattered here: it was run with the rain that fell, where the live
one had forecast rain. The estimate costs about 0.01–0.03 m over the first two
days against a working gauge; against the stale forecast it is about as good
over the first two days and much better over ten, and it starts now, so the
flag chances (which need a current start) come back.

## Model soil wetness vs measured soil moisture (COSMOS-UK, Oct 2026)

COSMOS-UK (UKCEH) cosmic-ray soil moisture, daily, from its public API
(`cosmos-api.ceh.ac.uk`, collection `1D`, parameter `cosmos_vwc`). In the
catchment: Chimney Meadows (CHIMN, Thames floodplain near Bampton, live to
yesterday); Wytham Woods (WYTH1) closed in 2016. `data/cosmos/CHIMN_1D_data.csv`
holds 2018 – Oct 2026.

The hybrid's soil store at the start of each forecast (`s`, the fraction full,
mean of the five copies), every day at 12:00 UTC 2019 – Jan 2026, against the
measured volumetric water content:

| | r | r of anomalies from the seasonal cycle |
|---|---|---|
| Godstow model soil | 0.79 (test window 0.86) | 0.47 |
| Isis model soil | 0.82 (test window 0.87) | 0.48 |
| rain-memory index, half-life 20 d | 0.70 | 0.76 |
| rain-memory index, half-life 40 d | 0.78 | 0.78 |

The model's soil store follows the measured seasonal cycle closely (it gets
wet and dry in the right months, and was not tuned to do so), but saturates at
80–90 % once the measured soil passes ~42 %, and follows the wet and dry spells
within a season less well than a simple 40-day rain-memory index.

Does the mismatch matter? Error-library forecasts (rain that fell) started at
12:00 on the same days, mismatch = measured minus the soil the model implies
(linear fit), 72 h error = observed − forecast:

| | starts | corr(mismatch, 72 h error) | mean 72 h error: model wetter / agree / measured wetter |
|---|---|---|---|
| Godstow, all | 636 | +0.08 | −0.005 / −0.011 / +0.001 m |
| Godstow, 15 mm+ rain ahead | 96 | +0.25 | −0.012 / −0.003 / +0.025 m |
| Isis, all | 604 | +0.05 | −0.005 / −0.014 / +0.003 m |
| Isis, 15 mm+ rain ahead | 94 | +0.30 | −0.038 / −0.010 / +0.046 m |

With heavy rain ahead, when the ground is wetter than the model thinks the
river ends up higher than forecast, and the other way round: a 0.04–0.08 m
spread, about the size of the typical error in these cases. A measured soil
moisture input (to the encoder, or to set the soil store's starting value)
looks worth trying; the samples are small (≈ 95 starts, a dozen or so events).

## Extra inputs: measured soil moisture and upstream levels (branch soil-upstream, Oct 2026)

`models/extra_inputs.py`, `train_extra_models.py`, `compare_extra_models.py`. The deployed
set-up (hybrid_c1_ps) with extra encoder inputs, joined after the usual features:

* COSMOS-UK soil moisture at Chimney Meadows (`soil_vwc`, `soil_vwc_d7`), each day's value
  used only from noon the next day (when it is published);
* eight EA level gauges upstream (Windrush at Worsham, Evenlode at Shipton, Coln at Fairford,
  Thames at Cricklade, Cherwell at Heyford Bridge, Ray at Islip, Thame at Wheatley, Ock at
  Abingdon), hydrology API, hourly 2018 – Oct 2026 (`data/upstream/`): level above its 14-day
  minimum (`rise`), and the 6 h and 24 h changes.

Input sets: `all` (26), `soil` (2), `soil_rise` (soil + the 8 rises, 10). Starts every 6 h in
2023 and Nov 2024 – Jan 2026 (not fitted), rain that fell; mean absolute error (m), one
trained copy each against the deployed seed-0 copy:

| | Isis all | Godstow all | Godstow test | Godstow heavy rain ahead | Wallingford all | Wallingford test |
|---|---|---|---|---|---|---|
| deployed (seed 0) | 0.0370 | 0.0484 | 0.0472 | 0.0812 | 0.0681 | 0.0659 |
| retrained, no extras (control) | 0.0379 | 0.0477 | 0.0460 | 0.0774 | | |
| `all` (hybrid_su) | 0.0400 | 0.0479 | 0.0448 | 0.0826 | | |
| `soil` (hybrid_s) | 0.0408 | 0.0484 | 0.0464 | 0.0730 | | |
| `soil_rise` (hybrid_sr) | 0.0400 | **0.0461** | **0.0440** | **0.0707** | **0.0662** | **0.0601** |

The five deployed copies span 0.0370–0.0390 (Isis), 0.0458–0.0488 (Godstow) and
0.0655–0.0681 (Wallingford), so single-copy differences of a few mm are noise. At the Isis
every extra-input version is worse than every deployed copy, while the control is not: the
extras do not help there (the Isis is set by the Iffley weir and the Cherwell, which they add
little to) and the Isis keeps the deployed model. At Godstow `soil_rise` is the best version;
five copies against five:

| Godstow, 5 v 5 | all | test window | heavy rain ahead | mid-range and rising | 1-24 h | right band (test) |
|---|---|---|---|---|---|---|
| deployed | 0.0449 | 0.0432 | 0.0732 | 0.0695 | 0.0190 | 92.4 % |
| hybrid_sr | **0.0443** | **0.0413** | **0.0677** | **0.0663** | 0.0193 | **92.9 %** |

Its error library (k = 200): calibrated CRPS 0.0335 against 0.0349 for the deployed model, 80 %
of observations inside the 10–90 % band. Shown on the preview page only
(`--preview-model hybrid_sr`; a stretch without hybrid_sr copies keeps the deployed model).
