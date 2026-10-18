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
