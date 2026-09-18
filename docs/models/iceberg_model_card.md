# Model card — Antarctic iceberg drift predictor

## Purpose

Predict 24-hour iceberg displacement from position, size, season and local
forcing, so a route planner can avoid predicted berg positions.

## Status

**Working, and the stronger of the two trained models here.** 3.913 km RMS
24-hour position error on held-out fixes, 96.7% within 10 km, 9.5% skill over a
constant-velocity baseline.

## Architecture

Two stages, both scikit-learn `HistGradientBoosting`:

1. **Gate** — a classifier deciding whether a berg is moving at all.
2. **Regressors** — separate models for `u_berg` and `v_berg`, applied **only**
   to bergs the gate says are moving.

Why two stages: most tracked bergs are grounded or fast-locked on any given day.
A single regressor trained on all of them is dragged toward zero and learns to
predict "barely moving" for everything. Gating first lets the regressors fit the
bergs that are actually in motion.

Implementation: `iceberg-drift/src/iceberg_drift/models/gbm_models.py`.

## Input

27 features per berg-fix: position; size (two axes); wind `u`/`v` and speed;
current `u`/`v` and speed; lagged velocities at 1, 2 and 3 steps; rolling means;
`frac_moving`; observation count; track age; and seasonal `sin`/`cos` of month.

## Output

```python
{"moving": bool, "u_ms": float, "v_ms": float, "drift_km_24h": float}
```

Stationary bergs return exactly zero displacement, not a small non-zero guess.

## Training data

US National Ice Center Antarctic iceberg fixes, 2008-01-13 → 2018-12-21.
14,913 fixes across 103 bergs, co-located with ERA5 wind and GLORYS12 currents.
See [`../DATA_LINEAGE.md`](../DATA_LINEAGE.md).

## Evaluation methodology

Held-out fixes, 24-hour position error in km. Reported as RMS **and** median,
because the distribution is heavily skewed: most bergs barely move, so a median
alone flatters and an RMS alone is dominated by a few fast bergs.

Compared against a **constant-velocity** baseline (the berg keeps doing what it
was doing) and against a single-stage GBM, to show the two-stage split earns its
complexity.

## Results

| variant | RMS km | median km | within 10 km | skill vs constant |
|---|---|---|---|---|
| constant velocity | 4.324 | 0.766 | 95.7% | 0.000 |
| single-stage GBM | 4.004 | 0.370 | 96.5% | 0.074 |
| **two-stage (shipped)** | **3.913** | **0.186** | **96.7%** | **0.095** |

Source: `iceberg-drift/output/drift_twostage_metrics.json`.

The median improves 4× over constant velocity while RMS improves only 9.5% —
exactly what the two-stage design predicts. It is much better on typical bergs;
the tail of fast-moving bergs still dominates squared error.

## Known limitations

1. **103 bergs.** Small, and they are the large, trackable ones. Behaviour of
   small bergs and growlers is not represented — and those are the ones that
   damage hulls.
2. **Modest skill over constant velocity** (9.5% RMS). Most of the win is on
   typical bergs, not the fast ones that matter most for collision avoidance.
3. **24-hour horizon only.** Longer predictions require iterating, which
   compounds error and has not been validated.
4. **Point positions, not swept volumes.** It predicts where a berg centre will
   be, not the area it sweeps during a passage.
5. **No calving or fragmentation.** A berg is assumed to persist intact.
6. **Not real-time.** Training fixes end 2018-12-21.

## Failure cases

| Situation | Behaviour |
|---|---|
| Berg grounds mid-interval | gate may still say moving; predicted displacement overshoots |
| Berg breaks free after long grounding | gate likely says stationary; displacement underestimated |
| Missing wind or current feature | row dropped, never imputed — an imputed wind drives a physical prediction |
| Berg outside training region | extrapolation, unvalidated |

## Intended use

- Predicting 24-hour displacement of large, tracked Antarctic bergs.
- Iceberg proximity as one term in a route cost function.

## Not intended for

- **Collision avoidance in real time.** Advisory only; it does not replace radar,
  visual watch or official ice charts.
- **Small bergs and growlers**, which are absent from the training data and are
  the primary hull-damage risk.
- **Horizons beyond 24 hours** without revalidation.
- **Arctic bergs**, which calve from different glaciers into different currents.
