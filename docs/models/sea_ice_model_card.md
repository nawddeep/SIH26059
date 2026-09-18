# Model card — Antarctic sea-ice concentration forecaster

## Purpose

Predict next-day sea-ice concentration across the Southern Ocean, to cost
navigation routes for research vessels. It is one input to a decision-support
system, not a forecast product in its own right.

## Status

**Research prototype with a known performance limitation.** It is beaten by
persistence at every horizon tested. Read "Known limitations" before using or
citing it.

## Architecture

| | |
|---|---|
| Type | U-Net encoder–decoder with a ConvLSTM bottleneck |
| Framework | PyTorch |
| Parameters | ~7.7 M |
| Encoder channels | 16, 32, 64, 128 |
| ConvLSTM layers | 1 |
| Output activation | sigmoid (bounds output to 0–1 by construction) |
| Dropout | 0.0 at inference |
| Implementation | `seaice_forecast/src/seaice_forecast/models/unet_convlstm.py` |

## Input

`[7 days, 7 channels, 332, 316]`, channels in fixed order:

`sic, wind_u, wind_v, air_temp, sst, current_u, current_v`

Six of the seven are z-scored with training-split statistics; SIC is left in its
native 0–1 range. Land cells are zeroed in the forcing channels.

## Output

`[332, 316]` sea-ice concentration in 0–1, on the NSIDC EPSG:3412 25 km grid.
Land is zeroed after inference.

## Training data

NSIDC CDR v6 sea ice with ERA5 and GLORYS12 forcing, 2008-01-01 → 2015-12-31.
Validation 2016. See [`../DATA_LINEAGE.md`](../DATA_LINEAGE.md).

## Training configuration

| | |
|---|---|
| Loss | masked MAE over the 83,019 ocean cells (land excluded) |
| `forecast_horizon` | **1** — this is a next-day model |
| Input window | 7 days |
| Precision | fp32 |
| Device | Apple MPS |
| Checkpoint selection | best validation loss |

## Evaluation methodology

Held-out 2017–2018 split, MAE over the **active ice zone** — cells reaching >15%
concentration anywhere in the period. The full-ocean metric is reported for
comparability but is not the honest one: ~83% of the domain is permanently
ice-free, is 0.0 in truth and in every forecast, and dilutes MAE about sixfold,
flattering everything equally.

Two scripts, and the distinction matters:

- `baseline_comparison.py` — single forward pass. **Valid only at +1d**, because
  this is a next-day model and at +7d the script grades a one-day forecast
  against truth a week later.
- `rollout_comparison.py` — autoregressive, with the predicted SIC fed back in
  and real forcing from the archive (perfect forcing). The honest multi-day test,
  and an **upper bound** on operational skill since real forcing would itself be
  forecast.

## Baselines

**Persistence** (tomorrow looks like today) and **climatology** (day-of-year mean
from the training split only). Persistence is the standard hard baseline for
short-range sea-ice forecasting and is genuinely difficult to beat at +1d — ice
barely moves in 24 hours.

## Results

At +1d, n = 300 forecast start dates:

| | MAE (ice zone) |
|---|---|
| persistence | **0.0205** |
| model | 0.0466 |
| climatology | 0.1125 |

Multi-day, autoregressive rollout:

| lead | persistence | climatology | model |
|---|---|---|---|
| +1d | **0.0200** | 0.1066 | 0.0454 |
| +3d | **0.0361** | 0.1069 | 0.1062 |
| +5d | **0.0471** | 0.1074 | 0.1509 |
| +7d | **0.0552** | 0.1066 | 0.1768 |

Regenerate with `./verify.sh --recompute`. `scripts/check_claims.py` fails the
build if these numbers stop matching `output/evaluation/`.

## Known limitations

1. **Beaten by persistence at every horizon tested.** It beats climatology ~2.4×
   at +1d, so it has learned real structure, but persistence is the stronger
   baseline and that is the honest summary.
2. **Next-day only.** Driving it autoregressively compounds its error; by +5d it
   is beaten by climatology as well.
3. **Training instability.** 20–95% of training batches produce non-finite loss,
   varying run to run with identical configuration. Gradients are always finite.
   Root cause unresolved — see [`../../seaice_forecast/OPEN_PROBLEM.md`](../../seaice_forecast/OPEN_PROBLEM.md).
4. **Not real-time.** The archive ends 2018-12-31.
5. **Antarctic only.** Trained on the Southern Hemisphere grid.

## Failure cases

| Situation | Behaviour |
|---|---|
| Malformed normalisation stats | previously produced a fully ice-covered ocean that looked plausible; now validated and refused |
| Lead > 1 day | error compounds; by +5d worse than climatology |
| Rapid ice edge advance | smooths transitions; the sharp edge is where persistence wins |
| Missing input day | sample dropped, never interpolated |

## Intended use

- Costing navigation routes at **+1 day** lead, as one input among several.
- Comparative and research use, with the limitations above stated.

## Not intended for

- **Operational navigation decisions.** Advisory only; does not replace an ice
  navigator or official ice charts.
- **Multi-day forecasting.** Use persistence, which is better at every lead here.
- **Safety-of-life decisions.**
- **Arctic use**, or any grid other than the one it was trained on.
- **Claiming skill over persistence.** It does not have it.
