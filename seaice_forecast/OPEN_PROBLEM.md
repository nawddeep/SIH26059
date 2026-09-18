# Open problem: 60-95% of training batches produce non-finite loss

## Summary

A U-Net + ConvLSTM trained on Antarctic sea-ice concentration produces a
**non-finite (NaN/Inf) loss on 20-95% of training batches**, varying run to run
with identical configuration. Gradients are always finite. The model still
learns from surviving batches and reaches positive forecast skill, but training
is inefficient and epoch statistics are unusable.

Environment: PyTorch 2.x, Apple MPS (M-series, 16 GB), Python 3.9, fp32.

## Setup

| | |
|---|---|
| Model | U-Net (encoder widths 16/32/64/128) + ConvLSTM at bottleneck, 7.85 M params |
| Input | `[B, T=7, C=7, H=332, W=316]` — 7 days x 7 variables, z-score normalised |
| Output | `[B, 1, 332, 316]`, **sigmoid** activation |
| Target | Sea-ice concentration in `[0, 1]` |
| Loss | masked MAE over 83,019 ocean pixels (land excluded by a 0/1 mask) |
| Optimiser | Adam, lr 5e-5, weight_decay 0, grad-clip 0.1 |
| Data | 2,404 training samples, all verified finite |

```python
class MaskedMAELoss(nn.Module):
    def forward(self, predictions, targets, mask):
        if mask.ndim == 2:   mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3: mask = mask.unsqueeze(1)
        mask = mask.expand_as(predictions)
        errors = torch.abs(predictions - targets)
        masked_errors = torch.where(mask.bool(), errors, torch.zeros_like(errors))
        n = mask.sum()
        return masked_errors.sum() / n if n > 0 else masked_errors.sum()
```

## The contradiction

With a sigmoid output and targets in `[0, 1]`, every element of `errors` is in
`[0, 1]`. The loss is `sum(masked_errors) / count(mask)`, a mean over a subset,
so it is **mathematically bounded by 1.0**.

Verified directly:

```
mask dtype float32, shape (332, 316), unique values [0.0, 1.0], sum 83019
random pred/target, mask [H,W]      -> loss 0.332764
random pred/target, mask [1,1,H,W]  -> loss 0.332764
pred=1, target=0 (worst case)       -> loss 1.000000     <- the bound holds
```

Yet training reports epoch-average losses of **1e10 to 1e19**:

```
epoch 09  train=0.01607           lr=5.00e-05
epoch 10  train=1.23e11           lr=5.00e-05
epoch 11  train=8.17e10           lr=5.00e-05
epoch 12  train=0.01560           lr=5.00e-05
epoch 13  train=0.01497           lr=5.00e-05
epoch 14  train=0.01418           lr=5.00e-05
epoch 15  train=1.59e11           lr=5.00e-05
```

Note the learning rate is identical across spiking and non-spiking epochs.
Since each batch loss should be <= 1.0, an epoch average of 1e11 should be
impossible.

## What has been ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Corrupt input data | scanned all 2,829 train files for non-finite values | 0 found; max abs z-score 18.1 |
| Unnormalised inputs | normalisation wired in and verified | all channels now in ~[-3, 3] |
| Gradient explosion | split skip counters into loss vs gradient | **0 non-finite gradients, ever** |
| `clip_grad_norm_` turning Inf into NaN | measure norm before clipping, skip if non-finite | never triggers |
| MPS vs CPU numerics | same weights, same 68 val batches, both devices | identical: 0/68 non-finite, mean loss 0.008540 both |
| Train-mode BatchNorm / dropout | same 40 batches, `train()` vs `eval()` | 0 vs 0 non-finite |
| Loss function itself | unit-tested, see above | correctly bounded by 1.0 |
| Residual output head | tested with and without | fails both ways |
| Learning rate | 1e-3, 1e-4, 5e-5 | spikes at all three |
| Weight decay | 1e-4 and 0 | fails both ways |
| Poisoned weights | inspected checkpoint after a failing epoch | all weights and BN buffers finite |

## The part that does not reproduce

A standalone loop — same dataset, same model, same loss, same mask handling,
same optimiser, real forward + backward + step on MPS — runs **0/100 batches
non-finite**, and a separate 400-batch run was also clean:

```
step   0 loss=0.009189 gradnorm=0.015
step 200 loss=0.010192 gradnorm=0.012
step 400 loss=0.007538 gradnorm=0.013   (400 batches, no NaN)
```

The failure appears only inside the project's `TemporalTrainer.train_epoch`
loop over the full 601-batch epoch. Something differs between that loop and the
standalone reproduction, and that difference has not been found.

## Current workaround

Asymmetric guards:

- **Training**: skip only non-finite losses. (A stricter `loss > 1.0` bound was
  tried and rejected 577/601 batches, starving learning.)
- **Validation**: skip both non-finite and `> 1.0`, since checkpoint selection,
  the LR scheduler and early stopping all depend on that number being real.
  Before this, validation reported `8.4e12` and `Best val loss: 0.000000`.

The model does train through this and reaches genuine skill (see below), but
20-95% of each epoch is discarded.

## Result despite the problem

Held-out test split 2017-2018, MAE over the active ice zone, **n = 300 forecast
start dates**:

| lead | persistence | climatology | model |
|---|---|---|---|
| +1d | **0.0205** | 0.1125 | 0.0466 |
| +3d | **0.0364** | 0.1127 | 0.0552 |
| +5d | **0.0459** | 0.1117 | 0.0613 |
| +7d | **0.0524** | 0.1089 | 0.0654 |

The model beats climatology by roughly 2.4x at +1d, so it has learned real
structure while training on a fraction of the data. **It does not beat
persistence at any horizon.**

### Correction

An earlier version of this table reported persistence at 0.0608 (+5d) and
0.0743 (+7d) and concluded the model beat it at both. That run used ~60 forecast
start dates out of roughly 700 available, and the claim did not survive a larger
sample: persistence at n=300 is 0.0459 and 0.0524. The model's own scores barely
moved, so the original conclusion was sampling noise in the baseline, not skill
in the model. The climatology column moved even more, from ~0.22 to ~0.11, for
the same reason.

Two lessons worth recording rather than quietly fixing: a held-out split of 700
samples is not an excuse to evaluate on 60 of them, and a baseline that looks
unexpectedly weak deserves more suspicion than a model that looks unexpectedly
strong.

### A second caveat on the multi-day rows

The numbers above come from `scripts/evaluation/baseline_comparison.py`, which
applies a single forward pass at every horizon. This checkpoint was trained with
`forecast_horizon=1`, so at +7d that script grades a one-day forecast against
truth a week later. **Only the +1d row is a valid comparison.**

`scripts/evaluation/rollout_comparison.py` advances the model autoregressively
instead, which is the honest multi-day test. Error compounds at every step and
the model does considerably worse there — by +5d it is beaten by climatology as
well as persistence. The scope this checkpoint supports is next-day only.

## The question

Why does `loss.item()` return values above the analytic bound of 1.0 inside the
trainer's epoch loop, when the same loss on the same data and model is correctly
bounded in a standalone loop and in unit tests?

Leading unverified suspicion: an MPS asynchronous-execution or `.item()`
synchronisation issue, where the host reads a buffer before the GPU has finished
writing it. This would explain values that are finite but garbage, why they are
intermittent, why gradients are unaffected, and why a simpler loop does not
reproduce it. Untested — the next step would be `torch.mps.synchronize()` before
each `.item()`, or running one epoch entirely on CPU to see whether the spikes
disappear.
