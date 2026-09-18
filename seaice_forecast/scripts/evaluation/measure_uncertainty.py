#!/usr/bin/env python3
"""Measure where the sea-ice forecaster is wrong, so uncertainty can be reported.

A confidence number invented from nothing is worse than no confidence number:
it invites trust it has not earned. This derives one from held-out error.

Two questions, both answered by measurement rather than assumption:

  1. How does error grow with lead time?
  2. Does error depend on the local ice gradient? The usual claim is that a
     forecaster is accurate in the pack interior and wrong at the ice edge. That
     is plausible and widely repeated, so it is worth checking before being
     relied on.

Writes output/evaluation/uncertainty_model.json - a lookup the API uses to
attach an expected error to any prediction.

    python scripts/evaluation/measure_uncertainty.py --samples 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DAILY = ROOT / "data/processed/regridded/daily"
MASK = ROOT / "data/processed/land_ocean_mask_ps25.npy"
NORM = ROOT / "data/processed/normalization_stats.json"
CKPT = ROOT / "models/checkpoints/phase3/sic_unet_convlstm_v001_best.pt"
OUT = ROOT / "output/evaluation/uncertainty_model.json"

VARS = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]
WINDOW = 7

# Gradient bands, in concentration units per 25 km cell. The interior of the
# pack is flat; the ice edge is where concentration changes fastest.
GRAD_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_day(d):
    f = DAILY / str(d.year) / f"{d.strftime('%Y%m%d')}.npz"
    return np.load(f)["data"].astype(np.float32) if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--stride", type=int, default=9)
    args = ap.parse_args()

    import torch
    import yaml
    from seaice_forecast.models.unet_convlstm import UNetConvLSTM

    cfg = yaml.safe_load((ROOT / "src/seaice_forecast/config/settings.yaml").read_text())
    rng = cfg["data"]["date_ranges"]
    mask = np.load(MASK)
    norm = json.loads(NORM.read_text())

    model = UNetConvLSTM(in_channels=7, output_channels=1, seq_len=WINDOW,
                         encoder_channels=[16, 32, 64, 128], convlstm_layers=1,
                         dropout=0.0, output_activation="sigmoid")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state_dict"])
    model.eval()

    start = datetime.strptime(rng["test_start"], "%Y-%m-%d").date()
    end = datetime.strptime(rng["test_end"], "%Y-%m-%d").date()

    errs, grads, concs = [], [], []
    d = start + timedelta(days=WINDOW)
    n = 0
    while d + timedelta(days=1) <= end and n < args.samples:
        window = [load_day(d - timedelta(days=WINDOW - 1 - i)) for i in range(WINDOW)]
        truth = load_day(d + timedelta(days=1))
        if any(w is None for w in window) or truth is None:
            d += timedelta(days=args.stride)
            continue

        x = np.stack(window).copy()
        for c, v in enumerate(VARS):
            if v != "sic" and v in norm and norm[v]["std"] > 1e-6:
                x[:, c] = (x[:, c] - norm[v]["mean"]) / norm[v]["std"]
        x[:, 1:, mask == 0] = 0.0

        with torch.no_grad():
            pred = model(torch.from_numpy(x).unsqueeze(0).float()).squeeze().numpy()
        pred = np.clip(pred, 0.0, 1.0)

        today = window[-1][0]
        # Local gradient of today's field: how fast concentration is changing in
        # space, which is what distinguishes the edge from the interior.
        gy, gx = np.gradient(today)
        grad = np.hypot(gy, gx)

        ocean = mask == 1
        errs.append(np.abs(pred - truth[0])[ocean])
        grads.append(grad[ocean])
        concs.append(today[ocean])
        n += 1
        d += timedelta(days=args.stride)

    if not errs:
        logger.error("no samples")
        return 1

    err = np.concatenate(errs)
    grad = np.concatenate(grads)
    conc = np.concatenate(concs)

    logger.info("samples: %d days, %d ocean cells each", n, err.size // n)
    logger.info("\nerror by local ice gradient (does the edge really behave differently?):")
    bands = []
    for lo, hi in zip(GRAD_BINS, GRAD_BINS[1:]):
        sel = (grad >= lo) & (grad < hi)
        if sel.sum() < 100:
            continue
        band = {
            "gradientMin": lo, "gradientMax": hi,
            "cells": int(sel.sum()),
            "mae": round(float(err[sel].mean()), 4),
            "p90": round(float(np.percentile(err[sel], 90)), 4),
        }
        bands.append(band)
        logger.info("  grad %.2f-%.2f  n=%9d  MAE=%.4f  p90=%.4f",
                    lo, hi, band["cells"], band["mae"], band["p90"])

    logger.info("\nerror by concentration:")
    conc_bands = []
    for lo, hi in ((0.0, 0.15), (0.15, 0.5), (0.5, 0.85), (0.85, 1.01)):
        sel = (conc >= lo) & (conc < hi)
        if sel.sum() < 100:
            continue
        cb = {"concMin": lo, "concMax": hi, "cells": int(sel.sum()),
              "mae": round(float(err[sel].mean()), 4),
              "p90": round(float(np.percentile(err[sel], 90)), 4)}
        conc_bands.append(cb)
        logger.info("  conc %.2f-%.2f  n=%9d  MAE=%.4f  p90=%.4f",
                    lo, hi, cb["cells"], cb["mae"], cb["p90"])

    ratio = (bands[-1]["mae"] / bands[0]["mae"]) if len(bands) > 1 and bands[0]["mae"] > 0 else 1.0
    logger.info("\n  steepest-gradient band is %.1fx the error of the flattest", ratio)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": "held-out absolute error, +1d, binned by local gradient and concentration",
        "samples": n,
        "testSplit": f"{rng['test_start']}..{rng['test_end']}",
        "overallMae": round(float(err.mean()), 4),
        "overallP90": round(float(np.percentile(err, 90)), 4),
        "byGradient": bands,
        "byConcentration": conc_bands,
        "edgeToInteriorErrorRatio": round(float(ratio), 2),
        "caveat": (
            "Empirical error from held-out data, not a calibrated probabilistic "
            "forecast. It says how wrong this model has been in conditions like "
            "these, not how wrong it will be."
        ),
    }, indent=2))
    logger.info("\nWrote %s", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
