#!/usr/bin/env python3
"""Multi-horizon evaluation that actually rolls the model forward.

Why this exists
---------------
The shipped checkpoint is a NEXT-DAY model: it was trained with
forecast_horizon=1. baseline_comparison.py evaluates it at +1d, +3d, +5d and
+7d, but applies a single forward pass at every horizon - so at +7d it grades a
one-day forecast against truth a week later. Persistence, meanwhile, is a fair
baseline at every horizon. The comparison is therefore only valid at +1d; the
longer leads understate the model by construction.

This script closes that gap by advancing the model autoregressively. At each
step the predicted SIC becomes the next day's SIC channel, while the six
atmospheric and oceanic channels are taken from the archive - the standard
"perfect forcing" assumption. That isolates what is being tested: the model's
own ice dynamics, not its ability to also forecast the weather driving them.

Perfect forcing makes these numbers an UPPER BOUND on operational skill. A real
deployment would feed forecast forcing, which carries its own error. Said plainly
here because it is the kind of caveat that otherwise gets lost.

    python scripts/evaluation/rollout_comparison.py --horizons 1 3 5 7 10 14
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DAILY = ROOT / "data/processed/regridded/daily"
MASK = ROOT / "data/processed/land_ocean_mask_ps25.npy"
NORM = ROOT / "data/processed/normalization_stats.json"
CKPT = ROOT / "models/checkpoints/phase3/sic_unet_convlstm_v001_best.pt"
OUT = ROOT / "output/evaluation/rollout_comparison.json"

VARS = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]
INPUT_WINDOW = 7

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_day(d: date) -> np.ndarray | None:
    f = DAILY / str(d.year) / f"{d.strftime('%Y%m%d')}.npz"
    if not f.exists():
        return None
    return np.load(f)["data"].astype(np.float32)


def normalise(stack: np.ndarray, norm: dict) -> np.ndarray:
    """Z-score every channel except SIC, exactly as training did."""
    out = stack.copy()
    for c, v in enumerate(VARS):
        if v != "sic" and v in norm and norm[v]["std"] > 1e-6:
            out[:, c] = (out[:, c] - norm[v]["mean"]) / norm[v]["std"]
    return out


def build_model(device):
    import torch
    from seaice_forecast.models.unet_convlstm import UNetConvLSTM

    m = UNetConvLSTM(
        in_channels=7, output_channels=1, seq_len=INPUT_WINDOW,
        encoder_channels=[16, 32, 64, 128], convlstm_layers=1,
        dropout=0.0, output_activation="sigmoid",
    )
    ck = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(device)
    return m


def rollout(model, torch, start: date, horizon: int, mask, norm) -> np.ndarray | None:
    """Advance the model `horizon` days from `start`, returning predicted SIC."""
    window = []
    for i in range(INPUT_WINDOW):
        day = load_day(start - timedelta(days=INPUT_WINDOW - 1 - i))
        if day is None:
            return None
        window.append(day)
    window = np.stack(window)                     # [T, C, H, W] raw

    for step in range(1, horizon + 1):
        x = normalise(window, norm)
        x[:, 1:, mask == 0] = 0.0
        with torch.no_grad():
            pred = model(torch.from_numpy(x).unsqueeze(0).float().to(
                next(model.parameters()).device))
        pred = pred.squeeze().cpu().numpy()
        pred = np.clip(pred, 0.0, 1.0)
        pred[mask == 0] = 0.0

        if step == horizon:
            return pred

        # Slide the window: real forcing for the new day, our own SIC for it.
        nxt = load_day(start + timedelta(days=step))
        if nxt is None:
            return None
        nxt = nxt.copy()
        nxt[0] = pred                              # channel 0 is SIC
        window = np.concatenate([window[1:], nxt[None]], axis=0)

    return None


def build_climatology(train_start: str, train_end: str, mask) -> dict:
    """Day-of-year mean SIC from the TRAIN split only, so there is no leakage."""
    acc, cnt = {}, {}
    d = datetime.strptime(train_start, "%Y-%m-%d").date()
    end = datetime.strptime(train_end, "%Y-%m-%d").date()
    while d <= end:
        day = load_day(d)
        if day is not None:
            doy = min(d.timetuple().tm_yday, 365)
            acc[doy] = acc.get(doy, 0.0) + day[0]
            cnt[doy] = cnt.get(doy, 0) + 1
        d += timedelta(days=1)
    return {k: acc[k] / cnt[k] for k in acc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 7, 10, 14])
    ap.add_argument("--stride", type=int, default=5,
                    help="days between forecast start dates (keeps runtime sane)")
    ap.add_argument("--max-samples", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="write results here instead of the committed location")
    args = ap.parse_args()

    import torch
    import yaml

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "src/seaice_forecast/config/settings.yaml").read_text())
    rng = cfg["data"]["date_ranges"]

    mask = np.load(MASK)
    norm = json.loads(NORM.read_text())
    model = build_model(device)

    logger.info("Building climatology from the train split (no leakage)...")
    clim = build_climatology(rng["train_start"], rng["train_end"], mask)

    test_start = datetime.strptime(rng["test_start"], "%Y-%m-%d").date()
    test_end = datetime.strptime(rng["test_end"], "%Y-%m-%d").date()

    results = {}
    for h in args.horizons:
        preds, pers, clm, tgts = [], [], [], []

        d = test_start + timedelta(days=INPUT_WINDOW)
        while d + timedelta(days=h) <= test_end and len(tgts) < args.max_samples:
            truth = load_day(d + timedelta(days=h))
            last = load_day(d)
            if truth is not None and last is not None:
                p = rollout(model, torch, d, h, mask, norm)
                if p is not None:
                    preds.append(p)
                    pers.append(last[0])           # persistence: today's ice
                    doy = min((d + timedelta(days=h)).timetuple().tm_yday, 365)
                    clm.append(clim.get(doy, np.zeros_like(last[0])))
                    tgts.append(truth[0])
            d += timedelta(days=args.stride)

        if not tgts:
            logger.warning("+%dd: no samples", h)
            continue

        tgt = np.stack(tgts)
        # The ice zone is the honest metric: ~83% of the domain is permanently
        # ice-free and 0.0 in truth and in every forecast, which dilutes MAE
        # roughly sixfold and flatters everything equally.
        ice_zone = ((tgt > 0.15).any(axis=0)) & (mask == 1)

        def mae(arr):
            a = np.stack(arr)
            return float(np.abs(a - tgt)[:, ice_zone].mean())

        row = {
            "n_samples": len(tgts),
            "model": {"mae_ice": round(mae(preds), 4)},
            "persistence": {"mae_ice": round(mae(pers), 4)},
            "climatology": {"mae_ice": round(mae(clm), 4)},
        }
        row["beats_persistence"] = row["model"]["mae_ice"] < row["persistence"]["mae_ice"]
        row["beats_climatology"] = row["model"]["mae_ice"] < row["climatology"]["mae_ice"]
        results[f"+{h}d"] = row

        logger.info(
            "+%-3dd  n=%-3d  model=%.4f  persistence=%.4f  climatology=%.4f   %s",
            h, len(tgts), row["model"]["mae_ice"], row["persistence"]["mae_ice"],
            row["climatology"]["mae_ice"],
            "BEATS persistence" if row["beats_persistence"] else "beaten by persistence",
        )

    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "method": "autoregressive rollout, perfect forcing",
        "caveat": (
            "The six atmospheric and oceanic channels are taken from the archive "
            "at every step, so these figures are an upper bound on operational "
            "skill: a real deployment feeds forecast forcing, which carries its "
            "own error."
        ),
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "test_split": f"{rng['test_start']}..{rng['test_end']}",
        "stride_days": args.stride,
        "horizons": results,
    }, indent=2))
    logger.info("\nWrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
