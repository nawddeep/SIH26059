#!/usr/bin/env python3
"""
Skill-vs-baseline evaluation on the held-out test split.

Answers the question every reviewer asks first: does the model actually beat
persistence? For daily sea-ice concentration, persistence ("tomorrow = today")
is a very strong baseline at short lead times, so a model is only interesting
where it beats it.

Baselines
  persistence  forecast = SIC on the last input day
  climatology  forecast = day-of-year mean SIC over the TRAIN split only

Metrics are masked to ocean pixels (land excluded) and reported per lead time.

Usage:
    venv/bin/python scripts/evaluation/baseline_comparison.py
    venv/bin/python scripts/evaluation/baseline_comparison.py --checkpoint models/checkpoints/....pt
"""

import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from seaice_forecast.data_processing.dataset_real import RealSeaIceDataset  # noqa: E402
from seaice_forecast.evaluation.metrics import masked_mae, masked_rmse      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("seaice_forecast").setLevel(logging.WARNING)
logger = logging.getLogger("baseline")

DAILY = ROOT / "data/processed/regridded/daily"
MASK = ROOT / "data/processed/land_ocean_mask_ps25.npy"
OUT = ROOT / "output/evaluation"


def load_cfg():
    import yaml
    c = yaml.safe_load(open(ROOT / "src/seaice_forecast/config/settings.yaml"))
    return c["data"]["date_ranges"]


def build_climatology(train_start, train_end, mask):
    """Day-of-year mean SIC from the TRAIN split only (no leakage)."""
    logger.info("Building day-of-year climatology from train split %s..%s", train_start, train_end)
    acc = {}
    ts, te = (datetime.strptime(x, "%Y-%m-%d").date() for x in (train_start, train_end))
    for f in sorted(DAILY.glob("*/*.npz")):
        try:
            d = datetime.strptime(f.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if not (ts <= d <= te):
            continue
        doy = min(d.timetuple().tm_yday, 365)          # fold leap day into 365
        sic = np.load(f)["data"][0]
        s, n = acc.get(doy, (None, 0))
        acc[doy] = (sic.astype(np.float64) if s is None else s + sic, n + 1)
    clim = {k: (s / n).astype(np.float32) for k, (s, n) in acc.items()}
    logger.info("Climatology covers %d days-of-year", len(clim))
    return clim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--input-window", type=int, default=7)
    ap.add_argument("--checkpoint", default=None, help="optional trained model .pt to include")
    ap.add_argument("--device", default=None)
    ap.add_argument("--encoder-channels", type=int, nargs="+", default=[16,32,64,128])
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None, help="cap samples per horizon (quick run)")
    args = ap.parse_args()

    cfg = load_cfg()
    mask = np.load(MASK)
    OUT.mkdir(parents=True, exist_ok=True)

    clim = build_climatology(cfg["train_start"], cfg["train_end"], mask)

    model = None
    if args.checkpoint:
        import torch
        from seaice_forecast.models.unet_convlstm import UNetConvLSTM
        dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        model = UNetConvLSTM(in_channels=7, output_channels=1, seq_len=args.input_window,
                             encoder_channels=args.encoder_channels, convlstm_layers=1,
                             dropout=0.0, residual=args.residual, output_activation="sigmoid")
        ck = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        model.to(dev).eval()
        logger.info("Loaded model from %s (epoch %s)", args.checkpoint, ck.get("epoch"))

    results = {}
    for h in args.horizons:
        # The model was trained on z-scored inputs, so evaluation MUST apply the
        # same normalisation. Feeding raw Kelvin to a model trained on normalised
        # inputs produces garbage predictions and a meaningless skill score.
        _ns = json.loads((ROOT / "data/processed/normalization_stats.json").read_text())
        ds = RealSeaIceDataset(
            daily_data_dir=DAILY, start_date=cfg["test_start"], end_date=cfg["test_end"],
            input_window=args.input_window, forecast_horizon=h,
            channel_concat=False, phase="phase2", mask_path=MASK,
            normalization_stats=_ns,
        )
        n = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
        if n == 0:
            logger.warning("horizon +%dd: no samples", h); continue

        preds = {"persistence": [], "climatology": []}
        if model is not None:
            preds["model"] = []
        targets = []

        for i in range(n):
            x, y = ds[i]                              # x [T,C,H,W]  y [1,H,W]
            xs = x.numpy()
            targets.append(y.numpy()[0])
            preds["persistence"].append(xs[-1, 0])    # last input day's SIC

            _, tgt_file, _ = ds.samples[i]
            d = datetime.strptime(Path(tgt_file).stem, "%Y%m%d").date()
            doy = min(d.timetuple().tm_yday, 365)
            preds["climatology"].append(clim.get(doy, np.zeros_like(xs[-1, 0])))

            if model is not None:
                import torch
                with torch.no_grad():
                    out = model(x.unsqueeze(0).to(next(model.parameters()).device))
                preds["model"].append(out.squeeze(0).squeeze(0).cpu().numpy())

        tgt = np.stack(targets)

        # Two masks, deliberately:
        #  full  = every ocean cell. ~83% of the Southern Ocean domain is
        #          permanently ice-free, is 0.0 in truth AND in every forecast,
        #          and so contributes zero error — this dilutes MAE ~6x and
        #          flatters every model. Reported only for comparability.
        #  ice   = the active ice zone (cells that reach >15% SIC anywhere in
        #          the evaluation period). This is the honest metric and the
        #          one to quote.
        ice_zone = ((tgt > 0.15).any(axis=0)) & (mask == 1)
        ice_mask = np.where(ice_zone, 1, 0)

        row = {"n_samples": n,
               "ice_zone_cells": int(ice_zone.sum()),
               "ocean_cells": int((mask == 1).sum())}
        for k, v in preds.items():
            pr = np.stack(v)
            row[k] = {
                "mae_full": masked_mae(pr, tgt, mask),
                "rmse_full": masked_rmse(pr, tgt, mask),
                "mae_ice": masked_mae(pr, tgt, ice_mask),
                "rmse_ice": masked_rmse(pr, tgt, ice_mask),
            }
        results[f"+{h}d"] = row
        logger.info("horizon +%dd (n=%d, ice cells=%d): %s", h, n, int(ice_zone.sum()),
                    "  ".join(f"{k} MAE_ice={row[k]['mae_ice']:.4f}" for k in preds))

    # report
    print("\n" + "=" * 68)
    print("SKILL VS BASELINES — test split %s .. %s" % (cfg["test_start"], cfg["test_end"]))
    print("=" * 68)
    cols = ["persistence", "climatology"] + (["model"] if model is not None else [])
    print("\nMAE over ACTIVE ICE ZONE  (the metric to quote)")
    print(f"{'lead':<7}{'n':>6}" + "".join(f"{c:>16}" for c in cols))
    for lead, row in results.items():
        line = f"{lead:<7}{row['n_samples']:>6}"
        for c in cols:
            line += f"{row[c]['mae_ice']:>16.4f}"
        print(line)

    print("\nMAE over FULL OCEAN MASK  (diluted ~6x by permanently ice-free cells)")
    print(f"{'lead':<7}{'n':>6}" + "".join(f"{c:>16}" for c in cols))
    for lead, row in results.items():
        line = f"{lead:<7}{row['n_samples']:>6}"
        for c in cols:
            line += f"{row[c]['mae_full']:>16.4f}"
        print(line)

    if model is not None:
        print("\nSkill score vs persistence, active ice zone")
        print("  (1 - MAE_model/MAE_persistence; > 0 means the model beats persistence)")
        for lead, row in results.items():
            ss = 1 - row["model"]["mae_ice"] / row["persistence"]["mae_ice"]
            print(f"  {lead:<6} {ss:+.3f}")
    print("=" * 68 + "\n")

    p = OUT / "baseline_comparison.json"
    p.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", p)


if __name__ == "__main__":
    sys.exit(main())
