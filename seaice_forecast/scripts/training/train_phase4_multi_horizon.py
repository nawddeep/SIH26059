#!/usr/bin/env python3
"""
Phase D: Train Multi-Horizon (+1d, +3d, +5d, +7d) Sea-Ice Forecasting Model.

Supports:
- Direct multi-head forecasting (dedicated spatial decoder heads per horizon)
- Recursive autoregressive rollout from single-step ConvLSTM
- Per-horizon weighted loss on ocean cells
- Gradient clipping (1.0) and Cosine Annealing learning rate schedule
- Atomic checkpointing with resume capability
- Evaluation with persistence baseline comparison
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.models.multi_horizon import (
    MultiHorizonForecaster,
    MultiHorizonLoss,
    MultiHorizonTrainer
)
from seaice_forecast.data_processing.dataset_real import MultiHorizonSeaIceDataset
from seaice_forecast.evaluation.multi_horizon_eval import evaluate_multi_horizon
from seaice_forecast.visualization.multi_horizon_plots import plot_multi_horizon_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_phase4_multi_horizon")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Phase D Multi-Horizon Sea-Ice Forecasting Model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase_d_full.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "recursive"],
        default=None,
        help="Override forecasting mode (direct or recursive)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('mps', 'cuda', or 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    mode = args.mode or cfg["model"].get("mode", "direct")
    epochs = args.epochs or cfg["training"].get("epochs", 50)
    horizons = cfg["data"].get("horizons", [1, 3, 5, 7])
    input_window = cfg["data"].get("input_window", 7)
    batch_size = cfg["training"].get("batch_size", 2)
    lr = float(cfg["training"].get("learning_rate", 1e-3))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    loss_weights = {int(k): float(v) for k, v in cfg["training"].get("loss_weights", {}).items()}

    # Land mask
    mask_path = Path(cfg["data"]["mask_path"])
    mask = np.load(mask_path) if mask_path.exists() else None

    # Datasets
    daily_dir = cfg["data"]["daily_dir"]
    phase = cfg["data"].get("phase", "phase2")

    logger.info(f"Building MultiHorizon datasets (window={input_window}, horizons={horizons}, phase={phase})...")
    train_dataset = MultiHorizonSeaIceDataset(
        daily_dir=daily_dir,
        split="train",
        input_window=input_window,
        horizons=horizons,
        phase=phase,
        mask_path=str(mask_path) if mask_path.exists() else None
    )
    val_dataset = MultiHorizonSeaIceDataset(
        daily_dir=daily_dir,
        split="val",
        input_window=input_window,
        horizons=horizons,
        phase=phase,
        mask_path=str(mask_path) if mask_path.exists() else None
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    # Initialize model
    logger.info(f"Initializing MultiHorizonForecaster (mode={mode})...")
    forecaster = MultiHorizonForecaster(
        mode=mode,
        horizons=horizons,
        in_channels=cfg["model"].get("in_channels", 7),
        seq_len=input_window,
        encoder_channels=cfg["model"].get("encoder_channels", [32, 64, 128, 256]),
        convlstm_layers=cfg["model"].get("convlstm_layers", 1),
        dropout=cfg["model"].get("dropout", 0.1),
        checkpoint_path=cfg["model"].get("checkpoint_path", None)
    )

    output_dir = Path(cfg["output"].get("dir", "output/phase_d"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MultiHorizonTrainer(
        forecaster=forecaster,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        horizons=horizons,
        horizon_weights=loss_weights,
        mask=mask,
        output_dir=output_dir / "checkpoints"
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(f"Starting Multi-Horizon training for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    # Final evaluation on val/test set
    eval_loader = val_loader or train_loader
    eval_results = evaluate_multi_horizon(
        forecaster=forecaster,
        dataloader=eval_loader,
        device=device,
        mask=mask,
        horizons=horizons,
        pixel_size_km=cfg["data"].get("pixel_size_km", 25.0),
        output_csv_path=cfg["output"].get("metrics_csv"),
        output_json_path=cfg["output"].get("metrics_json"),
        is_smoke_test=cfg["experiment"].get("is_smoke_test", False)
    )

    # Plot metrics vs lead time
    plots_dir = Path(cfg["output"].get("plots_dir", output_dir / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_prefix = "SMOKE_TEST_ONLY_" if cfg["experiment"].get("is_smoke_test", False) else ""
    plot_multi_horizon_metrics(
        metrics_df=eval_results["metrics_df"],
        save_path_png=str(plots_dir / f"{plot_prefix}error_vs_lead_time.png"),
        save_path_pdf=str(plots_dir / f"{plot_prefix}error_vs_lead_time.pdf"),
        is_smoke_test=cfg["experiment"].get("is_smoke_test", False)
    )

    logger.info("Multi-horizon training and evaluation pipeline complete.")


if __name__ == "__main__":
    main()
