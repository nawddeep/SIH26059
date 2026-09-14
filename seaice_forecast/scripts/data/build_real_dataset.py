#!/usr/bin/env python3
"""
Build Phase 1 sliding-window dataset from real NSIDC SIC files.

The real NSIDC CDR v6 files live in:
    data/raw/real/nsidc/YYYY/MM/sic_pss25_YYYYMMDD_F17_v06r00.nc

They are already on the 316×332 polar-stereographic grid with SIC in [0, 1].

This script:
1. Discovers all real daily files (recursive glob).
2. Loads each file, extracts `cdr_seaice_conc`, replaces NaN with 0.
3. Creates sliding windows (7-day input → next-day target).
4. Splits chronologically into train / val / test.
5. Saves `data/processed/{split}_data.npz` and `{split}_metadata.json`.
6. Recomputes the land-ocean mask from the real files.
7. Recomputes normalization statistics from the training split.

Usage:
    # using the project venv (needs h5py / xarray):
    venv/bin/python scripts/data/build_real_dataset.py

    # or with --dry-run to just report file counts and date range:
    venv/bin/python scripts/data/build_real_dataset.py --dry-run
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default date ranges for 2024 data (adjust if you download more years)
DEFAULT_SPLITS = {
    "train": ("2024-01-01", "2024-09-30"),
    "val":   ("2024-10-01", "2024-11-30"),
    "test":  ("2024-12-01", "2024-12-31"),
}

INPUT_WINDOW = 7
FORECAST_HORIZON = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_files(raw_dir: Path):
    """Recursively find all NSIDC .nc files and return sorted list."""
    files = sorted(raw_dir.rglob("*.nc"))
    # Filter to files that look like daily SIC (not flag files etc.)
    files = [f for f in files if "sic_pss25" in f.name]
    return files


def parse_date_from_filename(filepath: Path) -> datetime:
    """Extract date from filename like sic_pss25_20240101_F17_v06r00.nc."""
    parts = filepath.name.split("_")
    # sic_pss25_YYYYMMDD_F17_v06r00.nc  → parts[2] = '20240101'
    date_str = parts[2]
    return datetime.strptime(date_str, "%Y%m%d")


def load_sic_field(filepath: Path) -> np.ndarray:
    """Load a single NSIDC file and return (SIC field, validity mask).

    SIC is float32 [H, W] in [0, 1].  Land pixels are NaN in the source;
    we keep them as NaN so the mask builder can distinguish land from ocean.
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)
    sic = ds["cdr_seaice_conc"].values.squeeze()  # already 0-1, NaN over land
    ds.close()

    sic = sic.astype(np.float32)
    # Clip finite values to valid range (leave NaN as-is)
    sic = np.where(np.isfinite(sic), np.clip(sic, 0.0, 1.0), np.nan)
    return sic


def create_land_ocean_mask_from_files(files, step=10):
    """
    Build a land-ocean mask from a sample of real SIC files.

    A pixel is 'ocean' (1) if it has valid (non-zero, non-NaN) SIC in
    at least 50% of the sample files.
    """
    sample_files = files[::step][:100]
    logger.info(f"Building land-ocean mask from {len(sample_files)} sample files...")

    # Land pixels are NaN in the source; ocean pixels are finite.
    # A pixel is ocean if it has finite (non-NaN) data in >= 50% of samples.
    valid_counts = np.zeros((332, 316), dtype=np.int32)
    total = 0

    for fp in sample_files:
        sic = load_sic_field(fp)
        valid = np.isfinite(sic)
        valid_counts += valid.astype(np.int32)
        total += 1

    threshold = total * 0.5
    mask = (valid_counts >= threshold).astype(np.uint8)

    ocean = int(mask.sum())
    land = mask.size - ocean
    logger.info(f"Mask: {ocean} ocean, {land} land  ({ocean / mask.size:.1%} ocean)")
    return mask


def build_windows(file_dates, sic_stack, split_slices):
    """
    Build sliding-window inputs/targets and split them.

    Parameters
    ----------
    file_dates : list[datetime]
        Sorted list of dates, one per daily file.
    sic_stack : np.ndarray [N_days, H, W]
        Daily SIC fields.
    split_slices : dict[str, (str, str)]
        Date ranges for each split.

    Returns
    -------
    dict with 'train', 'val', 'test' keys → (inputs, targets, date_pairs)
    """
    n_days = len(file_dates)
    window = INPUT_WINDOW + FORECAST_HORIZON  # 8

    # First, build ALL windows
    all_inputs = []
    all_targets = []
    all_date_pairs = []

    for i in range(n_days - window + 1):
        inp = sic_stack[i : i + INPUT_WINDOW]          # [7, H, W]
        tgt = sic_stack[i + INPUT_WINDOW : i + window]  # [1, H, W]
        all_inputs.append(inp)
        all_targets.append(tgt)
        all_date_pairs.append((
            file_dates[i + INPUT_WINDOW - 1].strftime("%Y-%m-%d"),
            file_dates[i + INPUT_WINDOW].strftime("%Y-%m-%d"),
        ))

    all_inputs = np.stack(all_inputs, axis=0).astype(np.float32)   # [N, 7, H, W]
    all_targets = np.stack(all_targets, axis=0).astype(np.float32)  # [N, 1, H, W]

    logger.info(f"Total windows: {len(all_inputs)}")

    # Split by target date
    results = {}
    for split_name, (start_str, end_str) in split_slices.items():
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")

        indices = []
        for i, (_, target_date_str) in enumerate(all_date_pairs):
            td = datetime.strptime(target_date_str, "%Y-%m-%d")
            if start_dt <= td <= end_dt:
                indices.append(i)

        if len(indices) == 0:
            logger.warning(f"No windows fall in {split_name} ({start_str} → {end_str})")
            results[split_name] = (
                np.zeros((0, INPUT_WINDOW, 332, 316), dtype=np.float32),
                np.zeros((0, FORECAST_HORIZON, 332, 316), dtype=np.float32),
                [],
            )
        else:
            results[split_name] = (
                all_inputs[indices],
                all_targets[indices],
                [all_date_pairs[i] for i in indices],
            )
            logger.info(
                f"  {split_name}: {len(indices)} samples  "
                f"({all_date_pairs[indices[0]][1]} → {all_date_pairs[indices[-1]][1]})"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build real-data sliding-window dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/real/nsidc",
        help="Root directory containing NSIDC .nc files (recursive)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Where to save processed .npz and metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just report file counts and date range; don't build anything",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Override train end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="Override val end date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    files = discover_files(raw_dir)
    if not files:
        logger.error(f"No NSIDC files found under {raw_dir}")
        sys.exit(1)

    logger.info(f"Found {len(files)} daily SIC files")
    first_date = parse_date_from_filename(files[0])
    last_date = parse_date_from_filename(files[-1])
    logger.info(f"Date range: {first_date.date()} → {last_date.date()}")

    if args.dry_run:
        logger.info("DRY RUN — exiting without building datasets.")
        return

    # Override split dates if requested
    splits = dict(DEFAULT_SPLITS)
    if args.train_end:
        splits["train"] = (splits["train"][0], args.train_end)
    if args.val_end:
        splits["val"] = (splits["val"][0], args.val_end)

    # Load all SIC fields
    logger.info("Loading all daily SIC fields into memory...")
    file_dates = []
    sic_list = []
    for fp in files:
        try:
            d = parse_date_from_filename(fp)
            sic = load_sic_field(fp)
            file_dates.append(d)
            sic_list.append(sic)
        except Exception as e:
            logger.warning(f"Skipping {fp.name}: {e}")

    sic_stack = np.stack(sic_list, axis=0)  # [N_days, H, W]
    logger.info(f"Loaded {sic_stack.shape[0]} daily fields, shape={sic_stack.shape}")

    # Replace NaN (land) with 0 for the sliding-window arrays.
    # The land-ocean mask already encodes which pixels are land.
    sic_stack = np.nan_to_num(sic_stack, nan=0.0)

    # Build land-ocean mask
    mask = create_land_ocean_mask_from_files(files)
    mask_path = output_dir / "land_ocean_mask.npy"
    np.save(mask_path, mask)
    logger.info(f"Mask saved to {mask_path}")

    # Build sliding windows and split
    results = build_windows(file_dates, sic_stack, splits)

    # Save each split
    for split_name, (inputs, targets, date_pairs) in results.items():
        if len(inputs) == 0:
            logger.warning(f"Skipping {split_name} — no samples")
            continue

        npz_path = output_dir / f"{split_name}_data.npz"
        np.savez_compressed(npz_path, inputs=inputs, targets=targets)
        logger.info(f"Saved {npz_path}: {inputs.shape}")

        metadata = {
            "split": split_name,
            "n_samples": len(inputs),
            "input_shape": list(inputs.shape),
            "target_shape": list(targets.shape),
            "date_pairs": date_pairs,
            "input_window": INPUT_WINDOW,
            "forecast_horizon": FORECAST_HORIZON,
            "synthetic": False,
            "source": "NSIDC CDR v6 real data",
            "date_range": f"{date_pairs[0][0]} to {date_pairs[-1][1]}",
        }
        meta_path = output_dir / f"{split_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved {meta_path}")

    # Compute normalization stats from training set
    train_inputs, _, _ = results["train"]
    if len(train_inputs) > 0:
        ocean_mask_bool = mask == 1
        ocean_data = train_inputs[:, :, ocean_mask_bool]
        stats = {
            "mean": float(np.mean(ocean_data)),
            "std": float(np.std(ocean_data)),
        }
        stats_path = output_dir / "normalization_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Normalization stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        logger.info(f"Saved to {stats_path}")

    logger.info("=" * 60)
    logger.info("BUILD COMPLETE — real-data dataset ready")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
