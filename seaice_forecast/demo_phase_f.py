#!/usr/bin/env python3
"""
Phase F Demonstration Script: Risk-Aware Route Optimization over Real Antarctic Sea-Ice Data.

Executes:
1. Loads real Antarctic Sea-Ice Concentration (SIC) slice and land mask.
2. Computes IMO POLARIS navigational risk surface across the 332x316 grid.
3. Constructs unified distance-risk traversal cost surface (land cells = inf).
4. Executes 8-connected A* optimal pathfinding between maritime waypoints.
5. Computes straight-line reference baseline.
6. Generates high-resolution visualization comparing A* route against straight-line baseline.
7. Exports numerical summary table and cost surface artifacts.
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from seaice_forecast.routing.pathfinding import (
    load_land_mask,
    build_cost_grid,
    straight_line_path,
    astar_path,
    path_distance,
    path_cost,
    path_risk_summary,
)
from seaice_forecast.risk.polaris import risk_ice_array

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("demo_phase_f")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase F Route Optimization Demonstration.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase_f.yaml",
        help="Path to Phase F YAML configuration file.",
    )
    parser.add_argument(
        "--sic",
        type=str,
        default=None,
        help="Path to real SIC data (.npz or .npy). Overrides config data path.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to land/ocean mask (.npy). Overrides config mask path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Overrides config output directory.",
    )
    parser.add_argument(
        "--risk-weight",
        type=float,
        default=None,
        help="Risk weighting factor. Overrides config risk_weight.",
    )
    parser.add_argument(
        "--polar-class",
        type=str,
        default=None,
        help="Vessel Polar Class (e.g. 'PC1', 'PC4', 'PC7').",
    )
    return parser.parse_args()


def load_sic_slice(data_path: Path) -> Tuple[np.ndarray, bool]:
    """
    Load real SIC slice. Returns (sic_grid, is_real).
    """
    if data_path.exists():
        logger.info(f"Loading real SIC slice from: {data_path}")
        raw = np.load(data_path)
        if "data" in raw:
            # 7-channel array: channel 0 is SIC
            arr = raw["data"]
            if arr.ndim == 3:
                return arr[0].astype(np.float64), True
            return arr.astype(np.float64), True
        elif "regridded" in raw:
            return raw["regridded"][0].astype(np.float64), True
        elif "sic" in raw:
            return raw["sic"].astype(np.float64), True
        else:
            first_key = list(raw.keys())[0]
            arr = raw[first_key]
            if arr.ndim == 3:
                return arr[0].astype(np.float64), True
            return arr.astype(np.float64), True

    logger.warning(f"Data file not found at {data_path}. Generating synthetic test grid.")
    H, W = 332, 316
    yy, xx = np.ogrid[:H, :W]
    # Tongue of synthetic pack ice
    sic_syn = np.clip(0.9 * np.exp(-((yy - 50)**2 + (xx - 120)**2) / (2 * 25**2)), 0.0, 1.0)
    return sic_syn, False


def main():
    args = parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"data": {}, "routing": {}, "output": {}}

    # Override parameters with CLI if supplied
    sic_path = Path(args.sic or cfg["data"].get("sic_data_path", "data/processed/regridded/daily/2021/20210628.npz"))
    mask_path = Path(args.mask or cfg["data"].get("mask_path", "data/processed/land_ocean_mask_ps25.npy"))
    output_dir = Path(args.output or cfg["output"].get("dir", "output/phase_f"))
    output_dir.mkdir(parents=True, exist_ok=True)

    polar_class = args.polar_class or cfg["routing"].get("polar_class", "PC4")
    distance_weight = float(cfg["routing"].get("distance_weight", 1.0))
    risk_weight = float(args.risk_weight if args.risk_weight is not None else cfg["routing"].get("risk_weight", 2.0))
    pixel_size_km = float(cfg["data"].get("pixel_size_km", 25.0))
    start = tuple(cfg["routing"].get("start_coords", [65, 80]))
    goal = tuple(cfg["routing"].get("goal_coords", [30, 160]))

    logger.info("=" * 75)
    logger.info("PHASE F: ROUTE OPTIMIZATION & RISK-AWARE PATHFINDING DEMO")
    logger.info("=" * 75)

    # 1. Load real data and mask
    sic_grid, is_real_data = load_sic_slice(sic_path)
    land_mask = load_land_mask(mask_path)
    logger.info(f"Loaded SIC grid: {sic_grid.shape} (Range: [{sic_grid.min():.2f}, {sic_grid.max():.2f}])")
    logger.info(f"Loaded land mask: {land_mask.shape} (Land cells: {int(land_mask.sum())})")

    # 2. Compute risk grid and cost surface
    risk_grid = risk_ice_array(sic_grid, polar_class=polar_class)
    cost_grid = build_cost_grid(
        sic_grid=sic_grid,
        polar_class=polar_class,
        distance_weight=distance_weight,
        risk_weight=risk_weight,
        land_mask=land_mask,
        pixel_size_km=pixel_size_km,
    )
    logger.info(f"Constructed unified cost grid (Distance Weight={distance_weight}, Risk Weight={risk_weight})")

    # 3. Save cost grid artifacts
    npz_path = output_dir / "phase_f_cost_grid.npz"
    np.savez_compressed(
        npz_path,
        sic=sic_grid,
        risk=risk_grid,
        cost=cost_grid,
        land_mask=land_mask,
    )
    logger.info(f"✓ Saved cost grid archive: {npz_path}")

    # 4. Compute A* path
    logger.info(f"Computing A* path from {start} to {goal}...")
    astar_route = astar_path(cost_grid, start, goal, heuristic="euclidean", connectivity=8)
    if not astar_route:
        logger.error("No valid ocean path found between start and goal!")
        sys.exit(1)
    logger.info(f"✓ A* path found with {len(astar_route)} steps.")

    # 5. Compute straight-line reference baseline
    straight_route = straight_line_path(start, goal)

    # Verify land collision
    astar_hits_land = any(land_mask[r, c] for r, c in astar_route)
    straight_hits_land = any(land_mask[r, c] for r, c in straight_route)
    assert not astar_hits_land, "CRITICAL ERROR: A* route intersected land!"

    # 6. Metrics & Comparison Table
    astar_dist_km = path_distance(astar_route, pixel_size_km=pixel_size_km)
    straight_dist_km = path_distance(straight_route, pixel_size_km=pixel_size_km)

    astar_total_cost = path_cost(astar_route, cost_grid)
    straight_total_cost = path_cost(straight_route, cost_grid)

    astar_risk = path_risk_summary(astar_route, risk_grid, high_risk_threshold=0.5)
    straight_risk = path_risk_summary(straight_route, risk_grid, high_risk_threshold=0.5)

    comparison_data = [
        {
            "Metric": "Total Distance (km)",
            "A* Risk-Aware Route": f"{astar_dist_km:.1f} km",
            "Straight-Line Baseline": f"{straight_dist_km:.1f} km",
            "Difference": f"{(astar_dist_km - straight_dist_km):+.1f} km",
        },
        {
            "Metric": "Total Traversal Cost",
            "A* Risk-Aware Route": f"{astar_total_cost:.1f}",
            "Straight-Line Baseline": f"{straight_total_cost:.1f}" if not straight_hits_land else "INF (Hits Land)",
            "Difference": f"{(astar_total_cost - straight_total_cost):+.1f}" if not straight_hits_land else "N/A",
        },
        {
            "Metric": "Mean Navigational Risk",
            "A* Risk-Aware Route": f"{astar_risk['mean_risk']:.4f}",
            "Straight-Line Baseline": f"{straight_risk['mean_risk']:.4f}",
            "Difference": f"{(astar_risk['mean_risk'] - straight_risk['mean_risk']):+.4f}",
        },
        {
            "Metric": "Max Navigational Risk",
            "A* Risk-Aware Route": f"{astar_risk['max_risk']:.4f}",
            "Straight-Line Baseline": f"{straight_risk['max_risk']:.4f}",
            "Difference": f"{(astar_risk['max_risk'] - straight_risk['max_risk']):+.4f}",
        },
        {
            "Metric": "High-Risk Cells (Risk > 0.5)",
            "A* Risk-Aware Route": f"{astar_risk['high_risk_cells']} cells",
            "Straight-Line Baseline": f"{straight_risk['high_risk_cells']} cells",
            "Difference": f"{astar_risk['high_risk_cells'] - straight_risk['high_risk_cells']:+d} cells",
        },
        {
            "Metric": "Land Collisions",
            "A* Risk-Aware Route": "0 (Zero land intrusion)",
            "Straight-Line Baseline": f"{sum(1 for r, c in straight_route if land_mask[r, c])} cells",
            "Difference": "Valid marine passage",
        },
    ]

    df_comp = pd.DataFrame(comparison_data)
    logger.info("\n" + "=" * 75)
    logger.info(f"ROUTE PERFORMANCE COMPARISON TABLE (Polar Class: {polar_class})")
    logger.info("=" * 75)
    print(df_comp.to_string(index=False))
    logger.info("=" * 75)

    # Save summary JSON
    summary_json = {
        "status": "EVALUATED_ON_REAL_DATA" if is_real_data else "SMOKE_TEST_ONLY",
        "polar_class": polar_class,
        "weights": {"distance_weight": distance_weight, "risk_weight": risk_weight},
        "start": start,
        "goal": goal,
        "comparison": comparison_data,
    }
    json_path = output_dir / "phase_f_route_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info(f"✓ Saved comparison summary: {json_path}")

    # 7. Visualization
    logger.info("Generating route optimization map visualization...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display Risk / Navigational Heatmap
    # Create masked array so land is distinct
    display_grid = np.ma.masked_array(risk_grid, mask=land_mask)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#2b2b2b")  # Dark grey for land

    im = ax.imshow(display_grid, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("IMO POLARIS Navigational Risk $R_{ice} \in [0, 1]$", fontsize=11, fontweight="bold")

    # Overlay straight-line baseline (dashed cyan/blue)
    st_r, st_c = zip(*straight_route)
    ax.plot(st_c, st_r, linestyle="--", color="#00ffff", linewidth=2.0, alpha=0.85, label="Straight-Line Baseline")

    # Overlay A* route (vibrant red/magenta line)
    as_r, as_c = zip(*astar_route)
    ax.plot(as_c, as_r, linestyle="-", color="#ff0000", linewidth=3.0, label=f"A* Risk-Aware Route (PC: {polar_class})")

    # Mark Start and Goal
    ax.scatter(start[1], start[0], marker="o", color="#00ff00", s=140, edgecolors="black", zorder=10, label=f"Start: {start}")
    ax.scatter(goal[1], goal[0], marker="*", color="#ffff00", s=220, edgecolors="black", zorder=10, label=f"Goal: {goal}")

    title_text = "Antarctic Maritime Route Optimization: Risk-Aware A* Navigation"
    if not is_real_data:
        title_text = "[SMOKE TEST ONLY] " + title_text
    ax.set_title(title_text, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Polar Stereographic X Grid (25 km resolution)", fontsize=11)
    ax.set_ylabel("Polar Stereographic Y Grid (25 km resolution)", fontsize=11)

    # Add custom legend patch for land
    land_patch = mpatches.Patch(color="#2b2b2b", label="Antarctic Land (Impassable)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(land_patch)
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    # Focus view around route corridor with padding
    min_r = max(0, min(min(as_r), min(st_r)) - 25)
    max_r = min(332, max(max(as_r), max(st_r)) + 30)
    min_c = max(0, min(min(as_c), min(st_c)) - 25)
    max_c = min(316, max(max(as_c), max(st_c)) + 30)
    ax.set_ylim(max_r, min_r)
    ax.set_xlim(min_c, max_c)

    plt.tight_layout()

    route_png = output_dir / "phase_f_route.png"
    route_pdf = output_dir / "phase_f_route.pdf"
    fig.savefig(route_png, dpi=300, bbox_inches="tight")
    fig.savefig(route_pdf, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved Route Visualization PNG: {route_png}")
    logger.info(f"✓ Saved Route Visualization PDF: {route_pdf}")

    logger.info("\n" + "=" * 75)
    logger.info("PHASE F DEMONSTRATION EXECUTED SUCCESSFULLY (EXIT 0).")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
