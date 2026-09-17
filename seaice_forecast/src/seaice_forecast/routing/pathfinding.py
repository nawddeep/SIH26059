"""
Phase F: Route Optimization and Navigational Pathfinding Module.

Implements risk-aware route planning over Antarctic Polar Stereographic grids:
- A* and Dijkstra pathfinding with 8-connected and 4-connected topology.
- Land avoidance via binary land mask (land cells set to np.inf).
- Cost function integrating step distance and IMO POLARIS navigational risk.
- Straight-line baseline comparison and path risk profiling.
- Scalable from real single-day slices to multi-horizon spatio-temporal forecasts.
"""

import heapq
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from seaice_forecast.risk.polaris import PolarisRiskConfig, risk_ice_array


def load_land_mask(path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Load or generate a boolean land mask where True = land (impassable) and False = ocean.

    Args:
        path: Path to .npy file (where 1=ocean, 0=land or True=land).

    Returns:
        2D boolean array of shape [H, W], True = land.
    """
    if path is not None and Path(path).exists():
        raw = np.load(path)
        # Check standard convention in this repo: 1 = ocean, 0 = land
        if raw.dtype in [np.uint8, np.int32, np.int64, np.float32, np.float64]:
            # If values are 0 and 1, 0 is land
            if ((raw == 0) | (raw == 1)).all():
                return raw == 0
        return raw.astype(bool)

    # Default fallback path if exists
    default_mask = Path("data/processed/land_ocean_mask_ps25.npy")
    if default_mask.exists():
        raw = np.load(default_mask)
        return raw == 0

    # Synthetic placeholder (332 x 316) with central continent circular land mass
    H, W = 332, 316
    yy, xx = np.ogrid[:H, :W]
    center_y, center_x = 166, 158
    radius = 70
    land = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2
    return land


def build_cost_grid(
    sic_grid: np.ndarray,
    polar_class: Union[int, str] = "PC4",
    distance_weight: float = 1.0,
    risk_weight: float = 1.0,
    fuel_weight: float = 0.0,
    land_mask: Optional[np.ndarray] = None,
    pixel_size_km: float = 25.0,
    config: Optional[Union[dict, PolarisRiskConfig]] = None,
    fuel_config=None,
) -> np.ndarray:
    """
    Build 2D cost grid combining base traversal distance, POLARIS navigational
    risk, and ice-aware fuel burn.

    Cost formulation:
        C(r, c) = distance_weight * base_cost
                + risk_weight     * 10.0 * risk_cost(r, c)
                + fuel_weight     * fuel_cost(r, c)

    Land cells and invalid/NaN cells are set to np.inf.

    The fuel term is divided by the open-water burn for the same cell and
    vessel, so it is ~1.0 in open water and ~4.3 in heavy ice regardless of ship
    size. Raw tonnes would make fuel_weight dominate or vanish depending on the
    vessel, which would stop the three weights meaning the same thing across
    ships. The reference is recomputed from fuel_config and pixel_size_km, so
    overriding base_rate_t_per_km or v_ref_kn rescales it correctly.

    Args:
        sic_grid: 2D array of Sea-Ice Concentration in [0, 1].
        polar_class: Vessel Polar Class (e.g., 'PC1' - 'PC7' or 'UNCLASSED').
        distance_weight: Scaling weight for distance / step penalty.
        risk_weight: Scaling weight for navigational ice risk.
        fuel_weight: Scaling weight for ice-aware fuel burn. Defaults to 0.0,
            which reproduces the pre-fuel cost grid bitwise.
        land_mask: Boolean array where True = land (impassable).
        pixel_size_km: Grid cell resolution in km (default 25.0 km).
        config: Optional PolarisRiskConfig configuration.
        fuel_config: Optional FuelConfig for the fuel term.

    Returns:
        2D float array of traversal costs.
    """
    sic = np.asarray(sic_grid, dtype=np.float64)
    H, W = sic.shape

    # Handle NaNs
    nan_mask = np.isnan(sic)
    sic_clean = np.where(nan_mask, 0.0, sic)

    # Compute risk grid using Phase E IMO POLARIS function
    risk_grid = risk_ice_array(sic_clean, polar_class=polar_class, config=config)

    # Base cell traversal cost (normalized or physical)
    base_dist = 1.0

    cost_grid = (distance_weight * base_dist) + (risk_weight * 10.0 * risk_grid)

    if fuel_weight:
        from seaice_forecast.fuel import fuel_per_cell, fuel_per_cell_array

        fuel_grid = fuel_per_cell_array(
            pixel_size_km, sic_clean, polar_class=polar_class, config=fuel_config
        )
        fuel_reference = fuel_per_cell(
            pixel_size_km, 0.0, polar_class=polar_class, config=fuel_config
        )
        cost_grid = cost_grid + fuel_weight * (fuel_grid / fuel_reference)

    # Apply land mask: impassable barrier
    if land_mask is not None:
        cost_grid[land_mask] = np.inf

    # Also set NaNs to impassable
    cost_grid[nan_mask] = np.inf

    return cost_grid


def straight_line_path(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Compute straight-line discrete path between start and goal using Bresenham's algorithm.

    Args:
        start: (row, col)
        goal: (row, col)

    Returns:
        List of (row, col) tuples along the direct line.
    """
    r0, c0 = start
    r1, c1 = goal
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    path = []
    r, c = r0, c0
    while True:
        path.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return path


def astar_path(
    cost_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: str = "euclidean",
    connectivity: int = 8,
) -> List[Tuple[int, int]]:
    """
    Compute optimal least-cost path using A* search.

    Args:
        cost_grid: 2D array of positive costs (np.inf for obstacles/land).
        start: (row, col) start coordinates.
        goal: (row, col) goal coordinates.
        heuristic: 'euclidean', 'manhattan', or 'zero' (Dijkstra).
        connectivity: 4 or 8 neighbor connectivity.

    Returns:
        List of (row, col) tuples from start to goal, or empty list if no path exists.
    """
    H, W = cost_grid.shape
    sr, sc = start
    gr, gc = goal

    if cost_grid[sr, sc] == np.inf or cost_grid[gr, gc] == np.inf:
        return []

    if start == goal:
        return [start]

    # Neighbor offsets and step distances
    if connectivity == 8:
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, np.sqrt(2)),
            (-1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2)),
            (1, 1, np.sqrt(2)),
        ]
    else:
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
        ]

    # Heuristic function
    if heuristic == "euclidean":
        def h_func(r: int, c: int) -> float:
            return float(np.sqrt((r - gr) ** 2 + (c - gc) ** 2))
    elif heuristic == "manhattan":
        def h_func(r: int, c: int) -> float:
            return float(abs(r - gr) + abs(c - gc))
    else:
        def h_func(r: int, c: int) -> float:
            return 0.0

    # Priority queue: (f_score, tie_breaker, node)
    tie_breaker = itertools.count()
    open_heap = []
    heapq.heappush(open_heap, (h_func(sr, sc), next(tie_breaker), (sr, sc)))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed_set = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        cr, cc = current

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed_set:
            continue
        closed_set.add(current)

        curr_g = g_score[current]

        for dr, dc, dist_mult in neighbors:
            nr, nc = cr + dr, cc + dc

            if not (0 <= nr < H and 0 <= nc < W):
                continue

            neighbor_cost = cost_grid[nr, nc]
            if neighbor_cost == np.inf:
                continue

            # Traversal edge cost considers diagonal factor
            edge_cost = neighbor_cost * dist_mult
            tentative_g = curr_g + edge_cost

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + h_func(nr, nc)
                came_from[(nr, nc)] = current
                heapq.heappush(open_heap, (f_score, next(tie_breaker), (nr, nc)))

    return []


def dijkstra_path(
    cost_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 8,
) -> List[Tuple[int, int]]:
    """
    Fallback Dijkstra shortest path implementation.
    """
    return astar_path(cost_grid, start, goal, heuristic="zero", connectivity=connectivity)


def path_distance(path: List[Tuple[int, int]], pixel_size_km: float = 25.0) -> float:
    """
    Compute total physical distance along path in km (or grid cells if pixel_size_km=1.0).
    Properly accounts for diagonal steps (sqrt(2) * step).
    """
    if len(path) < 2:
        return 0.0

    total_dist = 0.0
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        step_len = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        total_dist += step_len

    return float(total_dist * pixel_size_km)


def path_cost(path: List[Tuple[int, int]], cost_grid: np.ndarray) -> float:
    """
    Compute total integrated cost along the path.
    """
    if len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        dist_mult = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        total += float(cost_grid[r1, c1] * dist_mult)

    return total


def path_risk_summary(
    path: List[Tuple[int, int]],
    risk_grid: np.ndarray,
    high_risk_threshold: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """
    Summarize navigational risk profile along a path.
    """
    if not path:
        return {"mean_risk": 0.0, "max_risk": 0.0, "high_risk_cells": 0}

    risks = [float(risk_grid[r, c]) for r, c in path if np.isfinite(risk_grid[r, c])]
    if not risks:
        return {"mean_risk": 0.0, "max_risk": 0.0, "high_risk_cells": 0}

    high_risk_count = sum(1 for r in risks if r > high_risk_threshold)

    return {
        "mean_risk": float(np.mean(risks)),
        "max_risk": float(np.max(risks)),
        "high_risk_cells": int(high_risk_count),
    }
