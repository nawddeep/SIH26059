"""
Phase F Unit Tests: Route Optimization & Navigational Pathfinding.

Tests:
1. A* shortest path on small synthetic grid with known obstacle.
2. Strict land avoidance (land cells with cost=inf are never traversed).
3. Risk-weight sensitivity: increasing risk_weight actively routes around high-risk cells.
4. Path distance and cost computations with diagonal sqrt(2) scaling.
5. Straight-line baseline path generation.
6. NaN and invalid value robustness in build_cost_grid.
7. Equivalence of Dijkstra and A* path costs.
8. Handling disconnected start/goal (enclosed by land barrier).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.routing.pathfinding import (
    load_land_mask,
    build_cost_grid,
    straight_line_path,
    astar_path,
    dijkstra_path,
    path_distance,
    path_cost,
    path_risk_summary,
)


class TestPathfindingAlgorithms:
    def test_astar_simple_corridor(self):
        # 5x5 grid with uniform cost 1.0
        grid = np.ones((5, 5), dtype=np.float64)
        start = (0, 0)
        goal = (4, 4)

        path = astar_path(grid, start, goal, heuristic="euclidean", connectivity=8)
        assert len(path) == 5  # Diagonal steps: (0,0)->(1,1)->(2,2)->(3,3)->(4,4)
        assert path[0] == start and path[-1] == goal

    def test_land_barrier_avoidance(self):
        # 7x7 grid with a vertical land wall at col 3, with gap at row 6
        grid = np.ones((7, 7), dtype=np.float64)
        # Wall across col 3 from row 0 to 5
        grid[0:6, 3] = np.inf

        start = (2, 1)
        goal = (2, 5)

        path = astar_path(grid, start, goal, heuristic="euclidean", connectivity=8)
        assert len(path) > 0

        # Assert no cell on the path is in the land wall
        for r, c in path:
            assert grid[r, c] != np.inf, f"Path traversed land cell ({r}, {c})!"

        # Path must pass through row 6
        rows = [r for r, c in path]
        assert max(rows) >= 6

    def test_risk_weight_aversion(self):
        # 10x10 grid:
        # Open water cost = 1.0 everywhere
        # Central patch (rows 3-7, cols 3-7) has severe pack ice
        sic_grid = np.zeros((10, 10), dtype=np.float64)
        sic_grid[3:7, 3:7] = 0.95

        start = (5, 1)
        goal = (5, 8)

        # 1. Distance-only routing (risk_weight = 0.0)
        cost_dist_only = build_cost_grid(
            sic_grid=sic_grid,
            polar_class="PC7",
            distance_weight=1.0,
            risk_weight=0.0,
        )
        path_direct = astar_path(cost_dist_only, start, goal, connectivity=8)

        # 2. Risk-averse routing (risk_weight = 10.0)
        cost_risk_averse = build_cost_grid(
            sic_grid=sic_grid,
            polar_class="PC7",
            distance_weight=1.0,
            risk_weight=10.0,
        )
        path_averse = astar_path(cost_risk_averse, start, goal, connectivity=8)

        assert len(path_direct) > 0
        assert len(path_averse) > 0

        # Direct path cuts through the center (rows 4 or 5)
        direct_rows = [r for r, c in path_direct if 3 <= c <= 6]
        assert any(r in [4, 5] for r in direct_rows)

        # Risk-averse path detours north or south (around rows 3-6)
        averse_rows = [r for r, c in path_averse if 3 <= c <= 6]
        assert all(r < 3 or r > 6 for r in averse_rows), "Risk-averse path failed to detour around pack ice!"

    def test_dijkstra_equivalence(self):
        # Random non-negative cost grid
        np.random.seed(42)
        grid = np.random.uniform(1.0, 5.0, size=(10, 10))
        grid[2:4, 4:6] = np.inf
        start = (0, 0)
        goal = (8, 8)

        p_astar = astar_path(grid, start, goal, heuristic="euclidean")
        p_dijkstra = dijkstra_path(grid, start, goal)

        cost_astar = path_cost(p_astar, grid)
        cost_dijkstra = path_cost(p_dijkstra, grid)

        assert pytest.approx(cost_astar, rel=1e-4) == cost_dijkstra

    def test_disconnected_no_path(self):
        # Goal completely surrounded by land
        grid = np.ones((7, 7), dtype=np.float64)
        grid[4:7, 4:7] = np.inf
        grid[5, 5] = 1.0  # isolated island inside land box
        path = astar_path(grid, (0, 0), (5, 5))
        assert path == []


class TestCostAndMetrics:
    def test_path_distance_calculations(self):
        # 1 horizontal step (1.0), 1 diagonal step (sqrt(2))
        path = [(0, 0), (0, 1), (1, 2)]
        dist_km = path_distance(path, pixel_size_km=25.0)
        expected_km = (1.0 + np.sqrt(2)) * 25.0
        assert pytest.approx(dist_km, rel=1e-4) == expected_km

    def test_straight_line_path(self):
        line = straight_line_path((0, 0), (0, 4))
        assert line == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

        diag = straight_line_path((0, 0), (2, 2))
        assert diag == [(0, 0), (1, 1), (2, 2)]

    def test_build_cost_grid_with_nans(self):
        sic = np.array([[0.1, np.nan], [0.5, 0.8]])
        mask = np.array([[False, False], [True, False]])  # [1, 0] is land
        cost = build_cost_grid(sic, land_mask=mask)

        # Land cell must be inf
        assert np.isinf(cost[1, 0])
        # NaN cell must be inf
        assert np.isinf(cost[0, 1])
        # Ocean cells must be finite and positive
        assert np.isfinite(cost[0, 0]) and cost[0, 0] > 0
        assert np.isfinite(cost[1, 1]) and cost[1, 1] > cost[0, 0]

    def test_path_risk_summary(self):
        risk_grid = np.array([[0.0, 0.2], [0.6, 0.9]])
        path = [(0, 0), (0, 1), (1, 0), (1, 1)]
        summary = path_risk_summary(path, risk_grid, high_risk_threshold=0.5)

        assert pytest.approx(summary["mean_risk"], rel=1e-4) == (0.0 + 0.2 + 0.6 + 0.9) / 4.0
        assert summary["max_risk"] == 0.9
        assert summary["high_risk_cells"] == 2  # 0.6 and 0.9
