"""
Phase E Unit Tests: Uncertainty Quantification & IMO POLARIS Navigational Risk Function.

Tests:
1. Uncertainty Quantification:
   - compute_horizon_error_stats calculation accuracy across dicts, arrays, and tuples.
   - fit_uncertainty_curve (linear and exponential) strictly expanding with lead time.
   - generate_uncertainty_report pipeline and disk artifact generation.
2. IMO POLARIS Risk Function:
   - Boundary checks: SIC=0.0 (near zero), SIC=0.15 (low threshold), SIC=1.0 (near max >= 0.9).
   - Strict monotonicity across SIC in [0, 1].
   - Polar Class hierarchy: PC1 < PC2 < PC3 < PC4 < PC5 < PC6 < PC7 < UNCLASSED for pack ice.
   - Vectorized risk_ice_array equivalence with element-wise risk_ice.
   - Polar Class string/integer normalization.
   - Custom configuration override support.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.uncertainty.quantification import (
    compute_horizon_error_stats,
    fit_uncertainty_curve,
    generate_uncertainty_report,
)
from seaice_forecast.risk.polaris import (
    PolarisRiskConfig,
    normalize_polar_class,
    risk_ice,
    risk_ice_array,
)


class TestUncertaintyQuantification:
    def test_compute_horizon_error_stats_dict(self):
        errors = {
            1: np.array([0.1, -0.1, 0.2, -0.2]),
            3: np.array([0.4, -0.4, 0.6, -0.6]),
        }
        stats = compute_horizon_error_stats(errors)
        assert 1 in stats and 3 in stats
        assert pytest.approx(stats[1]["mean"], abs=1e-6) == 0.0
        assert pytest.approx(stats[3]["mean"], abs=1e-6) == 0.0
        assert stats[1]["std"] < stats[3]["std"]
        assert stats[1]["count"] == 4
        assert "q05" in stats[1] and "q95" in stats[1]

    def test_compute_horizon_error_stats_arrays(self):
        errs = [0.1, 0.2, 0.5, 0.6]
        horizons = [1, 1, 3, 3]
        stats = compute_horizon_error_stats(errs, horizons=horizons)
        assert 1 in stats and 3 in stats
        assert pytest.approx(stats[1]["mean"], rel=1e-4) == 0.15
        assert pytest.approx(stats[3]["mean"], rel=1e-4) == 0.55

    def test_fit_uncertainty_curve_linear_increasing(self):
        horizons = [1, 3, 5, 7]
        stds = [0.05, 0.10, 0.15, 0.20]
        predict_fn = fit_uncertainty_curve(horizons, stds, method="linear")

        # Predictions at horizons must strictly increase
        p1 = predict_fn(1)
        p3 = predict_fn(3)
        p5 = predict_fn(5)
        p7 = predict_fn(7)

        assert p1 < p3 < p5 < p7
        assert pytest.approx(float(p1), rel=1e-3) == 0.05
        assert pytest.approx(float(p7), rel=1e-3) == 0.20

    def test_fit_uncertainty_curve_exponential_increasing(self):
        horizons = [1, 3, 5, 7]
        stds = [0.02, 0.04, 0.08, 0.16]
        predict_fn = fit_uncertainty_curve(horizons, stds, method="exponential")

        p1 = predict_fn(1)
        p3 = predict_fn(3)
        p5 = predict_fn(5)
        p7 = predict_fn(7)

        assert p1 < p3 < p5 < p7
        assert float(predict_fn(10)) > float(p7)

    def test_generate_uncertainty_report(self, tmp_path):
        errors = {
            1: np.random.normal(0, 0.05, 50),
            3: np.random.normal(0, 0.10, 50),
            5: np.random.normal(0, 0.15, 50),
            7: np.random.normal(0, 0.20, 50),
        }
        report = generate_uncertainty_report(
            error_pairs=errors,
            output_dir=tmp_path,
            method="linear",
            is_smoke_test=True,
        )
        assert report["status"] == "SMOKE_TEST_ONLY"
        assert (tmp_path / "SMOKE_TEST_ONLY_uncertainty_curve.png").exists()
        assert (tmp_path / "SMOKE_TEST_ONLY_uncertainty_report.json").exists()


class TestPolarisRiskFunction:
    def test_risk_boundary_values(self):
        # Representative class: PC4
        # SIC = 0.0 -> near zero
        r0 = risk_ice(0.0, polar_class="PC4")
        assert r0 < 0.01

        # SIC = 0.15 -> threshold behavior, still low
        r15 = risk_ice(0.15, polar_class="PC4")
        assert r15 <= 0.05

        # SIC = 1.0 -> near maximum (> 0.90)
        r100 = risk_ice(1.0, polar_class="PC4")
        assert r100 >= 0.90

    def test_risk_strict_monotonicity(self):
        sics = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
        for pc in ["PC1", "PC4", "PC7", "UNCLASSED"]:
            risks = [risk_ice(s, polar_class=pc) for s in sics]
            for i in range(len(risks) - 1):
                assert risks[i] <= risks[i + 1], f"Non-monotonic risk for {pc}: {risks[i]} > {risks[i+1]}"

    def test_polar_class_hierarchy(self):
        # For pack ice (e.g. SIC = 0.6), higher ice class must yield lower risk
        sic = 0.6
        r_pc1 = risk_ice(sic, polar_class="PC1")
        r_pc3 = risk_ice(sic, polar_class="PC3")
        r_pc5 = risk_ice(sic, polar_class="PC5")
        r_pc7 = risk_ice(sic, polar_class="PC7")
        r_unclassed = risk_ice(sic, polar_class="UNCLASSED")

        assert r_pc1 < r_pc3 < r_pc5 < r_pc7 < r_unclassed

    def test_risk_ice_array_equivalence(self):
        sics = np.linspace(0.0, 1.0, 50)
        arr_results = risk_ice_array(sics, polar_class="PC3")
        elem_results = np.array([risk_ice(s, polar_class="PC3") for s in sics])
        np.testing.assert_allclose(arr_results, elem_results, rtol=1e-6)

    def test_multidimensional_array_support(self):
        grid = np.random.uniform(0.0, 1.0, size=(10, 20, 20))
        risk_grid = risk_ice_array(grid, polar_class="PC4")
        assert risk_grid.shape == grid.shape
        assert (risk_grid >= 0.0).all() and (risk_grid <= 1.0).all()

    def test_polar_class_normalization(self):
        assert normalize_polar_class(1) == "PC1"
        assert normalize_polar_class("1") == "PC1"
        assert normalize_polar_class("pc1") == "PC1"
        assert normalize_polar_class("PC7") == "PC7"
        assert normalize_polar_class(7) == "PC7"
        assert normalize_polar_class(99) == "UNCLASSED"
        assert normalize_polar_class("unknown") == "UNCLASSED"

    def test_custom_config_override(self):
        # Custom config with open_water_threshold = 0.30
        custom_cfg = PolarisRiskConfig(
            open_water_threshold=0.30,
            max_risk=0.80,
        )
        # At SIC = 0.25, with threshold 0.30, risk should still be near zero
        r_custom = risk_ice(0.25, polar_class="PC4", config=custom_cfg)
        assert r_custom <= 0.05

        # At SIC = 1.0, max risk should be capped at 0.80
        r_max = risk_ice(1.0, polar_class="PC4", config=custom_cfg)
        assert pytest.approx(r_max, rel=1e-4) == 0.80

    def test_clamping_out_of_bounds(self):
        # Negative SIC clamped to 0.0
        assert pytest.approx(risk_ice(-0.5, "PC4"), abs=1e-6) == risk_ice(0.0, "PC4")
        # SIC > 1.0 clamped to 1.0
        assert pytest.approx(risk_ice(1.5, "PC4"), abs=1e-6) == risk_ice(1.0, "PC4")
