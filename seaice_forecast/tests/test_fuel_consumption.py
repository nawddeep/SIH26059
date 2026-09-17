"""Tests for the ice-aware fuel and speed model."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.fuel import (
    HEAVY_ICE_SPEED_RETENTION,
    FuelConfig,
    fuel_per_cell,
    fuel_per_cell_array,
    speed_in_ice,
    speed_in_ice_array,
    transit_time_hours,
)

PIXEL_KM = 25.0


class TestCalibration:
    def test_open_water_reference_is_exactly_the_documented_value(self):
        """The router's normalisation constant depends on this being 0.875."""
        assert fuel_per_cell(PIXEL_KM, 0.0, "PC4") == pytest.approx(0.875, abs=1e-9)

    def test_open_water_burn_is_base_rate_times_distance(self):
        cfg = FuelConfig(base_rate_t_per_km=0.05)
        assert fuel_per_cell(PIXEL_KM, 0.0, "PC4", cfg) == pytest.approx(1.25)

    def test_heavy_ice_penalty_is_4_3x(self):
        """Documented anchor: SIC=0.95 costs 4.3x open water for a PC4."""
        ratio = fuel_per_cell(PIXEL_KM, 0.95, "PC4") / fuel_per_cell(PIXEL_KM, 0.0, "PC4")
        assert ratio == pytest.approx(4.3, abs=0.05)

    def test_pc4_transit_time_roughly_triples_in_heavy_ice(self):
        ratio = transit_time_hours(PIXEL_KM, 0.95, "PC4") / transit_time_hours(
            PIXEL_KM, 0.0, "PC4"
        )
        assert ratio == pytest.approx(3.0, abs=0.1)

    def test_pc4_cell_costs_stay_in_the_documented_range(self):
        fuel = fuel_per_cell_array(PIXEL_KM, np.linspace(0, 1, 101), "PC4")
        assert fuel.min() == pytest.approx(0.875, abs=1e-9)
        assert 4.0 < fuel.max() < 4.5


class TestMonotonicity:
    def test_fuel_strictly_increases_with_sic(self):
        fuel = fuel_per_cell_array(PIXEL_KM, np.linspace(0, 1, 501), "PC4")
        assert np.all(np.diff(fuel) > 0)

    def test_speed_strictly_decreases_with_sic(self):
        speed = speed_in_ice_array(np.linspace(0, 1, 501), "PC4")
        assert np.all(np.diff(speed) < 0)

    @pytest.mark.parametrize("polar_class", list(HEAVY_ICE_SPEED_RETENTION))
    def test_monotonic_for_every_ice_class(self, polar_class):
        fuel = fuel_per_cell_array(PIXEL_KM, np.linspace(0, 1, 201), polar_class)
        assert np.all(np.diff(fuel) > 0)

    def test_open_water_speed_is_service_speed(self):
        assert speed_in_ice(0.0, "PC4") == pytest.approx(12.0)


class TestIceClassAwareness:
    def test_pc1_burns_less_than_unclassed_at_the_same_sic(self):
        """Stronger vessels hold speed, so they burn less over the same ground."""
        assert fuel_per_cell(PIXEL_KM, 0.95, "PC1") < fuel_per_cell(
            PIXEL_KM, 0.95, "UNCLASSED"
        )

    def test_fuel_increases_monotonically_as_class_weakens(self):
        burns = [
            fuel_per_cell(PIXEL_KM, 0.9, cls) for cls in HEAVY_ICE_SPEED_RETENTION
        ]
        assert burns == sorted(burns)

    def test_all_classes_agree_in_open_water(self):
        """Ice class is irrelevant with no ice to break."""
        burns = {
            cls: fuel_per_cell(PIXEL_KM, 0.0, cls) for cls in HEAVY_ICE_SPEED_RETENTION
        }
        assert len(set(round(v, 12) for v in burns.values())) == 1

    def test_unknown_class_is_rejected(self):
        with pytest.raises(KeyError, match="Unknown polar class"):
            fuel_per_cell(PIXEL_KM, 0.5, "PC99")


class TestInputHandling:
    def test_percent_and_fraction_inputs_agree(self):
        as_pct = fuel_per_cell_array(PIXEL_KM, np.array([0.0, 50.0, 95.0]), "PC4")
        as_frac = fuel_per_cell_array(PIXEL_KM, np.array([0.0, 0.5, 0.95]), "PC4")
        np.testing.assert_allclose(as_pct, as_frac)

    def test_sic_is_clipped_to_valid_range(self):
        assert fuel_per_cell(PIXEL_KM, -0.2, "PC4") == pytest.approx(
            fuel_per_cell(PIXEL_KM, 0.0, "PC4")
        )

    def test_distance_broadcasts_against_the_sic_grid(self):
        sic = np.array([[0.0, 0.5], [0.8, 1.0]])
        out = fuel_per_cell_array(PIXEL_KM, sic, "PC4")
        assert out.shape == sic.shape

    def test_per_step_distances_broadcast_elementwise(self):
        steps = np.array([25.0, 35.36])
        sic = np.array([0.0, 0.0])
        out = fuel_per_cell_array(steps, sic, "PC4")
        assert out[1] / out[0] == pytest.approx(35.36 / 25.0)


class TestConfigOverrides:
    def test_service_speed_changes_transit_time_but_not_fuel(self):
        """Fuel per km is set by base_rate; v_ref only sets the clock."""
        fast = FuelConfig(v_ref_kn=18.0)
        assert transit_time_hours(PIXEL_KM, 0.0, "PC4", fast) < transit_time_hours(
            PIXEL_KM, 0.0, "PC4"
        )
        assert fuel_per_cell(PIXEL_KM, 0.0, "PC4", fast) == pytest.approx(
            fuel_per_cell(PIXEL_KM, 0.0, "PC4")
        )

    def test_base_rate_scales_fuel_linearly(self):
        doubled = FuelConfig(base_rate_t_per_km=0.070)
        assert fuel_per_cell(PIXEL_KM, 0.6, "PC4", doubled) == pytest.approx(
            2 * fuel_per_cell(PIXEL_KM, 0.6, "PC4")
        )
