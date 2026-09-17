"""Ice-aware fuel consumption model."""

from seaice_forecast.fuel.consumption import (
    DEFAULT_FUEL_CONFIG,
    HEAVY_ICE_SPEED_RETENTION,
    KM_PER_NM,
    FuelConfig,
    fuel_per_cell,
    fuel_per_cell_array,
    fuel_per_km_array,
    speed_factor_array,
    speed_in_ice,
    speed_in_ice_array,
    transit_time_hours,
    transit_time_hours_array,
)

__all__ = [
    "DEFAULT_FUEL_CONFIG",
    "HEAVY_ICE_SPEED_RETENTION",
    "KM_PER_NM",
    "FuelConfig",
    "fuel_per_cell",
    "fuel_per_cell_array",
    "fuel_per_km_array",
    "speed_factor_array",
    "speed_in_ice",
    "speed_in_ice_array",
    "transit_time_hours",
    "transit_time_hours_array",
]
