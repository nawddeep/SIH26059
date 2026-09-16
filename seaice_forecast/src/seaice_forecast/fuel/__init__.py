"""Fuel consumption modelling for ice-going vessels."""
from seaice_forecast.fuel.consumption import (
    FuelConfig,
    DEFAULT_FUEL_CONFIG,
    speed_in_ice,
    speed_in_ice_array,
    fuel_per_cell,
    fuel_per_cell_array,
)

__all__ = [
    "FuelConfig",
    "DEFAULT_FUEL_CONFIG",
    "speed_in_ice",
    "speed_in_ice_array",
    "fuel_per_cell",
    "fuel_per_cell_array",
]
