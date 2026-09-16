"""
Fuel consumption model for ice-going vessels.

The PS asks for routes that are both *safe* and *fuel-efficient*. Those are not
the same objective, and the interesting result is that they often disagree: the
shortest path through heavy ice can burn more fuel than a longer detour through
open water, because speed collapses in ice and transit time — not distance —
drives consumption.

Chain
-----
    SIC  ->  attainable speed  ->  transit time  ->  fuel burned

Speed in ice
    v_ice = v_open * (1 - alpha * SIC^beta),  floored at v_min

    Ice-going vessels lose speed non-linearly with concentration: light ice
    costs little, heavy pack ice is near-impassable. `beta` > 1 captures that
    knee. Stronger Polar Classes lose less speed for the same SIC, which is
    consistent with how POLARIS treats ice capability, so we scale `alpha` by
    the same class ordering used in `seaice_forecast.risk.polaris`.

Fuel per transited cell
    Propulsion power rises steeply with speed (roughly cubic in calm water),
    but fuel per unit *distance* is what matters for routing:

        fuel = sfoc * P(v) * t,   t = d / v,   P(v) ~ P_ref * (v / v_ref)^3

    so per cell of length d:

        fuel(d, v) = base_rate * d * (v / v_ref)^2

    Pushing through ice at reduced speed does not simply scale that way: the
    engine works against ice resistance rather than coasting at low power, so
    we add an ice-resistance penalty proportional to SIC.

All units are kept explicit: distances in km, speeds in knots, fuel in tonnes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

KM_PER_NM = 1.852


@dataclass
class FuelConfig:
    """Vessel fuel/speed parameters. Defaults describe a mid-size polar research vessel."""

    v_open_kn: float = 12.0          # service speed in open water (knots)
    v_min_kn: float = 1.0            # minimum headway before considered beset
    alpha: float = 0.85              # fraction of speed lost as SIC -> 1
    beta: float = 1.6                # curvature of the speed/SIC response
    base_rate_t_per_km: float = 0.035  # tonnes per km at reference speed
    v_ref_kn: float = 12.0           # reference speed for the fuel curve
    ice_resistance_factor: float = 1.8  # extra burn pushing through ice
    open_water_threshold: float = 0.15  # below this, treat as open water

    # Speed-loss multiplier by Polar Class: stronger class -> less speed lost.
    # Ordering mirrors the POLARIS class exponents in risk/polaris.py.
    class_speed_factor: Dict[str, float] = field(default_factory=lambda: {
        "PC1": 0.45, "PC2": 0.55, "PC3": 0.65, "PC4": 0.75,
        "PC5": 0.85, "PC6": 0.95, "PC7": 1.05, "UNCLASSED": 1.35,
    })


DEFAULT_FUEL_CONFIG = FuelConfig()


def _cfg(config) -> FuelConfig:
    if config is None:
        return DEFAULT_FUEL_CONFIG
    if isinstance(config, FuelConfig):
        return config
    if isinstance(config, dict):
        return FuelConfig(**config)
    return DEFAULT_FUEL_CONFIG


def _class_factor(cfg: FuelConfig, polar_class) -> float:
    from seaice_forecast.risk.polaris import normalize_polar_class
    return cfg.class_speed_factor.get(normalize_polar_class(polar_class), 1.0)


def speed_in_ice(sic: float, polar_class: Union[int, str] = "PC4",
                 config: Optional[Union[dict, FuelConfig]] = None) -> float:
    """Attainable speed (knots) at a given sea-ice concentration."""
    cfg = _cfg(config)
    s = float(np.clip(sic, 0.0, 1.0))
    if s < cfg.open_water_threshold:
        return cfg.v_open_kn
    loss = cfg.alpha * _class_factor(cfg, polar_class) * (s ** cfg.beta)
    return float(max(cfg.v_open_kn * (1.0 - loss), cfg.v_min_kn))


def speed_in_ice_array(sic: np.ndarray, polar_class: Union[int, str] = "PC4",
                       config: Optional[Union[dict, FuelConfig]] = None) -> np.ndarray:
    """Vectorised `speed_in_ice` over a SIC grid."""
    cfg = _cfg(config)
    s = np.clip(np.asarray(sic, dtype=np.float64), 0.0, 1.0)
    loss = cfg.alpha * _class_factor(cfg, polar_class) * (s ** cfg.beta)
    v = np.maximum(cfg.v_open_kn * (1.0 - loss), cfg.v_min_kn)
    return np.where(s < cfg.open_water_threshold, cfg.v_open_kn, v)


def fuel_per_cell(distance_km: float, sic: float, polar_class: Union[int, str] = "PC4",
                  config: Optional[Union[dict, FuelConfig]] = None) -> float:
    """
    Fuel (tonnes) to traverse `distance_km` through ice of concentration `sic`.

    Open water reduces to `base_rate_t_per_km * distance_km`.
    """
    cfg = _cfg(config)
    v = speed_in_ice(sic, polar_class, cfg)
    s = float(np.clip(sic, 0.0, 1.0))
    # Two regimes. Open water: hydrodynamic, P ~ v^3, so fuel/km ~ (v/v_ref)^2
    # (slower is cheaper - real slow steaming). In ice: resistance is ice
    # breaking, power stays near-constant while speed collapses, so fuel/km
    # scales with transit TIME ~ (v_ref/v) - slower is now more expensive.
    open_term = (v / cfg.v_ref_kn) ** 2
    s_eff = s if s >= cfg.open_water_threshold else 0.0
    ice_term = cfg.ice_resistance_factor * s_eff * (cfg.v_ref_kn / v)
    return float(cfg.base_rate_t_per_km * distance_km * (open_term + ice_term))


def fuel_per_cell_array(distance_km: Union[float, np.ndarray], sic: np.ndarray,
                        polar_class: Union[int, str] = "PC4",
                        config: Optional[Union[dict, FuelConfig]] = None) -> np.ndarray:
    """Vectorised `fuel_per_cell` over a SIC grid."""
    cfg = _cfg(config)
    s = np.clip(np.asarray(sic, dtype=np.float64), 0.0, 1.0)
    v = speed_in_ice_array(s, polar_class, cfg)
    open_term = (v / cfg.v_ref_kn) ** 2
    s_eff = np.where(s >= cfg.open_water_threshold, s, 0.0)
    ice_term = cfg.ice_resistance_factor * s_eff * (cfg.v_ref_kn / v)
    return cfg.base_rate_t_per_km * np.asarray(distance_km, dtype=np.float64) * (open_term + ice_term)


def transit_time_hours(distance_km: float, sic: float,
                       polar_class: Union[int, str] = "PC4",
                       config: Optional[Union[dict, FuelConfig]] = None) -> float:
    """Hours to traverse `distance_km` at the ice-attainable speed."""
    cfg = _cfg(config)
    v_kn = speed_in_ice(sic, polar_class, cfg)
    return float(distance_km / (v_kn * KM_PER_NM))
