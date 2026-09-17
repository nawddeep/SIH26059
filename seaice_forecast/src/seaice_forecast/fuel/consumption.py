"""Ice-aware fuel consumption and speed model for polar transits.

Physical basis
--------------
A ship in open water holds its service speed ``v_ref`` at its service power
``P_ref``.  Fuel burn per kilometre is then a constant, ``base_rate_t_per_km``.

In ice two things happen at once, and they push fuel per kilometre in the same
direction:

1. **Speed collapses.**  Ice resistance rises steeply with concentration, so the
   attainable speed ``v(SIC)`` falls well below ``v_ref``.
2. **Power rises.**  The vessel pushes up toward its maximum continuous rating
   to keep moving at all, so delivered power ``P(SIC) > P_ref``.

Fuel burned per unit *time* scales with delivered power, so fuel per unit
*distance* is::

    fuel_per_km(SIC) = base_rate_t_per_km * power_ratio(SIC) / speed_factor(SIC)

Both effects multiply, which is why the penalty in heavy ice is large: the ship
burns more per hour *and* takes far longer to cover the same ground.

This also makes the model correctly ice-class-aware.  A PC1 icebreaker and an
UNCLASSED vessel in the same ice draw comparable power, but the PC1 *holds a
much higher speed*, so it spends fewer hours -- and therefore less fuel -- per
kilometre.  Class enters the model through :data:`HEAVY_ICE_SPEED_RETENTION`
only; nothing else needs to be tuned per class.

Speed model
-----------
Speed retention is a stretched exponential in concentration::

    speed_factor(SIC) = exp(-k * SIC ** sic_exponent)

which is exactly 1.0 in open water, strictly decreasing in SIC, and stays
positive (a vessel in 100% ice is slow, never mathematically stuck).  The decay
constant ``k`` is not a free knob: it is solved per ice class from the single
calibration anchor in :data:`HEAVY_ICE_SPEED_RETENTION`, the fraction of service
speed the class retains at ``heavy_ice_sic`` (0.95).

Power model
-----------
Delivered power ramps from service rating toward maximum continuous rating as
soon as the vessel meets meaningful ice::

    power_ratio(SIC) = 1 + (max_power_ratio - 1) * (1 - exp(-power_ramp_k * SIC))

Calibration (defaults, PC4, 25 km cell)
---------------------------------------
======================  ==========  =========================================
Quantity                Value       Meaning
======================  ==========  =========================================
open water fuel/cell    0.875 t     ``base_rate_t_per_km`` * 25 km
SIC=0.95 fuel/cell      3.77 t      **4.30x** the open-water burn
SIC=1.00 fuel/cell      4.24 t      4.85x
open water transit      1.12 h      25 km at 12 kn
SIC=0.95 transit        3.35 h      transit time roughly **triples**
======================  ==========  =========================================

Across ice classes at SIC=0.95 the penalty spans 2.63x (PC1) to 8.03x
(UNCLASSED); the PC4 default column spans 0.875-4.24 t per 25 km cell.

References
----------
Riska et al. (1997), *Performance of merchant vessels in the Baltic* -- speed
degradation with ice thickness/concentration.  IMO MSC.1/Circ.1519 (POLARIS)
for the ice-class ladder.  Values here are representative rather than
vessel-specific; supply a :class:`FuelConfig` to calibrate to a real ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

__all__ = [
    "FuelConfig",
    "KM_PER_NM",
    "HEAVY_ICE_SPEED_RETENTION",
    "speed_factor_array",
    "speed_in_ice_array",
    "speed_in_ice",
    "fuel_per_km_array",
    "fuel_per_cell_array",
    "fuel_per_cell",
    "transit_time_hours_array",
    "transit_time_hours",
]

#: Kilometres per nautical mile.
KM_PER_NM = 1.852

#: Fraction of open-water service speed retained at ``heavy_ice_sic`` (0.95),
#: by ice class.  This is the *only* per-class calibration in the model --
#: every other class difference follows from it.  Ordered strongest to weakest;
#: strictly decreasing, which is what makes fuel burn strictly increasing as
#: ice capability drops.  Order matches the POLARIS RIV dominance order in
#: :data:`~seaice_forecast.risk.polaris.RIV_TABLE`.
HEAVY_ICE_SPEED_RETENTION: dict[str, float] = {
    "PC1": 0.550,
    "PC2": 0.500,
    "PC3": 0.440,
    "PC4": 0.336,
    "PC5": 0.300,
    "PC6": 0.265,
    # IA Super shares PC6's POLARIS RIV row but is designed for Baltic rather
    # than polar service, so it is placed just below PC6 and above PC7.
    "IA_SUPER": 0.255,
    "PC7": 0.240,
    "IA": 0.215,
    "IB": 0.200,
    "IC": 0.190,
    "UNCLASSED": 0.180,
}


@dataclass(frozen=True)
class FuelConfig:
    """Vessel fuel/propulsion parameters.

    Defaults describe a mid-size ice-capable polar supply vessel (~4 MW at
    service rating) and are calibrated so a 25 km cell costs exactly
    0.875 t in open water.

    Attributes
    ----------
    base_rate_t_per_km:
        Open-water fuel burn at service speed, tonnes per kilometre.
    v_ref_kn:
        Open-water service speed, knots.
    max_power_ratio:
        Maximum continuous rating divided by service power.  The ceiling the
        vessel pushes toward in ice.
    power_ramp_k:
        How fast delivered power approaches ``max_power_ratio`` with SIC.
    sic_exponent:
        Exponent on SIC in the speed law.  >1 means light ice costs little
        speed while dense pack costs a lot.
    heavy_ice_sic:
        Concentration at which :data:`HEAVY_ICE_SPEED_RETENTION` is defined.
    """

    base_rate_t_per_km: float = 0.035
    v_ref_kn: float = 12.0
    max_power_ratio: float = 1.45
    power_ramp_k: float = 5.0
    sic_exponent: float = 2.0
    heavy_ice_sic: float = 0.95


DEFAULT_FUEL_CONFIG = FuelConfig()

ArrayLike = Union[float, np.ndarray]


def _resolve(config: Optional[Union[dict, FuelConfig]]) -> FuelConfig:
    if config is None:
        return DEFAULT_FUEL_CONFIG
    if isinstance(config, FuelConfig):
        return config
    if isinstance(config, dict):
        return FuelConfig(**config)
    return DEFAULT_FUEL_CONFIG


def _normalise_class(polar_class: Union[int, str]) -> str:
    """Accept ``4``, ``pc4``, ``PC4``, ``IA Super``, ``unclassed`` etc.

    Integer classes and the PC1-PC7/UNCLASSED ladder are resolved through
    risk.polaris so fuel and risk always agree on what a class means; the Baltic
    classes below extend that ladder and are matched directly.
    """
    key = str(polar_class).strip().upper().replace(" ", "_").replace("-", "_")
    if key in HEAVY_ICE_SPEED_RETENTION:
        return key

    # Numeric classes only. An unrecognised *name* stays an error rather than
    # falling through to UNCLASSED: silently costing a typo'd hull as the
    # weakest one gives plausible, wrong fuel figures and no warning.
    if isinstance(polar_class, (int, np.integer)) or key.isdigit():
        from seaice_forecast.risk.polaris import normalize_polar_class
        resolved = normalize_polar_class(polar_class)
        if resolved in HEAVY_ICE_SPEED_RETENTION:
            return resolved

    raise KeyError(
        f"Unknown polar class {polar_class!r}. "
        f"Known classes: {sorted(HEAVY_ICE_SPEED_RETENTION)}"
    )


def _decay_constant(polar_class: str, config: FuelConfig) -> float:
    """Solve the speed-law decay constant from the single class anchor.

    ``exp(-k * sic_h ** p) = retention``  =>  ``k = -ln(retention) / sic_h ** p``
    """
    retention = HEAVY_ICE_SPEED_RETENTION[_normalise_class(polar_class)]
    return -np.log(retention) / (config.heavy_ice_sic ** config.sic_exponent)


def _as_sic_fraction(sic: ArrayLike) -> np.ndarray:
    """Coerce SIC to a 0-1 float array.

    Accepts percent (0-100) or fraction (0-1); values above 1.5 are treated as
    percent.  NaNs are preserved so callers can mask them.
    """
    arr = np.asarray(sic, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) > 1.5:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------


def speed_factor_array(
    sic: ArrayLike,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> np.ndarray:
    """Attainable speed as a fraction of open-water service speed.

    Returns 1.0 at SIC=0 and decreases strictly and smoothly with SIC.
    """
    cfg = _resolve(config)
    frac = _as_sic_fraction(sic)
    k = _decay_constant(polar_class, cfg)
    return np.exp(-k * frac ** cfg.sic_exponent)


def speed_in_ice_array(
    sic: ArrayLike,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> np.ndarray:
    """Attainable speed in **knots** for each SIC value."""
    cfg = _resolve(config)
    return cfg.v_ref_kn * speed_factor_array(sic, polar_class, cfg)


def speed_in_ice(
    sic: float,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> float:
    """Scalar form of :func:`speed_in_ice_array`."""
    return float(speed_in_ice_array(sic, polar_class, config))


def _power_ratio(sic_fraction: np.ndarray, cfg: FuelConfig) -> np.ndarray:
    """Delivered power relative to service power."""
    return 1.0 + (cfg.max_power_ratio - 1.0) * (
        1.0 - np.exp(-cfg.power_ramp_k * sic_fraction)
    )


# --------------------------------------------------------------------------
# Fuel
# --------------------------------------------------------------------------


def fuel_per_km_array(
    sic: ArrayLike,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> np.ndarray:
    """Fuel burn in **tonnes per kilometre** for each SIC value.

    ``base_rate * power_ratio(SIC) / speed_factor(SIC)`` -- burning more per
    hour while covering less ground per hour.
    """
    cfg = _resolve(config)
    frac = _as_sic_fraction(sic)
    return (
        cfg.base_rate_t_per_km
        * _power_ratio(frac, cfg)
        / speed_factor_array(frac, polar_class, cfg)
    )


def fuel_per_cell_array(
    distance_km: ArrayLike,
    sic: ArrayLike,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> np.ndarray:
    """Fuel in **tonnes per cell** to traverse ``distance_km`` at each SIC.

    Parameters
    ----------
    distance_km:
        Traverse length per cell.  Scalar (uniform grid) or array broadcastable
        against ``sic``.
    sic:
        Sea-ice concentration grid, as a 0-1 fraction or 0-100 percent.
    polar_class:
        Vessel ice class, e.g. ``"PC4"``.
    config:
        Vessel parameters; defaults to :data:`DEFAULT_FUEL_CONFIG`.

    Returns
    -------
    np.ndarray
        Tonnes per cell, same shape as the broadcast of the inputs.  Strictly
        increasing in SIC.
    """
    return np.asarray(distance_km, dtype=float) * fuel_per_km_array(
        sic, polar_class, config
    )


def fuel_per_cell(
    distance_km: float,
    sic: float = 0.0,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> float:
    """Scalar form of :func:`fuel_per_cell_array`.

    Used by the router to derive its open-water normalisation reference.
    """
    return float(fuel_per_cell_array(distance_km, sic, polar_class, config))


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


def transit_time_hours_array(
    distance_km: ArrayLike,
    sic: ArrayLike,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> np.ndarray:
    """Hours to traverse ``distance_km`` at the ice-limited speed."""
    cfg = _resolve(config)
    speed_kmh = KM_PER_NM * speed_in_ice_array(sic, polar_class, cfg)
    return np.asarray(distance_km, dtype=float) / speed_kmh


def transit_time_hours(
    distance_km: float,
    sic: float = 0.0,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, FuelConfig]] = None,
) -> float:
    """Scalar form of :func:`transit_time_hours_array`."""
    return float(transit_time_hours_array(distance_km, sic, polar_class, config))
