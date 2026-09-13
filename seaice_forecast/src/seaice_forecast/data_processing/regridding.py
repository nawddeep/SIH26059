"""
Fast Spatial Regridding Engine for Antarctic Sea-Ice Forecasting.

Reprojects regular lat/lon fields from ERA5 and CMEMS onto the reference
NSIDC 25km Southern Hemisphere Polar Stereographic grid (EPSG:3412, shape 332x316).

Uses scipy.interpolate.RegularGridInterpolator for high-throughput bilinear interpolation
(~5ms per 2D slice) and handles coastal/boundary NaNs explicitly per variable.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from seaice_forecast.data_processing.grid import AntarcticGrid

logger = logging.getLogger(__name__)

FREEZING_SST_KELVIN = 271.35  # -1.8 °C (freezing point of seawater at typical salinity)


class EnvironmentalRegridder:
    """
    High-performance regridder for ERA5, CMEMS, and NSIDC variables.
    """

    def __init__(self, grid: Optional[AntarcticGrid] = None):
        self.grid = grid or AntarcticGrid()
        self.target_shape = self.grid.shape  # (332, 316)
        self.target_lons, self.target_lats = self.grid.get_lon_lat_grids()
        self.mask = self.grid.get_land_ocean_mask()  # 1 = ocean, 0 = land

        # Pre-flatten target query points for vectorized interpolation
        self._target_pts = np.column_stack((self.target_lats.ravel(), self.target_lons.ravel()))

    def _fill_coastal_nans(self, field: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
        """
        Fill coastal NaNs in ocean pixels using nearest valid ocean pixel.
        """
        filled = field.copy()
        nan_ocean = np.isnan(filled) & (ocean_mask == 1)
        valid_ocean = np.isfinite(filled) & (ocean_mask == 1)

        if not np.any(nan_ocean):
            return filled

        if not np.any(valid_ocean):
            filled[nan_ocean] = 0.0
            return filled

        # Nearest neighbor lookup via Euclidean distance transform
        indices = distance_transform_edt(
            ~valid_ocean,
            return_distances=False,
            return_indices=True
        )
        filled[nan_ocean] = filled[tuple(indices[:, nan_ocean])]
        return filled

    def regrid_regular_field(
        self,
        data_2d: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        variable_name: str
    ) -> np.ndarray:
        """
        Regrid a 2D array from regular (lat, lon) to NSIDC PS25 (332, 316).

        Args:
            data_2d: 2D array [len(lats), len(lons)]
            lats: 1D latitude coordinates
            lons: 1D longitude coordinates
            variable_name: Variable name for NaN handling strategy

        Returns:
            2D array [332, 316], float32, NaN-free
        """
        # Ensure latitudes are strictly ascending
        if lats[1] < lats[0]:
            lats = lats[::-1]
            data_2d = data_2d[::-1, :]

        # Ensure longitudes are strictly ascending and in [-180, 180]
        lons_norm = (lons + 180.0) % 360.0 - 180.0
        if not np.all(np.diff(lons_norm) > 0):
            sort_idx = np.argsort(lons_norm)
            lons_norm = lons_norm[sort_idx]
            data_2d = data_2d[:, sort_idx]

        # Extend longitudes cyclically across the -180/180 boundary to prevent edge NaNs
        lon_step = float(np.median(np.diff(lons_norm)))
        lons_extended = np.concatenate(([lons_norm[0] - lon_step], lons_norm, [lons_norm[-1] + lon_step]))
        data_extended = np.pad(data_2d, ((0, 0), (1, 1)), mode="wrap")

        # Create bilinear interpolator
        interp = RegularGridInterpolator(
            (lats, lons_extended),
            data_extended,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        # Vectorized interpolation
        out_flat = interp(self._target_pts)
        out = out_flat.reshape(self.target_shape).astype(np.float32)

        # Variable-specific coastal / boundary NaN fill strategy
        if variable_name in ("u10", "wind_u", "v10", "wind_v", "t2m", "air_temp"):
            # Atmospheric variables: extrapolate polewards or fill remaining edge NaNs with boundary mean
            if np.any(np.isnan(out)):
                nan_mask = np.isnan(out)
                valid_mask = ~nan_mask
                if np.any(valid_mask):
                    indices = distance_transform_edt(
                        nan_mask,
                        return_distances=False,
                        return_indices=True
                    )
                    out[nan_mask] = out[tuple(indices[:, nan_mask])]

        elif variable_name in ("sst", "sea_surface_temperature"):
            # SST: Defined over ocean. Sub-ice ocean or missing coastal ocean gets seawater freezing temp
            # South of CMEMS/ERA5 ice edge or unobserved pack ice:
            ocean_nans = np.isnan(out) & (self.mask == 1)
            out[ocean_nans] = FREEZING_SST_KELVIN

            # Any remaining ocean coastal NaNs: fill via nearest valid ocean pixel
            out = self._fill_coastal_nans(out, self.mask)

            # Mask land pixels to 0.0
            out[self.mask == 0] = 0.0

        elif variable_name in ("uo", "current_u", "vo", "current_v"):
            # Ocean surface currents:
            # Under ice shelves / south of -80°S: 0.0 m/s (no-slip boundary)
            out[np.isnan(out)] = 0.0

            # Land pixels are strictly 0.0 m/s
            out[self.mask == 0] = 0.0

        return out

    def process_daily_bundle(
        self,
        sic_file: Path,
        era5_ds: Optional[xr.Dataset] = None,
        era5_time_idx: int = 0,
        cmems_ds: Optional[xr.Dataset] = None,
        cmems_time_idx: int = 0,
        target_date: Optional[str] = None
    ) -> np.ndarray:
        """
        Assemble and regrid a single daily multi-variable tensor.

        Variables order:
          0: SIC [0, 1]
          1: Wind U (m/s)
          2: Wind V (m/s)
          3: 2m Air Temp (K)
          4: SST (K)
          5: Current U (m/s)
          6: Current V (m/s)

        Returns:
            np.ndarray of shape (7, 332, 316), float32.
        """
        daily_array = np.zeros((7, self.target_shape[0], self.target_shape[1]), dtype=np.float32)

        # 1. SIC (native on NSIDC PS25 grid)
        with xr.open_dataset(sic_file) as ds_sic:
            if "cdr_seaice_conc" in ds_sic:
                sic_raw = ds_sic["cdr_seaice_conc"].values[0]
            elif "seaice_conc_cdr" in ds_sic:
                sic_raw = ds_sic["seaice_conc_cdr"].values[0]
            else:
                for v in ds_sic.data_vars:
                    if "conc" in v.lower():
                        sic_raw = ds_sic[v].values[0]
                        break
                else:
                    raise KeyError(f"No SIC variable found in {sic_file}")

            # Normalize to [0, 1] and mask land to 0
            sic_field = np.nan_to_num(sic_raw, nan=0.0).astype(np.float32)
            if np.nanmax(sic_field) > 1.5:  # Scaled 0-100%
                sic_field = sic_field * 0.01
            sic_field = np.clip(sic_field, 0.0, 1.0)
            sic_field[self.mask == 0] = 0.0
            daily_array[0] = sic_field

        def _get_var_name(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in ds.data_vars:
                    return c
            return None

        # 2. ERA5 Variables: u10, v10, t2m, sst
        if era5_ds is not None:
            t_coord = "valid_time" if "valid_time" in era5_ds.coords else ("time" if "time" in era5_ds.coords else None)
            if target_date is not None and t_coord is not None:
                era_dates = [str(t)[:10] for t in era5_ds[t_coord].values]
                if target_date in era_dates:
                    era5_time_idx = era_dates.index(target_date)
                else:
                    raise KeyError(f"target_date {target_date} not found in ERA5 dates: {era_dates}")

            lat_coord = "latitude" if "latitude" in era5_ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in era5_ds.coords else "lon"
            era_lats = era5_ds[lat_coord].values
            era_lons = era5_ds[lon_coord].values

            # Wind U
            u_name = _get_var_name(era5_ds, ["u10", "10m_u_component_of_wind", "var165"])
            if u_name:
                u10 = era5_ds[u_name].values[era5_time_idx]
                daily_array[1] = self.regrid_regular_field(u10, era_lats, era_lons, "wind_u")

            # Wind V
            v_name = _get_var_name(era5_ds, ["v10", "10m_v_component_of_wind", "var166"])
            if v_name:
                v10 = era5_ds[v_name].values[era5_time_idx]
                daily_array[2] = self.regrid_regular_field(v10, era_lats, era_lons, "wind_v")

            # 2m Temp
            t_name = _get_var_name(era5_ds, ["t2m", "2m_temperature", "var167"])
            if t_name:
                t2m = era5_ds[t_name].values[era5_time_idx]
                daily_array[3] = self.regrid_regular_field(t2m, era_lats, era_lons, "air_temp")

            # SST
            sst_name = _get_var_name(era5_ds, ["sst", "sea_surface_temperature", "var34"])
            if sst_name:
                sst = era5_ds[sst_name].values[era5_time_idx]
                daily_array[4] = self.regrid_regular_field(sst, era_lats, era_lons, "sst")
            else:
                logger.debug("No SST variable found in ERA5 dataset; filling with freezing point over ocean.")
                daily_array[4] = np.where(self.mask == 1, FREEZING_SST_KELVIN, 0.0).astype(np.float32)

        # 3. CMEMS Variables: uo, vo
        if cmems_ds is not None:
            if target_date is not None and "time" in cmems_ds.coords:
                cmems_dates = [str(t)[:10] for t in cmems_ds["time"].values]
                if target_date in cmems_dates:
                    cmems_time_idx = cmems_dates.index(target_date)
                else:
                    raise KeyError(f"target_date {target_date} not found in CMEMS dates: {cmems_dates}")

            lat_coord = "latitude" if "latitude" in cmems_ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in cmems_ds.coords else "lon"
            cmems_lats = cmems_ds[lat_coord].values
            cmems_lons = cmems_ds[lon_coord].values

            # Slice surface depth if 4D (time, depth, lat, lon)
            uo_name = _get_var_name(cmems_ds, ["uo", "current_u"])
            if uo_name:
                uo_var = cmems_ds[uo_name]
                if "depth" in uo_var.dims:
                    uo_raw = uo_var.isel(time=cmems_time_idx, depth=0).values
                else:
                    uo_raw = uo_var.isel(time=cmems_time_idx).values
                daily_array[5] = self.regrid_regular_field(uo_raw, cmems_lats, cmems_lons, "current_u")

            vo_name = _get_var_name(cmems_ds, ["vo", "current_v"])
            if vo_name:
                vo_var = cmems_ds[vo_name]
                if "depth" in vo_var.dims:
                    vo_raw = vo_var.isel(time=cmems_time_idx, depth=0).values
                else:
                    vo_raw = vo_var.isel(time=cmems_time_idx).values
                daily_array[6] = self.regrid_regular_field(vo_raw, cmems_lats, cmems_lons, "current_v")

        return daily_array
