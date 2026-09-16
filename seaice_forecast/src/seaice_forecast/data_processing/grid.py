"""
Antarctic Polar Stereographic Reference Grid and Ground-Truth Mask.

Defines the exact NSIDC 25km Southern Hemisphere Polar Stereographic grid
(EPSG:3412 / Hughes 1980 ellipsoid) and provides precomputed latitude/longitude
coordinates for fast regridding of atmospheric and oceanographic datasets.

Reference specifications (from NOAA/NSIDC G02202 v6):
- Projection: Polar Stereographic South (EPSG:3412)
  - Latitude of origin (standard parallel): -70.0 degrees
  - Central meridian: 0.0 degrees
  - Ellipsoid: Hughes 1980 (a=6378273.0, b=6356889.449)
- Dimensions: Height (y) = 332, Width (x) = 316
- Coordinate arrays:
  - x: 316 points, -3,937,500.0 m to +3,937,500.0 m (spacing +25,000 m)
  - y: 332 points, +4,337,500.0 m to -3,937,500.0 m (spacing -25,000 m)
- Ground-truth Mask:
  - Ocean: 1 (valid sea ice / open ocean)
  - Land: 0 (Antarctica continent, sub-Antarctic land masses)
"""

from pathlib import Path
from typing import Tuple, Optional
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Official NSIDC Antarctic 25km Polar Stereographic parameters
NSIDC_EPSG = 3412
PROJ4_PARAMS = (
    "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 "
    "+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
)

GRID_HEIGHT = 332  # rows / y-dimension
GRID_WIDTH = 316   # cols / x-dimension

X_MIN = -3937500.0
X_MAX = 3937500.0
X_STEP = 25000.0

Y_MAX = 4337500.0
Y_MIN = -3937500.0
Y_STEP = -25000.0


class AntarcticGrid:
    """
    Manages the reference NSIDC PS25 Antarctic grid, coordinate transformations,
    and single ground-truth land/ocean mask.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.shape = (GRID_HEIGHT, GRID_WIDTH)

        self.x_coords = np.linspace(X_MIN, X_MAX, GRID_WIDTH, dtype=np.float64)
        self.y_coords = np.linspace(Y_MAX, Y_MIN, GRID_HEIGHT, dtype=np.float64)

        self.grid_x, self.grid_y = np.meshgrid(self.x_coords, self.y_coords)

        self.cache_dir = Path(cache_dir) if cache_dir else (
            Path(__file__).parent.parent.parent.parent / "data" / "processed"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lons: Optional[np.ndarray] = None
        self._lats: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

    def get_lon_lat_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute or load cached 2D (longitude, latitude) arrays matching the grid.

        Returns:
            Tuple of (lons, lats) arrays of shape (332, 316), float32.
            Longitude is in [-180, 180] degrees_east.
            Latitude is in [-90, -39.23] degrees_north.
        """
        if self._lons is not None and self._lats is not None:
            return self._lons, self._lats

        coord_cache = self.cache_dir / "nsidc_ps25_lon_lat.npz"
        if coord_cache.exists():
            data = np.load(coord_cache)
            self._lons = data["lons"]
            self._lats = data["lats"]
            logger.debug(f"Loaded cached grid coordinates from {coord_cache}")
            return self._lons, self._lats

        logger.info("Transforming polar stereographic grid to lat/lon coordinates...")
        import pyproj

        transformer = pyproj.Transformer.from_crs(
            PROJ4_PARAMS, "EPSG:4326", always_xy=True
        )
        lons, lats = transformer.transform(self.grid_x, self.grid_y)

        # Ensure longitudes are wrapped to [-180, 180]
        lons = (lons + 180.0) % 360.0 - 180.0

        self._lons = lons.astype(np.float32)
        self._lats = lats.astype(np.float32)

        np.savez_compressed(coord_cache, lons=self._lons, lats=self._lats)
        logger.info(f"Saved cached grid coordinates to {coord_cache}")
        return self._lons, self._lats

    def get_land_ocean_mask(self, reference_nc_path: Optional[Path] = None) -> np.ndarray:
        """
        Get ground-truth binary land/ocean mask: 1 = ocean, 0 = land.

        Args:
            reference_nc_path: Optional path to real NSIDC NetCDF file to derive mask.

        Returns:
            np.ndarray: uint8 binary mask of shape (332, 316).
        """
        if self._mask is not None:
            return self._mask

        mask_path = self.cache_dir / "land_ocean_mask_ps25.npy"
        if mask_path.exists():
            self._mask = np.load(mask_path).astype(np.uint8)
            logger.debug(f"Loaded land/ocean mask from {mask_path}")
            return self._mask

        # Fallback to standard locations for reference NSIDC file
        if reference_nc_path is None:
            candidates = [
                self.cache_dir.parent / "raw" / "real" / "nsidc" / "2024" / "01" / "sic_pss25_20240101_F17_v06r00.nc",
                self.cache_dir.parent / "raw" / "nsidc" / "2024" / "01" / "sic_pss25_20240101_F17_v06r00.nc",
            ]
            for c in candidates:
                if c.exists():
                    reference_nc_path = c
                    break

            # Fall back to any downloaded NSIDC daily file (data/data/sic/{YYYY}/*.nc).
            # The CDR land mask is static, so any single day is a valid reference.
            if reference_nc_path is None:
                for sic_root in (self.cache_dir.parent / "data" / "sic",
                                 self.cache_dir.parent / "sic"):
                    if sic_root.is_dir():
                        found = sorted(sic_root.glob("*/sic_*.nc"))
                        if found:
                            reference_nc_path = found[0]
                            break

        if reference_nc_path is None or not Path(reference_nc_path).exists():
            raise FileNotFoundError(
                f"Cannot build ground-truth land/ocean mask: reference NSIDC NetCDF not found at {reference_nc_path}. "
                "Provide a valid NSIDC NetCDF file path."
            )

        logger.info(f"Building ground-truth land/ocean mask from {reference_nc_path}...")
        ds = xr.open_dataset(reference_nc_path)
        try:
            # Check for SIC variable
            if "cdr_seaice_conc" in ds:
                sic = ds["cdr_seaice_conc"].values[0]
            elif "seaice_conc_cdr" in ds:
                sic = ds["seaice_conc_cdr"].values[0]
            else:
                for v in ds.data_vars:
                    if "conc" in v.lower():
                        sic = ds[v].values[0]
                        break
                else:
                    raise KeyError("No SIC concentration variable found in reference file.")

            # In NSIDC CDR v6:
            # - Valid ocean/sea-ice cells have finite float values in [0.0, 1.0] (or [0, 100])
            # - Land cells are NaN (over the Antarctic continent & islands)
            ocean_mask = np.isfinite(sic) & (sic >= 0.0)

            self._mask = ocean_mask.astype(np.uint8)
            np.save(mask_path, self._mask)
            logger.info(
                f"Ground-truth mask created: {np.sum(self._mask == 1)} ocean cells, "
                f"{np.sum(self._mask == 0)} land cells (shape {self._mask.shape}). "
                f"Saved to {mask_path}"
            )
            return self._mask
        finally:
            ds.close()
