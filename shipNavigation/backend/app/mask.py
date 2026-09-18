"""Land/ocean mask rasterized from Natural Earth polygons onto a 0.5-degree grid.

The grid is 360 rows (lat) x 720 cols (lon), cell size 0.5 degrees. A cell is
land when its centre falls inside a Natural Earth land polygon. This is the
navigable-water field used by the A* land-avoidance stage. At this coarse
resolution narrow straits/canals are not individually resolvable, which is an
acknowledged v1 simplification.
"""

from __future__ import annotations

import json
import os
import time

import shapely.geometry as sgeo
from shapely import points as shapely_points
from shapely.ops import unary_union
from shapely.prepared import prep

from . import geo

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
LAND_FILE = os.path.join(DATA, "ne_110m_land.geojson")

SPACING = 0.5
NROWS = int(180 / SPACING)          # 360
NCOLS = int(360 / SPACING)          # 720
LAT_TOP = 90.0 - SPACING / 2.0      # 89.75
LON_LEFT = -180.0 + SPACING / 2.0   # -179.75


class LandMask:
    def __init__(self, land_geojson_path: str = LAND_FILE):
        t0 = time.time()
        with open(land_geojson_path) as f:
            fc = json.load(f)
        polys = [sgeo.shape(feat["geometry"]) for feat in fc["features"]]
        land = unary_union([p for p in polys if not p.is_empty])
        print(f"[mask] building {NROWS}x{NCOLS} land grid ...", flush=True)
        self.land = self._build_grid(land)
        print(f"[mask] mask ready in {time.time() - t0:.1f}s", flush=True)

    def _build_grid(self, land):
        grid = bytearray(NROWS * NCOLS)
        # Use prepared geometry for much faster point-in-polygon tests
        prepared_land = prep(land)
        chunk_rows = 16  # Process more rows at once for better vectorization
        for r0 in range(0, NROWS, chunk_rows):
            r_end = min(r0 + chunk_rows, NROWS)
            pts = []
            idx = []
            for r in range(r0, r_end):
                lat = LAT_TOP - r * SPACING
                base = r * NCOLS
                for c in range(NCOLS):
                    lon = LON_LEFT + c * SPACING
                    pts.append((lon, lat))
                    idx.append(base + c)
            # Use prepared geometry contains for faster testing
            points_arr = shapely_points(pts)
            for j, (pt, i) in enumerate(zip(points_arr, idx)):
                if prepared_land.contains(pt):
                    grid[i] = 1
            del pts, idx, points_arr
        return grid

    def cell(self, lat: float, lon: float) -> bool:
        """True if the cell containing (lat, lon) is land."""
        r = round((LAT_TOP - lat) / SPACING)
        c = round((lon - LON_LEFT) / SPACING)
        if r < 0 or r >= NROWS:
            r = max(0, min(NROWS - 1, r))
        c = c % NCOLS
        return bool(self.land[r * NCOLS + c])

    def is_water(self, lat: float, lon: float) -> bool:
        return not self.cell(lat, lon)

    def nearest_water(self, lat: float, lon: float, max_radius: int = 14):
        """Return the (lat, lon) centre of the nearest water cell."""
        if self.is_water(lat, lon):
            return lat, lon
        r0 = round((LAT_TOP - lat) / SPACING)
        c0 = round((lon - LON_LEFT) / SPACING)
        for radius in range(1, max_radius + 1):
            candidates = []
            for dr in (-radius, radius):
                rr = r0 + dr
                if rr < 0 or rr >= NROWS:
                    continue
                for dc in range(-radius, radius + 1):
                    cc = (c0 + dc) % NCOLS
                    if self.land[rr * NCOLS + cc] == 0:
                        candidates.append((rr, cc))
            for dc in (-radius, radius):
                cc = (c0 + dc) % NCOLS
                for dr in range(-radius + 1, radius):
                    rr = r0 + dr
                    if rr < 0 or rr >= NROWS:
                        continue
                    if self.land[rr * NCOLS + cc] == 0:
                        candidates.append((rr, cc))
            if candidates:
                best = None
                best_d = float("inf")
                for rr, cc in candidates:
                    lat2 = LAT_TOP - rr * SPACING
                    lon2 = LON_LEFT + cc * SPACING
                    d = geo.geodesic_distance_nm(lat, lon, lat2, lon2)
                    if d < best_d:
                        best_d = d
                        best = (lat2, lon2)
                return best
        return None

    def carve_water(self, polyline_latlon):
        """Force a water corridor along a polyline of (lat, lon) points.

        Narrow real-water straits/canals are thinner than a single grid cell, so
        the centre-point land test erases them and splits one sea into two
        disconnected graph components ("enclosed sea" failures). This re-opens a
        belt of cells along the given corridor — the same named chokepoints used
        for crossing detection — restoring connectivity at the coarse resolution.

        Cells inside the belt are only converted to water when they are (chain-)
        adjacent to existing water, so a corridor floating over solid land is
        a no-op rather than a tunnel through a landmass.
        """
        if not polyline_latlon:
            return
        belt = set()
        for (la1, lo1), (la2, lo2) in zip(polyline_latlon, polyline_latlon[1:]):
            sub = geo.sample_geodesic(la1, lo1, la2, lo2, step_nm=SPACING * 60.0)
            for la, lo in sub:
                r = round((LAT_TOP - la) / SPACING)
                c = round((lo - LON_LEFT) / SPACING) % NCOLS
                if 0 <= r < NROWS:
                    belt.add((r, c))
        wide = set()
        for r, c in belt:
            for dr in (-1, 0, 1):
                rr = r + dr
                if rr < 0 or rr >= NROWS:
                    continue
                for dc in (-1, 0, 1):
                    wide.add((rr, (c + dc) % NCOLS))
        changed = True
        while changed:
            changed = False
            for r, c in wide:
                i = r * NCOLS + c
                if self.land[i] == 0:
                    continue
                if self._touches_water(r, c):
                    self.land[i] = 0
                    changed = True

    def _touches_water(self, r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            if rr < 0 or rr >= NROWS:
                continue
            if self.land[rr * NCOLS + (c + dc) % NCOLS] == 0:
                return True
        return False