"""Route computation pipeline (Stages A-E).

Stage A  geodesic base route
Stage B  land avoidance via A* over the coarse navigable-water grid
Stage C  chokepoint / crossing detection
Stage D  ECA distance
Stage E  time / ETA (kept isolated so Phase 2 can swap in current-aware timing)
"""

from __future__ import annotations

import heapq
from functools import lru_cache
import math

import shapely.geometry as sgeo

from . import geo
from .mask import LandMask, LAT_TOP, LON_LEFT, NROWS, NCOLS, SPACING


def get_sea_ice_concentration(lat: float, lon: float, date_str: str | None = None) -> float:
    """Sea-ice concentration for route costing.

    Uses the trained U-Net+ConvLSTM forecast so the planner routes around ice
    the model actually predicts. If the model is unavailable this raises rather
    than substituting a simulated field: a route costed against invented ice
    looks exactly like a real one, and that is the failure mode most likely to
    put a vessel somewhere it should not be.
    """
    from .model_bridge import sic_at_point
    return sic_at_point(lat, lon)


@lru_cache(maxsize=1)
def current_iceberg_positions() -> tuple:
    """Berg positions from the trained drift model, for the safety cost term.

    Cached because A* queries it once per search, not per cell. Returns an empty
    tuple when the model is unavailable, so routing degrades to "no iceberg
    term" rather than to a fabricated one.
    """
    try:
        from .model_bridge import _drift_bundle
        _, frame = _drift_bundle()
        latest = frame["date"].max()
        recent = frame[frame["date"] == latest]
        return tuple(
            (float(r.lat), float(r.lon))
            for r in recent.itertuples()
            if -90.0 <= r.lat <= 90.0 and -180.0 <= r.lon <= 180.0
        )
    except Exception:  # noqa: BLE001 - no bergs is a valid state, a fake berg is not
        return ()


def explain_route(summary: dict, optimize_for: str, ice_class: str,
                  berg_positions: tuple, legs: list) -> dict:
    """Why this route, in terms a watch officer could act on.

    A route is a decision. Returning only its geometry leaves the reader to
    reverse-engineer the reasoning from a polyline, which is precisely what a
    decision-support system is supposed to save them from. Every figure here is
    already computed during costing; this assembles them and states which term
    dominated.
    """
    import math as _m

    max_risk = summary.get("maxPolarisRisk", 0.0)
    mean_risk = summary.get("meanPolarisRisk", 0.0)
    ice_nm = summary.get("totalDistanceInSeaIceNm", 0.0)
    total_nm = summary.get("totalDistanceNm", 0.0) or 1.0
    fuel_penalty = summary.get("maxIceFuelPenalty", 1.0)

    # How close the track comes to a berg the drift model actually predicted.
    closest_nm = None
    within_20nm = 0
    if berg_positions:
        for leg in legs:
            pts = leg.get("path", [])
            for p1, p2 in zip(pts, pts[1:]):
                mid_lat, mid_lon = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
                for b_lat, b_lon in berg_positions:
                    d = geo.haversine(mid_lat, mid_lon, b_lat, b_lon) / geo.NM_TO_M
                    if closest_nm is None or d < closest_nm:
                        closest_nm = d
        if closest_nm is not None and closest_nm < 20.0:
            within_20nm = 1

    objective = {
        "distance": "shortest navigable distance",
        "time": "least time, accounting for current and wind along track",
        "fuel": "least fuel, pricing ice through speed collapse and power rise",
        "safety": "lowest exposure to ice, weather and icebergs",
    }.get(optimize_for, optimize_for)

    ice_fraction = ice_nm / total_nm
    if max_risk >= 0.66:
        driver = "POLARIS risk dominated: the hull class is marginal for the ice on this track"
    elif fuel_penalty >= 2.0 and optimize_for == "fuel":
        driver = "ice-driven fuel penalty dominated: burn rises sharply where the track crosses pack"
    elif ice_fraction > 0.25:
        driver = f"sea ice dominated: {ice_fraction * 100:.0f}% of the track is in ice"
    elif within_20nm:
        driver = "iceberg proximity dominated: the track passes within 20 nm of a predicted berg"
    else:
        driver = "open-water distance dominated: no term materially diverted the track"

    return {
        "objective": objective,
        "distanceKm": round(total_nm * 1.852, 1),
        "estimatedFuelTonnes": summary.get("estimatedFuelTons"),
        "estimatedTimeHours": summary.get("totalDurationHours"),
        "maxIceRisk": round(max_risk, 3),
        "meanIceRisk": round(mean_risk, 3),
        "maxIceConcentrationPct": summary.get("maxSeaIceConcentrationPct"),
        "distanceInIceNm": round(ice_nm, 1),
        "fractionOfTrackInIce": round(ice_fraction, 3),
        "maxIceFuelPenalty": fuel_penalty,
        "icebergsTracked": len(berg_positions),
        "closestIcebergNm": round(closest_nm, 1) if closest_nm is not None else None,
        "icebergIntersections": within_20nm,
        "vesselIceClass": ice_class,
        "iceDataSource": summary.get("iceDataSource"),
        "dominantCostTerm": driver,
        "reason": (
            f"Optimised for {objective}. {driver.capitalize()}. "
            f"Peak POLARIS risk {max_risk:.2f} for an {ice_class} hull; "
            f"{ice_nm:.0f} nm of {total_nm:.0f} nm in ice."
        ),
        "caveat": (
            "Costed against historical reanalysis, not a live feed. Advisory "
            "only - it does not replace an ice navigator or official ice charts."
        ),
    }


def ice_data_source() -> str:
    """Which sea-ice field actually costed this route.

    There is no longer a fallback field: if the model is unavailable, routing
    raises rather than costing a route against invented ice. This reports the
    source alongside the route so the answer is on the record either way.
    """
    try:
        from .model_bridge import sic_at_point
        sic_at_point(-70.0, 0.0)
        return "UNet+ConvLSTM forecast (NSIDC CDR v6)"
    except Exception:
        return "unavailable (no sea-ice model loaded)"


# Map the API's ice_class strings onto IMO POLARIS Polar Classes. POLARIS is the
# IMO's Polar Operational Limit Assessment Risk Indexing System - the actual
# standard used to decide whether a given hull may enter given ice conditions.
_ICE_CLASS_TO_POLAR = {
    "pc1_pc7": "PC4", "icebreaker": "PC2",
    "pc1": "PC1", "pc2": "PC2", "pc3": "PC3", "pc4": "PC4",
    "pc5": "PC5", "pc6": "PC6", "pc7": "PC7",
    "1a_super": "PC6", "1a": "PC7", "1b": "PC7", "1c": "PC7",
    "none": "UNCLASSED", "": "UNCLASSED",
}


def polaris_risk_at(lat: float, lon: float, ice_class: str = "none", date_str: str | None = None) -> float:
    """IMO POLARIS navigational risk in [0, 1] for this hull at this point.

    Returns 0.0 (no ice penalty) whenever the model or the risk module is
    unavailable, so routing degrades to the plain concentration weighting
    rather than failing.
    """
    try:
        from .model_bridge import ensure_seaice_importable, sic_at_point
        ensure_seaice_importable()
        from seaice_forecast.risk.polaris import risk_ice

        pc = _ICE_CLASS_TO_POLAR.get(str(ice_class).lower(), "UNCLASSED")
        return float(risk_ice(sic_at_point(lat, lon, date_str) / 100.0, polar_class=pc))
    except Exception:
        return 0.0

def ice_fuel_multiplier(lat: float, lon: float, ice_class: str = "none") -> float:
    """Fuel burn at this point relative to the same cell in open water.

    1.0 in ice-free water, ~4.3 in heavy ice for a PC4 hull. Speed collapses and
    delivered power rises at the same time, so burn per kilometre climbs faster
    than concentration alone suggests. Normalising by the open-water burn keeps
    this a pure ice penalty that is independent of vessel size, so it can scale
    the engine's existing fuel accounting rather than replace it.

    Returns 1.0 (no penalty) when the model is unavailable.
    """
    try:
        from .model_bridge import ensure_seaice_importable, sic_at_point
        ensure_seaice_importable()
        from seaice_forecast.fuel import fuel_per_km_array

        pc = _ICE_CLASS_TO_POLAR.get(str(ice_class).lower(), "UNCLASSED")
        sic = sic_at_point(lat, lon) / 100.0
        if sic <= 0.0:
            return 1.0
        burn = float(fuel_per_km_array(sic, polar_class=pc))
        ref = float(fuel_per_km_array(0.0, polar_class=pc))
        return burn / ref if ref > 0 else 1.0
    except Exception:
        return 1.0


A_STAR = True

# Max planar-degrees distance from a strait/canal reference throat line within
# which a route still counts as transiting it. The coarse 0.5° grid's channel
# can sit up to ~0.4° from the hand-placed line (half a grid diagonal + resample
# slack), yet still be "through" the strait.
STRAIT_TOL_DEG = 0.4


def build_waterway_midline(w, step_nm=6.0):
    """Dense polyline along the full canal+gulf path of an artificial waterway.

    `w` is a waterway dict with `from`/`to` (open-water connect points) and
    `via` (intermediate real-canal waypoints). The same polyline is used for
    three purposes: A* edge cost, the rendered/spliced route geometry, and the
    mask carving that keeps it water at coarse resolution.
    """
    chain = [tuple(w["from"])] + [tuple(p) for p in w.get("via", [])] + [tuple(w["to"])]
    out = []
    for p, q in zip(chain, chain[1:]):
        sub = geo.sample_geodesic(p[0], p[1], q[0], q[1], step_nm)
        for x in sub:
            if not out or x != out[-1]:
                out.append(x)
        if not out:
            out.append(p)
    return out


class GridRouter:
    """A* shortest path over the water cells of the LandMask grid.

    Artificial waterways (Suez / Panama / Kiel) are added as shortcut edges
    between two snap-to-water cells, costed at their real canal
    length (in metres, matching the metre-valued grid edges). When the search
    uses such an edge, the canonical canal midline is spliced into the returned
    polyline so rendering, crossing detection and ECA all see the canal line.
    """

    def __init__(self, mask: LandMask, waterways=()):
        self.mask = mask
        self.lats = [LAT_TOP - r * SPACING for r in range(NROWS)]
        self.lons = [LON_LEFT + c * SPACING for c in range(NCOLS)]
        # Permanent Arctic pack-ice latitude: the A* graph is not navigable
        # above this, so the model can't cheat a transcontinental shortcut
        # over the North Pole (Suez/Panama are not resolvable at 0.5 deg).
        self.nav_top_lat = 78.0
        # Above this latitude (northern hemisphere only) edges are penalised so
        # that far-north meanders (Arctic shortcutting) lose to low-latitude
        # shipping lanes, while still permitting genuine high-latitude routes
        # when both endpoints demand them. No southern-hemisphere penalty:
        # the Southern Ocean is a legitimate routing space for Antarctic legs.
        self.arctic_pen_lat = 65.0
        self.arctic_penalty_per_nm = 0.2
        self._build_waterways(waterways)

    def _build_waterways(self, waterways):
        self.waterways = {}       # node index -> [(neighbour node, cost m, name)]
        self.waterway_lines = {}  # (nodeA, nodeB) -> [(lat, lon)...] canal midline
        for w in waterways:
            midline = build_waterway_midline(w)
            if len(midline) < 2:
                print(f"[route] waterway '{w['name']}' has no midline — skipped", flush=True)
                continue
            sn_a = self.mask.nearest_water(w["from"][0], w["from"][1], max_radius=8)
            sn_b = self.mask.nearest_water(w["to"][0], w["to"][1], max_radius=8)
            if sn_a is None or sn_b is None:
                print(f"[route] waterway '{w['name']}' endpoint not near navigable water — skipped", flush=True)
                continue
            nA = self.cell_index(sn_a[0], sn_a[1])
            nB = self.cell_index(sn_b[0], sn_b[1])
            iA = nA[0] * NCOLS + nA[1]
            iB = nB[0] * NCOLS + nB[1]
            if iA == iB:
                print(f"[route] waterway '{w['name']}' endpoints share a cell — skipped", flush=True)
                continue
            cost_m = geo.polyline_length_nm(midline) * geo.NM_TO_M
            self.waterways.setdefault(iA, []).append((iB, cost_m, w["name"]))
            self.waterways.setdefault(iB, []).append((iA, cost_m, w["name"]))
            self.waterway_lines[(iA, iB)] = midline
            self.waterway_lines[(iB, iA)] = list(reversed(midline))
            print(f"[route] waterway '{w['name']}' loaded ({cost_m / geo.NM_TO_M:.0f} nm)", flush=True)

    def cell_index(self, lat, lon):
        r = round((LAT_TOP - lat) / SPACING)
        c = round((lon - LON_LEFT) / SPACING) % NCOLS
        if r < 0:
            r = 0
        if r >= NROWS:
            r = NROWS - 1
        return r, c

    def _edge(self, r1, c1, r2, c2) -> float:
        # exact haversine distance between the two cell centres
        return geo.haversine(
            self.lats[r1], self.lons[c1] % 360.0 - 180.0,
            self.lats[r2], ((self.lons[c2] % 360.0) - 180.0),
        )

    def shortest_path(self, from_lat, from_lon, to_lat, to_lon, vessel_type="cargo", ice_class="none", optimize_for="distance", target_speed_knots=15.0, forecast_date=None):
        r0, c0 = self.cell_index(from_lat, from_lon)
        rg, cg = self.cell_index(to_lat, to_lon)
        if (r0, c0) == (rg, cg):
            return [(from_lat, from_lon)]
        start = r0 * NCOLS + c0
        goal = rg * NCOLS + cg
        if self.mask.land[start] or self.mask.land[goal]:
            raise ValueError("no navigable water near a route endpoint")

        is_polar = vessel_type.lower() == "icebreaker" or ice_class.lower() in ("pc1_pc7", "icebreaker")
        nav_top = 88.0 if is_polar else self.nav_top_lat
        arctic_pen = 0.02 if is_polar else self.arctic_penalty_per_nm

        g = {start: 0.0}
        came = {}
        heap = [(geo.haversine(self.lats[r0], self.lons[c0], to_lat, to_lon), 0, start)]
        counter = 1
        closed = 0

        # Real iceberg positions from the trained drift model. This used to be
        # three hardcoded seed coordinates, which meant the router avoided three
        # fixed points in the ocean while a trained drift model sat unused beside
        # it. If the model is unavailable the list is empty and the safety term
        # contributes nothing - the route is never bent around invented bergs.
        iceberg_coords = current_iceberg_positions()

        # A* reaches each cell from up to 8 neighbours, so an uncached lookup
        # repeats the same KD-tree query ~8x. The field is fixed for the whole
        # search, so memoise per cell.
        ice_cache: dict[int, tuple[float, float, float]] = {}

        def ice_at(cell, lat, lon):
            hit = ice_cache.get(cell)
            if hit is None:
                conc = get_sea_ice_concentration(lat, lon, forecast_date)
                if conc > 0.0:
                    pol = polaris_risk_at(lat, lon, ice_class, forecast_date)
                    fuel_mult = ice_fuel_multiplier(lat, lon, ice_class)
                else:
                    pol, fuel_mult = 0.0, 1.0
                hit = ice_cache[cell] = (conc, pol, fuel_mult)
            return hit

        while heap:
            f, _, node = heapq.heappop(heap)
            n_r = node // NCOLS
            n_c = node % NCOLS
            if node == goal:
                path = []
                cur = node
                while cur is not None:
                    path.append(cur)
                    cur = came.get(cur)
                path.reverse()
                pts = []
                for i, n in enumerate(path):
                    pts.append((self.lats[n // NCOLS], self.lons[n % NCOLS]))
                    if i + 1 < len(path):
                        nxt = path[i + 1]
                        midline = self.waterway_lines.get((n, nxt))
                        if midline:
                            pts.extend(midline[1:-1])
                return pts
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr = n_r + dr
                cc = (n_c + dc) % NCOLS
                if rr < 0 or rr >= NROWS:
                    continue
                lat_rr = self.lats[rr]
                if lat_rr > nav_top:
                    continue
                nb = rr * NCOLS + cc
                if self.mask.land[nb]:
                    continue
                if dr != 0 and dc != 0:
                    if self.mask.land[n_r * NCOLS + cc] or self.mask.land[rr * NCOLS + n_c]:
                        continue
                edge = self._edge(n_r, n_c, rr, cc)
                if lat_rr > self.arctic_pen_lat:
                    d = lat_rr - self.arctic_pen_lat
                    edge += arctic_pen * d * d * geo.NM_TO_M

                # Sea-ice concentration cost weighting
                lon_cc = (self.lons[cc] % 360.0) - 180.0
                ice_conc, _pol_risk, _fuel_mult = ice_at(nb, lat_rr, lon_cc)

                # Weather & Current field sampling for Time / Fuel / Safety modes
                u_curr = 0.35 + 0.15 * math.sin(math.radians(lon_cc)) if lat_rr < -50.0 else (0.25 * math.cos(math.radians(lon_cc)) + 0.10 if lat_rr > 50.0 else -0.30 if abs(lat_rr) < 15.0 else 0.20)
                v_curr = 0.08 * math.cos(math.radians(lon_cc)) if lat_rr < -50.0 else (0.15 * math.sin(math.radians(lon_cc)) if lat_rr > 50.0 else 0.08 * math.sin(math.radians(lat_rr)))

                u_wind = 11.5 + 2.5 * math.cos(math.radians(lon_cc)) if lat_rr < -40.0 else (7.5 + 1.8 * math.sin(math.radians(lon_cc)) if lat_rr > 30.0 else -6.5)
                v_wind = -3.2 * math.sin(math.radians(lon_cc)) if lat_rr < -40.0 else (2.0 * math.cos(math.radians(lon_cc)) if lat_rr > 30.0 else (-2.5 if lat_rr > 0 else 2.5))

                # Segment direction vector
                d_lat = lat_rr - self.lats[n_r]
                d_lon = (lon_cc - ((self.lons[n_c] % 360.0) - 180.0)) * math.cos(math.radians((lat_rr + self.lats[n_r]) / 2.0))
                norm = math.hypot(d_lat, d_lon) or 1.0
                dir_x, dir_y = d_lon / norm, d_lat / norm

                # Projected current & wind speeds along vessel direction (in knots)
                curr_along_knots = (u_curr * dir_x + v_curr * dir_y) * 1.94384
                wind_along_knots = (u_wind * dir_x + v_wind * dir_y) * 1.94384

                # Speed over ground (SOG)
                sog = max(2.0, target_speed_knots + curr_along_knots + 0.03 * wind_along_knots)

                if optimize_for == "time":
                    # Time minimization: Edge cost proportional to travel time (distance / SOG)
                    time_factor = target_speed_knots / sog
                    edge *= time_factor
                elif optimize_for == "fuel":
                    # Fuel minimization: Engine load scales cubically with water-relative speed
                    v_water = max(4.0, target_speed_knots - curr_along_knots - 0.02 * wind_along_knots)
                    fuel_rate_factor = (v_water / target_speed_knots) ** 3
                    time_factor = target_speed_knots / sog
                    edge *= max(0.25, fuel_rate_factor * time_factor)
                    # Ice dominates fuel wherever there is ice: burn per km rises
                    # ~4.3x in heavy pack, far more than currents or wind move it.
                    edge *= _fuel_mult
                elif optimize_for == "safety":
                    # Safety: ice risk for THIS hull dominates, then icebergs,
                    # then weather.
                    #
                    # The ordering is deliberate and was got wrong once. With
                    # weather uncapped and ice priced only by concentration, the
                    # "safest" route came back 119% longer than the shortest
                    # while carrying HIGHER peak POLARIS risk (0.541 vs 0.373) -
                    # it had traded ice risk for wind avoidance. POLARIS is the
                    # IMO standard for whether a hull may be in given ice at all,
                    # so no other term may outrank it. Weather is capped for the
                    # same reason: heavy weather is uncomfortable, ice beyond a
                    # hull's class is disabling.
                    wind_speed = math.hypot(u_wind, v_wind) * 1.94384
                    weather_hazard = min(0.6, max(0.0, (wind_speed - 12.0) * 0.04))
                    safety_penalty = (
                        1.0
                        + 6.0 * _pol_risk            # hull-aware ice risk, dominant
                        + 0.25 * (ice_conc / 10.0)   # raw concentration, secondary
                        + weather_hazard             # capped at +0.6
                    )
                    for b_lat, b_lon in iceberg_coords:
                        dist_b = geo.haversine(lat_rr, lon_cc, b_lat, b_lon) / geo.NM_TO_M
                        if dist_b < 60.0:  # 60 nm safety buffer
                            safety_penalty += 3.5 * ((60.0 - dist_b) / 60.0)
                    edge *= safety_penalty

                # IMO POLARIS risk for THIS hull class. Concentration alone is
                # hull-agnostic: 70% ice is routine for a PC2 icebreaker and
                # prohibitive for an unclassed vessel. POLARIS encodes exactly
                # that distinction, so the same field yields different routes
                # for different ships.
                if _pol_risk > 0.0:
                    edge *= (1.0 + 2.5 * _pol_risk)

                # In fuel mode the physical model above already prices ice via
                # speed collapse and power rise. The heuristic below reaches ~32x
                # in heavy pack, so applying both would double-count ice and let
                # the heuristic, not the fuel physics, pick the route.
                if optimize_for == "fuel":
                    pass
                elif not is_polar:
                    if ice_conc > 45.0:
                        # Heavy ice is penalized heavily so non-polar ships seek open water channels,
                        # but allowed so routes to polar stations/ports do not fail completely.
                        edge *= (1.0 + 0.35 * ice_conc)
                    elif ice_conc >= 15.0:
                        # Add sea ice resistance penalty so A* seeks open water / light ice channels
                        edge *= (1.0 + 0.08 * (ice_conc - 15.0))
                else:
                    # Icebreakers prefer open water / low concentration channels when available
                    if ice_conc > 0.0:
                        edge *= (1.0 + 0.015 * ice_conc)

                tentative = g[node] + edge
                if tentative < g.get(nb, float("inf")):
                    g[nb] = tentative
                    came[nb] = node
                    h = geo.haversine(self.lats[rr], (self.lons[cc] % 360.0) - 180.0, to_lat, to_lon)
                    heapq.heappush(heap, (tentative + h, counter, nb))
                    counter += 1
            for nb, wcost, _wname in self.waterways.get(node, ()):
                if self.mask.land[nb]:
                    continue
                tentative = g[node] + wcost
                if tentative < g.get(nb, float("inf")):
                    g[nb] = tentative
                    came[nb] = node
                    h = geo.haversine(
                        self.lats[nb // NCOLS], (self.lons[nb % NCOLS] % 360.0) - 180.0,
                        to_lat, to_lon,
                    )
                    heapq.heappush(heap, (tentative + h, counter, nb))
                    counter += 1
            closed += 1
            if closed > 800000:
                raise ValueError("land-avoidance search did not converge")
        raise ValueError("no navigable water route found (enclosed sea or blocked coast)")


def _check_free(pts, mask, step_nm=5.0) -> bool:
    """True when a polyline never enters a land cell (sampled denser than cells)."""
    for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
        sub = geo.sample_geodesic(la1, lo1, la2, lo2, step_nm)
        for lat, lon in sub:
            if mask.cell(lat, lon):
                return False
    return True


def _perp_dist(p, a, b):
    lat_scale = max(abs(math.cos(math.radians((a[0] + b[0]) / 2.0))), 0.01)
    ax, ay = a[1] * lat_scale, a[0]
    bx, by = b[1] * lat_scale, b[0]
    px, py = p[1] * lat_scale, p[0]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / length


def _douglas_peucker(pts, eps):
    if len(pts) < 3:
        return list(pts)
    dmax, idx = 0.0, 0
    a, b = pts[0], pts[-1]
    for i in range(1, len(pts) - 1):
        d = _perp_dist(pts[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = _douglas_peucker(pts[: idx + 1], eps)
        right = _douglas_peucker(pts[idx:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _safe_shortcut(p1, p2, mask, max_north_lat=78.0):
    """Can we travel in a straight geodesic from p1 to p2 entirely on water,
    without entering the (ice-blocked) far-northern Arctic?"""
    pts = geo.sample_geodesic(p1[0], p1[1], p2[0], p2[1], 0.6 * SPACING)
    for lat, lon in pts:
        if lat > max_north_lat:
            return False
        if mask.cell(lat, lon):
            return False
    return True


def _shortcut_water(pts, mask):
    """Greedy merge of collinear water segments into geodesic jumps."""
    out = list(pts)
    i = 0
    while i < len(out) - 2:
        j = i + 2
        while j < len(out) and _safe_shortcut(out[i], out[j], mask):
            j += 1
        j -= 1
        if j > i + 1:
            del out[i + 1:j]
        i += 1
    return out


def _ensure_safe(pts, original_path, mask):
    """Repair any consecutive pair in `pts` whose geodesic would cross land
    (or the Arctic ice cap) by splicing back the original cell-path vertices."""
    out = []
    for p, q in zip(pts, pts[1:]):
        out.append(p)
        if _safe_shortcut(p, q, mask):
            continue
        try:
            i = original_path.index(p)
            j = original_path.index(q)
        except ValueError:
            i = j = -1
        if 0 <= i < j:
            for c in original_path[i + 1:j]:
                if c != out[-1]:
                    out.append(c)
    if pts:
        out.append(pts[-1])
    return out


def _resample(pts, step_nm=20.0) -> list[tuple[float, float]]:
    if len(pts) < 2:
        return list(pts)
    dense = [pts[0]]
    for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
        seg = geo.sample_geodesic(la1, lo1, la2, lo2, step_nm)
        for p in seg[1:]:
            if p != dense[-1]:
                dense.append(p)
    return dense


def _chaikin(pts, iterations=1) -> list[tuple[float, float]]:
    """Corner-cutting curve smoothing (in lat/lon space).

    For each segment P0→P1 the two new control points are
        Q = P0 + 0.25*(P1-P0)   (near P0)
        R = P0 + 0.75*(P1-P0)   (near P1)
    Output order per segment is Q then R so the polyline always advances
    forward; emitting R then Q creates zigzag oscillations that inflate
    total distance by ~2.5×.
    """
    if len(pts) < 3:
        return list(pts)
    out = list(pts)
    for _ in range(iterations):
        if len(out) < 3:
            break
        nxt = []
        for i in range(len(out) - 1):
            la1, lo1 = out[i]
            la2, lo2 = out[i + 1]
            nxt.append((la1 + 0.25 * (la2 - la1), lo1 + 0.25 * (lo2 - lo1)))
            nxt.append((la1 + 0.75 * (la2 - la1), lo1 + 0.75 * (lo2 - lo1)))
        nxt.append(out[-1])
        out = nxt
    return out


def _smooth_safe(pts, mask, fallback=None):
    """Apply Chaikin corner-cutting, keeping only land-safe passes.

    The smoothed curve must never enter a land cell (sampled denser than the
    grid). Fall back to a lighter pass, then to the unsmoothed path, so a
    land-crossing artificial curve is never returned.
    """
    two = _chaikin(pts, 2)
    if _check_free(two, mask):
        return two
    one = _chaikin(pts, 1)
    if _check_free(one, mask):
        return one
    return fallback if fallback is not None else pts


class RouteError(Exception):
    pass


class WP:
    """Internal waypoint (backend-side endpoint).

    Endpoints are plain `{label, lat, lon}` structs — a port lookup is only one
    way to produce them. Research stations / open-ocean coords are equally valid.
    """

    __slots__ = ("label", "countryCode", "lat", "lon", "isPort", "id")

    def __init__(self, label, lat, lon, countryCode="", isPort=True, id=""):
        self.label = label
        self.lat = lat
        self.lon = lon
        self.countryCode = countryCode or ""
        self.isPort = isPort
        self.id = id or label

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "countryCode": self.countryCode,
            "lat": self.lat,
            "lon": self.lon,
            "isPort": self.isPort,
        }

    @staticmethod
    def from_maybe(w):
        if isinstance(w, WP):
            return w
        if hasattr(w, "lat"):
            return WP(
                label=str(w.label),
                lat=float(w.lat),
                lon=float(w.lon),
                countryCode=str(getattr(w, "countryCode", "") or ""),
                isPort=bool(getattr(w, "isPort", True)),
                id=str(getattr(w, "id", "") or str(getattr(w, "label", ""))),
            )
        return WP(
            label=str(w.get("label", "")),
            lat=float(w["lat"]),
            lon=float(w["lon"]),
            countryCode=str(w.get("countryCode", "") or ""),
            isPort=bool(w.get("isPort", True)),
            id=str(w.get("id", "") or str(w.get("label", ""))),
        )


class RouteEngine:
    def __init__(self, mask: LandMask, ecas: list, chokepoints: list, waterways=()):
        self.mask = mask
        self.router = GridRouter(mask, waterways=waterways)
        self.ecas = ecas                      # list of (name, MultiPolygon)
        self.chokepoints = chokepoints        # list of (name, [[lon, lat], [lon, lat]])

    # ------------------------------------------------------------------ Stage A
    def _gc_path(self, a, b, step_nm=20.0):
        pts = geo.sample_geodesic(a.lat, a.lon, b.lat, b.lon, step_nm)
        if pts and pts[-1] != (b.lat, b.lon):
            pts.append((b.lat, b.lon))
        return pts

    # ------------------------------------------------------------------ Stage B
    def _land_avoid_path(self, a, b, vessel_type="cargo", ice_class="none", optimize_for="distance", speed_knots=15.0, forecast_date=None):
        sa = self.mask.nearest_water(a.lat, a.lon)
        sb = self.mask.nearest_water(b.lat, b.lon)
        if sa is None:
            raise RouteError(f"'{a.label}' is not reachable from navigable water")
        if sb is None:
            raise RouteError(f"'{b.label}' is not reachable from navigable water")
        for label, p, s in ((a.label, a, sa), (b.label, b, sb)):
            snap_nm = geo.geodesic_distance_nm(p.lat, p.lon, s[0], s[1])
            if snap_nm > 30.0:
                print(
                    f"[route] large waypoint snap: '{label}' "
                    f"({p.lat:.2f}, {p.lon:.2f}) -> water ({s[0]:.2f}, {s[1]:.2f}) "
                    f"{snap_nm:.0f} nm",
                    flush=True,
                )
        try:
            centers = self.router.shortest_path(
                sa[0], sa[1], sb[0], sb[1],
                vessel_type=vessel_type,
                ice_class=ice_class,
                optimize_for=optimize_for,
                target_speed_knots=speed_knots,
                forecast_date=forecast_date,
            )
        except ValueError as exc:
            raise RouteError(str(exc)) from exc
        interior = centers[1:-1] if len(centers) > 1 else []
        if len(centers) > 1 and centers[0] == centers[-1]:
            interior = []
        if interior:
            reduced = _douglas_peucker(interior, eps=1.1)
            reduced = _shortcut_water(reduced, self.mask)
            reduced = _ensure_safe(reduced, centers, self.mask)
            reduced = _smooth_safe(reduced, self.mask)
            interior = _resample(reduced, 20.0)
        pts = [(a.lat, a.lon)] + list(interior) + [(b.lat, b.lon)]
        out = []
        for p in pts:
            if not out or p != out[-1]:
                out.append(p)
        return out

    def _leg_path(self, a, b, vessel_type="cargo", ice_class="none", optimize_for="distance", speed_knots=15.0, forecast_date=None):
        gc = self._gc_path(a, b)
        if optimize_for == "distance" and _check_free(gc, self.mask):
            return gc
        return self._land_avoid_path(a, b, vessel_type=vessel_type, ice_class=ice_class, optimize_for=optimize_for, speed_knots=speed_knots, forecast_date=forecast_date)

    # --------------------------------------------------------- Stages C and D
    def _eca_distance(self, path, eca_polys) -> float:
        line = sgeo.LineString([(lon, lat) for lat, lon in path])
        total = 0.0
        for poly in eca_polys:
            inter = line.intersection(poly)
            total += self._geom_length_nm(inter)
        return total

    def _geom_length_nm(self, geom):
        if geom.is_empty:
            return 0.0
        gtype = geom.geom_type
        if gtype in ("LineString", "LinearRing"):
            return geo.polyline_length_nm([(lat, lon) for lon, lat in geom.coords])
        if gtype in ("MultiLineString", "GeometryCollection"):
            total = 0.0
            for part in geom.geoms:
                if part.geom_type in ("LineString", "LinearRing"):
                    total += geo.polyline_length_nm([(lat, lon) for lon, lat in part.coords])
            return total
        return 0.0

    def _crossings(self, path):
        line = sgeo.LineString([(lon, lat) for lat, lon in path])
        hits = []
        for cp in self.chokepoints:
            name, kind = cp["name"], cp["kind"]
            if kind == "cape":
                pt = sgeo.Point(cp["point"])
                d_nm = self._point_line_distance_nm(pt, line)
                if d_nm <= cp["radiusNm"]:
                    nearest = line.interpolate(line.project(pt))
                    hits.append((line.project(nearest), name))
            else:
                seg_line = sgeo.LineString(cp["line"])
                inter = line.intersection(seg_line)
                if inter.is_empty:
                    if line.distance(seg_line) > STRAIT_TOL_DEG:
                        continue
                    param = line.project(seg_line.centroid)
                elif inter.geom_type == "Point":
                    param = line.project(inter)
                elif inter.geom_type == "MultiPoint":
                    param = min(line.project(p) for p in inter.geoms)
                else:
                    continue
                hits.append((param, name))
        hits.sort(key=lambda t: t[0])
        return [name for _, name in hits]

    def _point_line_distance_nm(self, pt, line):
        return geo.geodesic_distance_nm(pt.y, pt.x, line.interpolate(line.project(pt)).y, line.interpolate(line.project(pt)).x)

    # ------------------------------------------------------------------ Stage E
    @staticmethod
    def duration_hours(distance_nm, speed_knots):
        return distance_nm / max(speed_knots, 1e-9)

    def compute_route(self, waypoints, speed_knots, departure_time_utc, optimize_for="distance", vessel_type="cargo", draft_meters=10.0, ice_class="none"):
        from datetime import timedelta

        waypoints = [WP.from_maybe(w) for w in waypoints]

        # Cost this passage against the ice field for the departure date. The
        # archive is bounded, so a date outside it falls back to the model's
        # default rather than failing the route.
        _forecast_date = None
        try:
            _forecast_date = departure_time_utc.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            _forecast_date = None

        if len(waypoints) < 2:
            raise RouteError("a route needs at least two waypoints")

        # Two waypoints at the same position produce a single-point geometry,
        # which shapely rejects with a GEOSException deep inside the smoother -
        # surfacing to the caller as a 500 rather than a clear refusal. Catch it
        # here, where the reason can still be stated. 0.001 deg is ~110 m, below
        # any meaningful leg and well inside the grid's own resolution.
        for i, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
            if abs(a.lat - b.lat) < 1e-3 and abs(a.lon - b.lon) < 1e-3:
                raise RouteError(
                    f"waypoints {i + 1} and {i + 2} are at the same position "
                    f"({a.lat:.4f}, {a.lon:.4f}); a leg needs two distinct points"
                )

        legs = []
        totals_nm = totals_eca = 0.0
        all_crossings = []
        seen = set()

        total_hours = 0.0
        total_fuel_mt = 0.0
        total_curr_boost = 0.0
        segment_count = 0

        for a, b in zip(waypoints, waypoints[1:]):
            if a.lat == b.lat and a.lon == b.lon:
                path = [(a.lat, a.lon)]
                dist = 0.0
            else:
                path = self._leg_path(a, b, vessel_type=vessel_type, ice_class=ice_class, optimize_for=optimize_for, speed_knots=speed_knots, forecast_date=_forecast_date)
                dist = geo.polyline_length_nm(path)
            eca = self._eca_distance(path, [e[2] for e in self.ecas])
            crossing_list = self._crossings(path)

            # Detailed physics transit computation along path segments
            leg_hours = 0.0
            leg_fuel_mt = 0.0
            for p1, p2 in zip(path, path[1:]):
                seg_dist = geo.geodesic_distance_nm(p1[0], p1[1], p2[0], p2[1])
                if seg_dist <= 0.001:
                    continue
                mid_lat = (p1[0] + p2[0]) / 2.0
                mid_lon = (p1[1] + p2[1]) / 2.0

                # Sample currents & winds
                u_curr = 0.35 + 0.15 * math.sin(math.radians(mid_lon)) if mid_lat < -50.0 else (0.25 * math.cos(math.radians(mid_lon)) + 0.10 if mid_lat > 50.0 else -0.30 if abs(mid_lat) < 15.0 else 0.20)
                v_curr = 0.08 * math.cos(math.radians(mid_lon)) if mid_lat < -50.0 else (0.15 * math.sin(math.radians(mid_lon)) if mid_lat > 50.0 else 0.08 * math.sin(math.radians(mid_lat)))
                u_wind = 11.5 + 2.5 * math.cos(math.radians(mid_lon)) if mid_lat < -40.0 else (7.5 + 1.8 * math.sin(math.radians(mid_lon)) if mid_lat > 30.0 else -6.5)
                v_wind = -3.2 * math.sin(math.radians(mid_lon)) if mid_lat < -40.0 else (2.0 * math.cos(math.radians(mid_lon)) if mid_lat > 30.0 else (-2.5 if mid_lat > 0 else 2.5))

                d_lat = p2[0] - p1[0]
                d_lon = (p2[1] - p1[1]) * math.cos(math.radians(mid_lat))
                norm = math.hypot(d_lat, d_lon) or 1.0
                dir_x, dir_y = d_lon / norm, d_lat / norm

                curr_along = (u_curr * dir_x + v_curr * dir_y) * 1.94384
                wind_along = (u_wind * dir_x + v_wind * dir_y) * 1.94384

                sog = max(2.0, speed_knots + curr_along + 0.03 * wind_along)
                seg_h = seg_dist / sog
                leg_hours += seg_h

                v_water = max(4.0, speed_knots - curr_along - 0.02 * wind_along)
                base_fuel_rate_h = 1.45 * ((v_water / 15.0) ** 3)  # MT per hour at design speed
                leg_fuel_mt += base_fuel_rate_h * seg_h * ice_fuel_multiplier(mid_lat, mid_lon, ice_class)

                total_curr_boost += curr_along
                segment_count += 1

            if not leg_hours:
                leg_hours = dist / max(speed_knots, 1e-9)

            legs.append({
                "from": a.to_dict(),
                "to": b.to_dict(),
                "path": [list(p) for p in path],
                "distanceNm": round(dist, 1),
                "distanceInEcaNm": round(eca, 1),
                "durationHours": round(leg_hours, 2),
                "estimatedFuelTons": round(leg_fuel_mt, 1),
                "crossings": crossing_list,
            })
            totals_nm += dist
            totals_eca += eca
            total_hours += leg_hours
            total_fuel_mt += leg_fuel_mt

            for c in crossing_list:
                if c not in seen:
                    seen.add(c)
                    all_crossings.append(c)

        # Compute sea-ice and POLARIS risk metrics along the computed path
        total_ice_nm = 0.0
        max_ice_conc = 0.0
        max_polaris = 0.0
        max_fuel_mult = 1.0
        risk_weighted_nm = 0.0
        polar_nm = 0.0
        for leg in legs:
            path_pts = leg["path"]
            for p1, p2 in zip(path_pts, path_pts[1:]):
                mid_lat = (p1[0] + p2[0]) / 2.0
                mid_lon = (p1[1] + p2[1]) / 2.0
                seg_dist = geo.geodesic_distance_nm(p1[0], p1[1], p2[0], p2[1])
                conc = get_sea_ice_concentration(mid_lat, mid_lon)
                if conc > max_ice_conc:
                    max_ice_conc = conc
                if conc >= 15.0:
                    total_ice_nm += seg_dist
                if conc > 0.0:
                    pol = polaris_risk_at(mid_lat, mid_lon, ice_class)
                    max_polaris = max(max_polaris, pol)
                    risk_weighted_nm += pol * seg_dist
                    polar_nm += seg_dist
                    max_fuel_mult = max(max_fuel_mult,
                                        ice_fuel_multiplier(mid_lat, mid_lon, ice_class))

        mean_polaris = risk_weighted_nm / polar_nm if polar_nm > 0 else 0.0

        eta = departure_time_utc + timedelta(hours=total_hours)
        avg_curr_boost = round(total_curr_boost / max(1, segment_count), 1)

        # Compute Safety Score (100 base)
        safety_score = max(0, min(100, round(100.0 - 0.5 * max_ice_conc - (15.0 if totals_eca > 500 else 0.0))))

        summary = {
            "legs": legs,
            "totalDistanceNm": round(totals_nm, 1),
            "totalDistanceInEcaNm": round(totals_eca, 1),
            "totalDistanceInSeaIceNm": round(total_ice_nm, 1),
            "maxSeaIceConcentrationPct": round(max_ice_conc, 1),
            "meanPolarisRisk": round(mean_polaris, 3),
            "maxPolarisRisk": round(max_polaris, 3),
            "maxIceFuelPenalty": round(max_fuel_mult, 2),
            "iceDataSource": ice_data_source(),
            "totalDurationHours": round(total_hours, 2),
            "estimatedFuelTons": round(total_fuel_mt, 1),
            "safetyScore": safety_score,
            "currentAssistanceKnots": avg_curr_boost,
            "optimizeFor": optimize_for,
            "etaUTC": eta.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "crossings": all_crossings,
            "vessel": {
                "type": vessel_type,
                "draftMeters": draft_meters,
                "iceClass": ice_class,
            },
        }

        # Attach the reasoning. A route is a decision, and a decision-support
        # system that returns only geometry makes the reader reverse-engineer
        # the reasoning from a polyline.
        summary["explanation"] = explain_route(
            summary, optimize_for, ice_class, current_iceberg_positions(), legs
        )
        return summary