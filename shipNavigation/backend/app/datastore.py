"""Static data services: ports DB, reverse geocoding, ECA polygons, chokepoints."""

from __future__ import annotations

import json
import os

import shapely.geometry as sgeo

from . import geo

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

PORTS_FILE = os.path.join(DATA, "ports.json")
ECAS_FILE = os.path.join(DATA, "ecas.json")
CHOKEPOINTS_FILE = os.path.join(DATA, "chokepoints.json")
WATERWAYS_FILE = os.path.join(DATA, "waterways.json")


class Datastore:
    def __init__(self):
        with open(PORTS_FILE) as f:
            self._entries = json.load(f)
        # index only places worth returning from autocomplete
        self.places = self._entries

        with open(ECAS_FILE) as f:
            eca_data = json.load(f)["ecas"]
        self.ecas = []
        for e in eca_data:
            polys = []
            for ring in e["polygons"]:
                p = sgeo.Polygon(ring)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.geom_type == "MultiPolygon":
                    polys.extend(list(p.geoms))
                elif not p.is_empty:
                    polys.append(p)
            mp = polys[0] if len(polys) == 1 else sgeo.MultiPolygon(polys)
            self.ecas.append((e["name"], {"type": e.get("type", "")}, mp))

        with open(CHOKEPOINTS_FILE) as f:
            cp_data = json.load(f)["chokepoints"]
        self.chokepoints = [
            {
                "name": c["name"],
                "kind": c.get("kind", "strait"),
                "line": [tuple(pt) for pt in c.get("line", [])],
                "point": tuple(c.get("point", [None, None])),
                "radiusNm": float(c.get("radiusNm", 120)),
            }
            for c in cp_data
        ]

        with open(WATERWAYS_FILE) as f:
            ww_data = json.load(f)["waterways"]
        self.waterways = []
        for w in ww_data:
            via = [tuple(p) for p in w.get("via", [])]
            self.waterways.append({
                "name": w["name"],
                "from": tuple(w["from"]),
                "to": tuple(w["to"]),
                "via": via,
            })

    # ------------------------------------------------------------- port search
    def search_ports(self, q: str, limit: int = 12) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []
        scored = []
        for e in self.places:
            name = e["name"]
            low = name.lower()
            aliases = [a.lower() for a in e.get("aliases", [])]
            if q in low or any(q in a for a in aliases):
                starts = low.startswith(q) or any(a.startswith(q) for a in aliases)
                scored.append((0 if starts else 1, len(low), e))
        scored.sort(key=lambda t: (t[0], t[1], t[2]["name"]))
        return [dict(e) for _, _, e in scored[:limit]]

    # ------------------------------------------------------------- reverse geo
    def reverse_geocode(self, lat: float, lon: float) -> dict:
        stations = [e for e in self.places if not e["isPort"]]
        best_station = None
        best_station_d = 900.0
        for s in stations:
            d = geo.geodesic_distance_nm(lat, lon, s["lat"], s["lon"])
            if d < best_station_d and d < 120.0:
                best_station_d = d
                best_station = s

        best_port = None
        best_port_d = 12.0
        for p in self.places:
            if not p["isPort"]:
                continue
            d = geo.geodesic_distance_nm(lat, lon, p["lat"], p["lon"])
            if d < best_port_d and d < 8.0:
                best_port_d = d
                best_port = p

        if best_port:
            return dict(best_port)
        if best_station:
            return dict(best_station)
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        label = f"{abs(lat):.1f}{ns} {abs(lon):.1f}{ew}"
        return {"name": label, "lat": round(lat, 4), "lon": round(lon, 4), "isPort": False, "countryCode": ""}

    # ------------------------------------------------------------ API helpers
    def ecas_geojson(self) -> dict:
        features = []
        for name, meta, mp in self.ecas:
            features.append({
                "type": "Feature",
                "properties": {"name": name, **meta},
                "geometry": sgeo.mapping(mp),
            })
        return {"type": "FeatureCollection", "features": features}

    def chokepoints_geojson(self) -> dict:
        features = []
        for cp in self.chokepoints:
            if cp["kind"] == "cape":
                pt = cp["point"]
                geometry = {"type": "Point", "coordinates": [pt[0], pt[1]]}
            else:
                geometry = {"type": "LineString", "coordinates": [list(pt) for pt in cp["line"]]}
            features.append({
                "type": "Feature",
                "properties": {"name": cp["name"], "kind": cp["kind"], "radiusNm": cp.get("radiusNm", 0)},
                "geometry": geometry,
            })
        return {"type": "FeatureCollection", "features": features}