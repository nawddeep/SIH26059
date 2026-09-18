"""Backend Weather & Marine Forecasting Engine.

Integrates Open-Meteo Weather Forecast (16-day horizon) and Marine Weather (8-day horizon)
APIs with multi-coordinate batch requests. Route points beyond the forecast horizon fall back
to predict_weather_fallback(), an isolated climatological prediction module.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import math
import urllib.request
import urllib.error
from typing import Any, Optional

from . import geo

WEATHER_HORIZON_DAYS = 16
MARINE_HORIZON_DAYS = 8


def predict_weather_fallback(lat: float, lon: float, target_time: datetime) -> dict[str, Any]:
    """Modeled estimate for a point/time beyond the real forecast horizon.

    Returns the same shape as a real forecast sample (windSpeedKn, waveHeightM,
    hazardLevel, etc.) so callers don't need to branch on data source internally.
    """
    abs_lat = abs(lat)

    # Climatological wind and wave modeling based on latitude bands
    if lat < -40.0:
        # Roaring Forties / Southern Ocean: High winds & heavy seas
        wind_speed_kn = round(22.0 + 6.0 * math.cos(math.radians(lon)), 1)
        wind_gusts_kn = round(wind_speed_kn + 8.0, 1)
        wave_height_m = round(4.2 + 1.2 * math.sin(math.radians(lon)), 1)
        wave_period_s = 9.5
        weather_code = 3  # Overcast / windy
    elif abs_lat > 60.0:
        # High Polar
        wind_speed_kn = round(16.0 + 4.0 * math.sin(math.radians(lon)), 1)
        wind_gusts_kn = round(wind_speed_kn + 5.0, 1)
        wave_height_m = round(2.5 + 0.8 * math.cos(math.radians(lon)), 1)
        wave_period_s = 7.0
        weather_code = 71  # Snow / polar chill
    elif abs_lat < 30.0:
        # Equatorial / Trade Winds: Generally calmer seas
        wind_speed_kn = round(11.0 + 3.0 * math.sin(math.radians(lat)), 1)
        wind_gusts_kn = round(wind_speed_kn + 4.0, 1)
        wave_height_m = round(1.4 + 0.4 * math.cos(math.radians(lon)), 1)
        wave_period_s = 6.0
        weather_code = 1  # Mainly clear
    else:
        # Mid-Latitudes
        wind_speed_kn = round(15.0 + 5.0 * math.cos(math.radians(lat)), 1)
        wind_gusts_kn = round(wind_speed_kn + 6.0, 1)
        wave_height_m = round(2.2 + 0.6 * math.sin(math.radians(lon)), 1)
        wave_period_s = 7.5
        weather_code = 2  # Partly cloudy

    # Determine hazard level
    if wind_speed_kn >= 34.0 or wind_gusts_kn >= 40.0 or wave_height_m >= 5.0 or weather_code in (95, 96, 99):
        hazard_level = "severe"
    elif wind_speed_kn >= 22.0 or wind_gusts_kn >= 28.0 or wave_height_m >= 3.0:
        hazard_level = "moderate"
    else:
        hazard_level = "none"

    return {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "timestamp": target_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataSource": "modeled_estimate",
        "windSpeedKn": wind_speed_kn,
        "windGustsKn": wind_gusts_kn,
        "waveHeightM": wave_height_m,
        "wavePeriodS": wave_period_s,
        "weatherCode": weather_code,
        "hazardLevel": hazard_level,
    }


def _http_get_json(url: str, timeout: float = 5.0) -> Optional[Any]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ShipRoutePlanner/1.0 (Python)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                return json.loads(body)
    except Exception as exc:
        print(f"[weather_engine] HTTP request failed for {url[:60]}...: {exc}", flush=True)
    return None


def _find_closest_hour_index(hourly_time_list: list[str], target_dt: datetime) -> int:
    target_str = target_dt.strftime("%Y-%m-%dT%H:00")
    if target_str in hourly_time_list:
        return hourly_time_list.index(target_str)
    
    # Fallback to closest hour by string distance/timestamp
    target_ts = target_dt.timestamp()
    best_idx = 0
    best_diff = float("inf")
    for idx, t_str in enumerate(hourly_time_list):
        try:
            dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            diff = abs((dt.timestamp() - target_ts))
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        except Exception:
            continue
    return best_idx


def get_route_weather(
    route: dict[str, Any],
    departure_time_utc: datetime,
    step_hours: float = 4.0,
) -> dict[str, Any]:
    """Sample the route polyline in time and fetch/modeled forecast data for each point."""
    if not route or not route.get("legs"):
        return {"samples": []}

    total_dist_nm = float(route.get("totalDistanceNm", 0))
    total_dur_hours = float(route.get("totalDurationHours", 0))
    if total_dist_nm <= 0 or total_dur_hours <= 0:
        return {"samples": []}

    # Extract clean polyline of path points [(lat, lon)]
    pts = []
    for leg in route["legs"]:
        path = leg.get("path", [])
        for pt in path:
            if not pts or pts[-1] != (pt[0], pt[1]):
                pts.append((pt[0], pt[1]))

    if len(pts) < 2:
        return {"samples": []}

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    weather_cutoff = now_utc + timedelta(days=WEATHER_HORIZON_DAYS)
    marine_cutoff = now_utc + timedelta(days=MARINE_HORIZON_DAYS)

    # 1. Generate sample points along route timeline
    curr_hours = 0.0
    sample_points = []
    
    while curr_hours <= total_dur_hours + 0.1:
        elapsed = min(curr_hours, total_dur_hours)
        progress = elapsed / total_dur_hours
        target_dist = progress * total_dist_nm
        target_dt = departure_time_utc + timedelta(hours=elapsed)

        # Interpolate position along polyline at target_dist
        acc = 0.0
        sample_lat, sample_lon = pts[-1]
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            d = geo.geodesic_distance_nm(p1[0], p1[1], p2[0], p2[1])
            if acc + d >= target_dist or i == len(pts) - 2:
                frac = (target_dist - acc) / d if d > 1e-6 else 0.0
                frac = max(0.0, min(1.0, frac))
                sample_lat = p1[0] + frac * (p2[0] - p1[0])
                sample_lon = p1[1] + frac * (p2[1] - p1[1])
                break
            acc += d

        sample_points.append({
            "lat": round(sample_lat, 4),
            "lon": round(sample_lon, 4),
            "target_dt": target_dt,
            "elapsed_hours": round(elapsed, 1),
            "weather_in_horizon": target_dt <= weather_cutoff and target_dt >= (now_utc - timedelta(days=1)),
            "marine_in_horizon": target_dt <= marine_cutoff and target_dt >= (now_utc - timedelta(days=1)),
        })
        curr_hours += step_hours

    # 2. Group sample points in-horizon for Open-Meteo batch fetching
    weather_in_pts = [p for p in sample_points if p["weather_in_horizon"]]
    marine_in_pts = [p for p in sample_points if p["marine_in_horizon"]]

    weather_batch_data = None
    marine_batch_data = None

    if weather_in_pts:
        lats_str = ",".join(f"{p['lat']:.4f}" for p in weather_in_pts)
        lons_str = ",".join(f"{p['lon']:.4f}" for p in weather_in_pts)
        url_w = f"https://api.open-meteo.com/v1/forecast?latitude={lats_str}&longitude={lons_str}&hourly=wind_speed_10m,wind_gusts_10m,weather_code,precipitation,cape&wind_speed_unit=kn&forecast_days=16"
        res_w = _http_get_json(url_w)
        if res_w:
            weather_batch_data = res_w if isinstance(res_w, list) else [res_w]

    if marine_in_pts:
        lats_str_m = ",".join(f"{p['lat']:.4f}" for p in marine_in_pts)
        lons_str_m = ",".join(f"{p['lon']:.4f}" for p in marine_in_pts)
        url_m = f"https://marine-api.open-meteo.com/v1/marine?latitude={lats_str_m}&longitude={lons_str_m}&hourly=wave_height,wave_period,swell_wave_height&forecast_days=8"
        res_m = _http_get_json(url_m)
        if res_m:
            marine_batch_data = res_m if isinstance(res_m, list) else [res_m]

    # Map sample points to weather_batch index and marine_batch index
    w_index_map = {id(p): idx for idx, p in enumerate(weather_in_pts)}
    m_index_map = {id(p): idx for idx, p in enumerate(marine_in_pts)}

    # 3. Construct final sample records
    output_samples = []

    for p in sample_points:
        target_dt = p["target_dt"]
        lat, lon = p["lat"], p["lon"]

        # Check if we have valid Open-Meteo weather data for this point
        has_weather = False
        wind_speed_kn = 0.0
        wind_gusts_kn = 0.0
        weather_code = 0

        if p["weather_in_horizon"] and weather_batch_data:
            w_idx = w_index_map.get(id(p))
            if w_idx is not None and w_idx < len(weather_batch_data):
                w_item = weather_batch_data[w_idx]
                hourly = w_item.get("hourly", {})
                times = hourly.get("time", [])
                if times:
                    h_idx = _find_closest_hour_index(times, target_dt)
                    wind_speed_kn = round(float(hourly.get("wind_speed_10m", [0])[h_idx] or 0.0), 1)
                    wind_gusts_kn = round(float(hourly.get("wind_gusts_10m", [wind_speed_kn])[h_idx] or wind_speed_kn), 1)
                    weather_code = int(hourly.get("weather_code", [0])[h_idx] or 0)
                    has_weather = True

        # Check if we have valid Open-Meteo marine data for this point
        has_marine = False
        wave_height_m = 0.0
        wave_period_s = 6.0

        if p["marine_in_horizon"] and marine_batch_data:
            m_idx = m_index_map.get(id(p))
            if m_idx is not None and m_idx < len(marine_batch_data):
                m_item = marine_batch_data[m_idx]
                hourly_m = m_item.get("hourly", {})
                times_m = hourly_m.get("time", [])
                if times_m:
                    h_idx_m = _find_closest_hour_index(times_m, target_dt)
                    raw_wh = hourly_m.get("wave_height", [0.0])[h_idx_m]
                    raw_wp = hourly_m.get("wave_period", [6.0])[h_idx_m]
                    if raw_wh is not None:
                        wave_height_m = round(float(raw_wh), 1)
                        has_marine = True
                    if raw_wp is not None:
                        wave_period_s = round(float(raw_wp), 1)

        # Fallback values if outside horizon or API data missing
        fallback = predict_weather_fallback(lat, lon, target_dt)

        if not has_weather:
            wind_speed_kn = fallback["windSpeedKn"]
            wind_gusts_kn = fallback["windGustsKn"]
            weather_code = fallback["weatherCode"]

        if not has_marine:
            wave_height_m = fallback["waveHeightM"]
            wave_period_s = fallback["wavePeriodS"]

        # Data source tag: "forecast" if both weather & marine (or weather) came from live API, else "modeled_estimate"
        data_source = "forecast" if (has_weather or has_marine) else "modeled_estimate"

        # Determine hazard level
        if wind_speed_kn >= 34.0 or wind_gusts_kn >= 40.0 or wave_height_m >= 5.0 or weather_code in (95, 96, 99):
            hazard_level = "severe"
        elif wind_speed_kn >= 22.0 or wind_gusts_kn >= 28.0 or wave_height_m >= 3.0:
            hazard_level = "moderate"
        else:
            hazard_level = "none"

        has_rain = weather_code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 85, 86)
        has_storms = weather_code in (95, 96, 99) or wind_gusts_kn >= 34.0 or wind_speed_kn >= 34.0
        has_wind = wind_speed_kn >= 15.0 or wind_gusts_kn >= 22.0
        has_waves = wave_height_m >= 2.0
        has_hazard = hazard_level in ("moderate", "severe")

        output_samples.append({
            "lat": lat,
            "lon": lon,
            "timestamp": target_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsedHours": p["elapsed_hours"],
            "dataSource": data_source,
            "windSpeedKn": wind_speed_kn,
            "windGustsKn": wind_gusts_kn,
            "waveHeightM": wave_height_m,
            "wavePeriodS": wave_period_s,
            "weatherCode": weather_code,
            "hazardLevel": hazard_level,
            "hasWind": has_wind,
            "hasWaves": has_waves,
            "hasStorms": has_storms,
            "hasRain": has_rain,
            "hasHazard": has_hazard,
        })

    return {"samples": output_samples}
