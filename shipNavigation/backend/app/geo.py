"""Pure-python geodesic math on the WGS84 ellipsoid.

- vincenty_inverse : distance + initial/final azimuths between two points
- vincenty_direct  : propagate a point by a distance along an azimuth
- sample_geodesic  : dense polyline between two points (used by Stage A)
- haversine        : fast spherical distance (used for A* heuristics)

No external deps here so the routing math is self-contained.
"""

from __future__ import annotations

import math

A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)
NM_TO_M = 1852.0


def _norm_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def vincenty_inverse(lat1: float, lon1: float, lat2: float, lon2: float):
    """Return (initial_azimuth_deg, final_azimuth_deg, distance_m)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam = math.radians(_norm_lon(lon2 - lon1))

    u1 = math.atan((1.0 - F) * math.tan(phi1))
    u2 = math.atan((1.0 - F) * math.tan(phi2))
    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)

    lam_prev = lam
    sin_sigma = cos_sigma = sigma = sin_alpha = cos_sq_alpha = cos_2_sigma_m = delta_sigma = 0.0
    converged = False
    for _ in range(200):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2
        )
        if sin_sigma < 1e-14:
            return 0.0, 0.0, 0.0
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = (cos_u1 * cos_u2 * sin_lam) / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha
        cos_2_sigma_m = cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha if cos_sq_alpha else 0.0
        cc = (F / 16.0) * cos_sq_alpha * (4.0 + F * (4.0 - 3.0 * cos_sq_alpha))
        lam_prev = lam
        lam = math.radians(_norm_lon(lon2 - lon1)) + (1.0 - cc) * F * sin_alpha * (
            sigma
            + cc
            * sin_sigma
            * (cos_2_sigma_m + cc * cos_sigma * (-1.0 + 2.0 * cos_2_sigma_m**2))
        )
        if abs(lam - lam_prev) < 1e-12:
            converged = True
            break

    if not converged:
        # Antipodal / degenerate: fall back to haversine on a sphere.
        s = haversine(lat1, lon1, lat2, lon2)
        az1 = math.degrees(math.atan2(
            math.sin(lam) * cos_u2,
            cos_u1 * sin_u2 - sin_u1 * cos_u2 * math.cos(lam),
        ))
        az2 = math.degrees(math.atan2(
            math.sin(lam) * cos_u1,
            -sin_u1 * cos_u2 + cos_u1 * sin_u2 * math.cos(lam),
        ))
        return az1 % 360.0, az2 % 360.0, s

    u_sq = cos_sq_alpha * (A * A - B * B) / (B * B)
    aa = 1.0 + (u_sq / 16384.0) * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    bb = (u_sq / 1024.0) * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))
    delta_sigma = bb * sin_sigma * (
        cos_2_sigma_m
        + (bb / 4.0)
        * (
            cos_sigma * (-1.0 + 2.0 * cos_2_sigma_m**2)
            - (bb / 6.0) * cos_2_sigma_m * (-3.0 + 4.0 * sin_sigma**2) * (-3.0 + 4.0 * cos_2_sigma_m**2)
        )
    )
    s = B * aa * (sigma - delta_sigma)

    az1 = math.degrees(math.atan2(
        cos_u2 * math.sin(lam_prev), cos_u1 * sin_u2 - sin_u1 * cos_u2 * math.cos(lam_prev)
    ))
    az2 = math.degrees(math.atan2(
        cos_u1 * math.sin(lam_prev), -sin_u1 * cos_u2 + cos_u1 * sin_u2 * math.cos(lam_prev)
    ))
    return az1 % 360.0, az2 % 360.0, s


def vincenty_direct(lat1: float, lon1: float, az1: float, dist_m: float) -> tuple[float, float]:
    """Given start + azimuth + distance, return (lat2, lon2)."""
    if dist_m <= 0.0:
        return lat1, lon1
    phi1 = math.radians(lat1)
    lam1 = math.radians(lon1)
    alpha1 = math.radians(az1)

    sin_a1, cos_a1 = math.sin(alpha1), math.cos(alpha1)
    tan_u1 = (1.0 - F) * math.tan(phi1)
    cos_u1 = 1.0 / math.sqrt(1.0 + tan_u1 * tan_u1)
    sin_u1 = tan_u1 * cos_u1
    sigma1 = math.atan2(tan_u1, cos_a1)
    sin_alpha = cos_u1 * sin_a1
    cos_sq_alpha = 1.0 - sin_alpha * sin_alpha
    u_sq = cos_sq_alpha * (A * A - B * B) / (B * B)
    aa = 1.0 + (u_sq / 16384.0) * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    bb = (u_sq / 1024.0) * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))

    sigma = dist_m / (B * aa)
    sin_sigma_cos = cos_2_sigma_m = 0.0
    for _ in range(200):
        cos_2_sigma_m = math.cos(2.0 * sigma1 + sigma)
        sin_sigma, cos_sigma = math.sin(sigma), math.cos(sigma)
        delta_sigma = bb * sin_sigma * (
            cos_2_sigma_m
            + (bb / 4.0)
            * (
                cos_sigma * (-1.0 + 2.0 * cos_2_sigma_m**2)
                - (bb / 6.0) * cos_2_sigma_m * (-3.0 + 4.0 * sin_sigma**2) * (-3.0 + 4.0 * cos_2_sigma_m**2)
            )
        )
        sigma_new = dist_m / (B * aa) + delta_sigma
        if abs(sigma_new - sigma) < 1e-12:
            sigma = sigma_new
            break
        sigma = sigma_new

    sin_sigma, cos_sigma = math.sin(sigma), math.cos(sigma)
    xx = sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_a1
    phi2 = math.atan2(
        sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_a1,
        (1.0 - F) * math.sqrt(sin_alpha * sin_alpha + xx * xx),
    )
    lam = math.atan2(sin_sigma * sin_a1, cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_a1)
    cc = (F / 16.0) * cos_sq_alpha * (4.0 + F * (4.0 - 3.0 * cos_sq_alpha))
    ll = lam - (1.0 - cc) * F * sin_alpha * (
        sigma
        + cc
        * sin_sigma
        * (cos_2_sigma_m + cc * cos_sigma * (-1.0 + 2.0 * cos_2_sigma_m**2))
    )
    lon2 = math.degrees(lam1 + ll)
    return math.degrees(phi2), _norm_lon(lon2)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres on a mean-radius sphere."""
    r = 6371009.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(_norm_lon(lon2 - lon1))
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.asin(math.sqrt(min(1.0, h)))


def geodesic_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, _, m = vincenty_inverse(lat1, lon1, lat2, lon2)
    return m / NM_TO_M


def sample_geodesic(lat1: float, lon1: float, lat2: float, lon2: float, step_nm: float) -> list[tuple[float, float]]:
    """Dense polyline along the geodesic between two points, spacing ~step_nm."""
    az1, _, dist_m = vincenty_inverse(lat1, lon1, lat2, lon2)
    if dist_m <= 1.0:
        return [(lat1, lon1)]
    step_m = step_nm * NM_TO_M
    n = max(1, int(math.ceil(dist_m / step_m)))
    pts = [(lat1, lon1)]
    for i in range(1, n):
        d = dist_m * i / n
        lat, lon = vincenty_direct(lat1, lon1, az1, d)
        pts.append((lat, lon))
    pts.append((lat2, lon2))
    return pts


def polyline_length_nm(pts: list[tuple[float, float]]) -> float:
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
        total += geodesic_distance_nm(la1, lo1, la2, lo2)
    return total