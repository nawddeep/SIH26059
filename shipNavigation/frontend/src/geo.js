// Geodesic math on the WGS84 ellipsoid — pure JS port of backend/app/geo.py.
// No dependencies; mirrors the backend so route length and the drawn line
// always use the same formulas.

export const A = 6378137.0
export const F = 1.0 / 298.257223563
export const B = A * (1.0 - F)
export const NM_TO_M = 1852.0
export const MEAN_R = 6371009.0
const DEG = Math.PI / 180.0

export function normLon(lon) { return ((lon + 180) % 360 + 360) % 360 - 180 }

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

function fringe(deg) { return ((deg % 360) + 360) % 360 }

// Return { az1, az2, distanceM } (azimuths in degrees, 0..360).
export function vincentyInverse(lat1, lon1, lat2, lon2) {
  const phi1 = lat1 * DEG
  const phi2 = lat2 * DEG
  const lam0 = normLon(lon2 - lon1) * DEG

  const u1 = Math.atan((1.0 - F) * Math.tan(phi1))
  const u2 = Math.atan((1.0 - F) * Math.tan(phi2))
  const sinU1 = Math.sin(u1); const cosU1 = Math.cos(u1)
  const sinU2 = Math.sin(u2); const cosU2 = Math.cos(u2)

  let lam = lam0
  let lamPrev = lam
  let sinSigma = 0; let cosSigma = 0; let sigma = 0; let sinAlpha = 0
  let cosSqAlpha = 0; let cos2SigmaM = 0; let deltaSigma = 0
  let converged = false

  for (let it = 0; it < 200; it += 1) {
    const sinLam = Math.sin(lam)
    const cosLam = Math.cos(lam)
    sinSigma = Math.sqrt((cosU2 * sinLam) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2)
    if (sinSigma < 1e-14) return { az1: 0, az2: 0, distanceM: 0 }
    cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
    sigma = Math.atan2(sinSigma, cosSigma)
    sinAlpha = (cosU1 * cosU2 * sinLam) / sinSigma
    cosSqAlpha = 1.0 - sinAlpha * sinAlpha
    cos2SigmaM = cosSqAlpha ? cosSigma - (2.0 * sinU1 * sinU2) / cosSqAlpha : 0.0
    const cc = (F / 16.0) * cosSqAlpha * (4.0 + F * (4.0 - 3.0 * cosSqAlpha))
    lamPrev = lam
    lam = lam0 + (1.0 - cc) * F * sinAlpha * (
      sigma + cc * sinSigma * (cos2SigmaM + cc * cosSigma * (-1.0 + 2.0 * cos2SigmaM ** 2))
    )
    if (Math.abs(lam - lamPrev) < 1e-12) { converged = true; break }
  }

  if (!converged) {
    const s = haversine(lat1, lon1, lat2, lon2)
    const az1 = Math.atan2(Math.sin(lam) * cosU2, cosU1 * sinU2 - sinU1 * cosU2 * Math.cos(lam)) / DEG
    const az2 = Math.atan2(Math.sin(lam) * cosU1, -sinU1 * cosU2 + cosU1 * sinU2 * Math.cos(lam)) / DEG
    return { az1: fringe(az1), az2: fringe(az2), distanceM: s }
  }

  const uSq = cosSqAlpha * (A * A - B * B) / (B * B)
  const aa = 1.0 + (uSq / 16384.0) * (4096.0 + uSq * (-768.0 + uSq * (320.0 - 175.0 * uSq)))
  const bb = (uSq / 1024.0) * (256.0 + uSq * (-128.0 + uSq * (74.0 - 47.0 * uSq)))
  deltaSigma = bb * sinSigma * (
    cos2SigmaM
    + (bb / 4.0) * (
      cosSigma * (-1.0 + 2.0 * cos2SigmaM ** 2)
      - (bb / 6.0) * cos2SigmaM * (-3.0 + 4.0 * sinSigma ** 2) * (-3.0 + 4.0 * cos2SigmaM ** 2)
    )
  )
  const s = B * aa * (sigma - deltaSigma)

  const az1 = Math.atan2(cosU2 * Math.sin(lamPrev), cosU1 * sinU2 - sinU1 * cosU2 * Math.cos(lamPrev)) / DEG
  const az2 = Math.atan2(cosU1 * Math.sin(lamPrev), -sinU1 * cosU2 + cosU1 * sinU2 * Math.cos(lamPrev)) / DEG
  return { az1: fringe(az1), az2: fringe(az2), distanceM: s }
}

// Return [lat2, lon2] from start + initial azimuth + distance.
export function vincentyDirect(lat1, lon1, az1Deg, distM) {
  if (distM <= 0.0) return [lat1, lon1]
  const phi1 = lat1 * DEG
  const lam1 = lon1 * DEG
  const alpha1 = az1Deg * DEG

  const sinA1 = Math.sin(alpha1); const cosA1 = Math.cos(alpha1)
  const tanU1 = (1.0 - F) * Math.tan(phi1)
  const cosU1 = 1.0 / Math.sqrt(1.0 + tanU1 * tanU1)
  const sinU1 = tanU1 * cosU1
  const sigma1 = Math.atan2(tanU1, cosA1)
  const sinAlpha = cosU1 * sinA1
  const cosSqAlpha = 1.0 - sinAlpha * sinAlpha
  const uSq = cosSqAlpha * (A * A - B * B) / (B * B)
  const aa = 1.0 + (uSq / 16384.0) * (4096.0 + uSq * (-768.0 + uSq * (320.0 - 175.0 * uSq)))
  const bb = (uSq / 1024.0) * (256.0 + uSq * (-128.0 + uSq * (74.0 - 47.0 * uSq)))

  let sigma = distM / (B * aa)
  let cos2SigmaM = 0.0
  for (let it = 0; it < 200; it += 1) {
    cos2SigmaM = Math.cos(2.0 * sigma1 + sigma)
    const ss = Math.sin(sigma); const cs = Math.cos(sigma)
    const ds = bb * ss * (
      cos2SigmaM
      + (bb / 4.0) * (
        cs * (-1.0 + 2.0 * cos2SigmaM ** 2)
        - (bb / 6.0) * cos2SigmaM * (-3.0 + 4.0 * ss ** 2) * (-3.0 + 4.0 * cos2SigmaM ** 2)
      )
    )
    const sigmaNew = distM / (B * aa) + ds
    if (Math.abs(sigmaNew - sigma) < 1e-12) { sigma = sigmaNew; break }
    sigma = sigmaNew
  }

  const ss = Math.sin(sigma); const cs = Math.cos(sigma)
  const xx = sinU1 * ss - cosU1 * cs * cosA1
  const phi2 = Math.atan2(
    sinU1 * cs + cosU1 * ss * cosA1,
    (1.0 - F) * Math.sqrt(sinAlpha * sinAlpha + xx * xx),
  )
  const lam = Math.atan2(ss * sinA1, cosU1 * cs - sinU1 * ss * cosA1)
  const cc = (F / 16.0) * cosSqAlpha * (4.0 + F * (4.0 - 3.0 * cosSqAlpha))
  const ll = lam - (1.0 - cc) * F * sinAlpha * (
    sigma
    + cc * ss * (cos2SigmaM + cc * cs * (-1.0 + 2.0 * cos2SigmaM ** 2))
  )
  return [phi2 / DEG, normLon((lam1 + ll) / DEG)]
}

export function haversine(lat1, lon1, lat2, lon2) {
  const p1 = lat1 * DEG; const p2 = lat2 * DEG
  const dp = p2 - p1
  const dl = normLon(lon2 - lon1) * DEG
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2
  return 2.0 * MEAN_R * Math.asin(Math.sqrt(clamp(h, 0, 1)))
}

export function geodesicDistanceNm(lat1, lon1, lat2, lon2) {
  return vincentyInverse(lat1, lon1, lat2, lon2).distanceM / NM_TO_M
}

// Dense polyline along the geodesic between two points, spaced ~stepNm apart
// (default 30nm). Endpoints are exact; the last one is exactly [lat2, lon2].
export function sampleGeodesic(lat1, lon1, lat2, lon2, stepNm = 30) {
  const { az1, distanceM } = vincentyInverse(lat1, lon1, lat2, lon2)
  if (distanceM <= 1.0) return [[lat1, lon1]]
  const stepM = stepNm * NM_TO_M
  const n = Math.max(1, Math.ceil(distanceM / stepM))
  const pts = [[lat1, lon1]]
  for (let i = 1; i < n; i += 1) {
    const d = distanceM * i / n
    pts.push(vincentyDirect(lat1, lon1, az1, d))
  }
  pts.push([lat2, lon2])
  return pts
}

// Length of a polyline in nautical miles, measured with the same geodesic math
// that generates the line — so the number matches what is drawn.
export function polylineLengthNm(points) {
  let total = 0.0
  for (let i = 1; i < points.length; i += 1) {
    const [la1, lo1] = points[i - 1]
    const [la2, lo2] = points[i]
    total += geodesicDistanceNm(la1, lo1, la2, lo2)
  }
  return total
}

// Convert a [[lat, lon], ...] polyline into continuous longitudes (can exceed
// +/-180) so a great circle that crosses the antimeridian stays unbroken.
// Renders correctly after splitIntoRenderedParts().
export function unwrapLongitudes(points) {
  const out = [[points[0][0], points[0][1]]]
  let carry = 0
  for (let i = 1; i < points.length; i += 1) {
    let lon = points[i][1] + carry
    let d = lon - out[i - 1][1]
    while (d > 180) { carry -= 360; lon -= 360; d = lon - out[i - 1][1] }
    while (d < -180) { carry += 360; lon += 360; d = lon - out[i - 1][1] }
    out.push([points[i][0], lon])
  }
  return out
}

// Split continuous-longitude points into renderable LineString parts, each in
// the [-180, 180] world copy, breaking exactly where the route crosses the
// antimeridian. Returns [[[lon, lat], ...], ...] (Note: lon-lat order).
export function splitIntoRenderedParts(points) {
  const parts = []
  let part = []
  let shift = 0
  for (const [lat, lon] of points) {
    let lr = lon - shift
    let nextShift = shift
    if (lr > 180) nextShift += 360
    else if (lr <= -180) nextShift -= 360
    if (nextShift !== shift) {
      if (part.length) parts.push(part)
      part = []
      shift = nextShift
      lr = lon - shift
    }
    part.push([lr, lat])
  }
  if (part.length) parts.push(part)
  return parts
}