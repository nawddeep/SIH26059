export function flagEmoji(countryCode = '') {
  if (!countryCode || countryCode.length !== 2) return '🌐'
  const cc = countryCode.toUpperCase()
  if (!/^[A-Z]{2}$/.test(cc)) return '🌐'
  const OFFSET = 127397
  return String.fromCodePoint(...[...cc].map(c => OFFSET + c.charCodeAt(0)))
}

const CTYPE = {
  sox: { value: 0.06, order: 0 },
  'sox, nox': { value: 0.03, order: 1 },
}

export function chipClass(t) {
  const k = (t || '').toLowerCase()
  return CTYPE[k] ? (CTYPE[k].value) : 1
}

// Parse "62°05'S 74°12'E", "62.1S 74.2E", "62.1, 74.2", "-62.1 74.2", etc.
export function parseCoordLabel(text) {
  const s = (text || '').trim()
  if (!s) return null

  const dms = /(-?\d+(?:\.\d+)?)\s*°?\s*(\d+(?:\.\d+)?)?\s*(?:'|′)?\s*(\d+(?:\.\d+)?)?\s*(?:''|″)?\s*([NSEW])?\s*[,;\s]+\s*(-?\d+(?:\.\d+)?)\s*°?\s*(\d+(?:\.\d+)?)?\s*(?:'|′)?\s*(\d+(?:\.\d+)?)?\s*(?:''|″)?\s*([NSEW])?/i.exec(s)
  if (dms) {
    const [, la, lam, las, laH, lo, lom, los, loH] = dms
    let lat = Number(la) + (Number(lam) || 0) / 60 + (Number(las) || 0) / 3600
    let lon = Number(lo) + (Number(lom) || 0) / 60 + (Number(los) || 0) / 3600
    if (laH && /[SW]/i.test(laH)) lat = -lat
    if (laH && /[NE]/i.test(laH)) lat = Math.abs(lat)
    if (loH && /[SW]/i.test(loH)) lon = -lon
    if (loH && /[NE]/i.test(loH)) lon = Math.abs(lon)
    if (!isNaN(lat) && !isNaN(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return { lat, lon }
  }

  const pair = /^([S±\-]?\s?\d+(?:\.\d+)?)\s*[,;\s]\s*([EW±\-]?\s?-?\d+(?:\.\d+)?)$/i.exec(s)
  if (pair) {
    let lat = parseFloat(pair[1].replace(/[SEWN±\s]/gi, ''))
    let lon = parseFloat(pair[2].replace(/[SEWN±\s]/gi, ''))
    if (/S/i.test(pair[1])) lat = -Math.abs(lat)
    if (/W/i.test(pair[2])) lon = -Math.abs(lon)
    if (!isNaN(lat) && !isNaN(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return { lat, lon }
  }
  return null
}

export function formatCoord(lat, lon) {
  const ns = lat >= 0 ? 'N' : 'S'
  const ew = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(2)}°${ns} ${Math.abs(lon).toFixed(2)}°${ew}`
}

export function fmtHours(hours) {
  const h = Math.floor(hours)
  const days = Math.floor(h / 24)
  const mins = Math.round((hours - Math.floor(hours)) * 60)
  if (days > 0) return `${days}d ${h % 24}h ${mins}m`
  if (h > 0) return `${h}h ${mins}m`
  return `${mins}m`
}

export function haversineNm(lat1, lon1, lat2, lon2) {
  const R_NM = 3440.065
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R_NM * c
}

export function getShipPositionAtTime(route, departureTimeISO, targetTimeISO) {
  if (!route || !route.legs || !route.legs.length) return null
  const startMs = new Date(departureTimeISO).getTime()
  const targetMs = new Date(targetTimeISO).getTime()
  const totalDurationHours = route.totalDurationHours || 1
  const totalDistNm = route.totalDistanceNm || 0

  const allPts = []
  for (const leg of route.legs) {
    for (const pt of leg.path) {
      if (!allPts.length || allPts[allPts.length - 1][0] !== pt[0] || allPts[allPts.length - 1][1] !== pt[1]) {
        allPts.push(pt)
      }
    }
  }
  if (!allPts.length) return null
  if (allPts.length === 1) return { lat: allPts[0][0], lon: allPts[0][1], progress: 0 }

  const elapsedHours = Math.max(0, (targetMs - startMs) / 3600000)
  const progress = Math.min(1.0, elapsedHours / totalDurationHours)
  const targetDistNm = progress * totalDistNm

  let accumulated = 0
  for (let i = 0; i < allPts.length - 1; i++) {
    const p1 = allPts[i]
    const p2 = allPts[i + 1]
    const segDist = haversineNm(p1[0], p1[1], p2[0], p2[1])
    if (accumulated + segDist >= targetDistNm || i === allPts.length - 2) {
      const segFrac = segDist > 1e-6 ? Math.min(1, Math.max(0, (targetDistNm - accumulated) / segDist)) : 0
      const lat = p1[0] + segFrac * (p2[0] - p1[0])
      const lon = p1[1] + segFrac * (p2[1] - p1[1])
      return { lat, lon, progress }
    }
    accumulated += segDist
  }

  const last = allPts[allPts.length - 1]
  return { lat: last[0], lon: last[1], progress: 1.0 }
}

export function getIcebergPositionAtTime(iceberg, targetTimeISO) {
  if (!iceberg || !iceberg.trajectory || !iceberg.trajectory.length) {
    const init = iceberg?.initialPosition || { lat: 0, lon: 0 }
    return { lat: init.lat, lon: init.lon }
  }
  const targetMs = new Date(targetTimeISO).getTime()
  const traj = iceberg.trajectory

  const t0Ms = new Date(traj[0].timestamp).getTime()
  if (targetMs <= t0Ms) return { lat: traj[0].lat, lon: traj[0].lon }

  const tEndMs = new Date(traj[traj.length - 1].timestamp).getTime()
  if (targetMs >= tEndMs) return { lat: traj[traj.length - 1].lat, lon: traj[traj.length - 1].lon }

  for (let i = 0; i < traj.length - 1; i++) {
    const p1 = traj[i]
    const p2 = traj[i + 1]
    const t1 = new Date(p1.timestamp).getTime()
    const t2 = new Date(p2.timestamp).getTime()
    if (targetMs >= t1 && targetMs <= t2) {
      const frac = t2 > t1 ? (targetMs - t1) / (t2 - t1) : 0
      const lat = p1.lat + frac * (p2.lat - p1.lat)
      const lon = p1.lon + frac * (p2.lon - p1.lon)
      return { lat, lon }
    }
  }

  const last = traj[traj.length - 1]
  return { lat: last.lat, lon: last.lon }
}

export function computeClosestApproaches(route, departureTimeISO, icebergs = []) {
  if (!route || !icebergs.length) return []
  const startMs = new Date(departureTimeISO).getTime()
  const totalHours = route.totalDurationHours || 24
  const endMs = startMs + totalHours * 3600000

  const results = []
  const stepMs = 30 * 60 * 1000 // 30-min steps

  for (const berg of icebergs) {
    let minDist = Infinity
    let closestTimeMs = startMs
    let closestShipPos = null
    let closestBergPos = null

    for (let t = startMs; t <= endMs; t += stepMs) {
      const tISO = new Date(t).toISOString()
      const sPos = getShipPositionAtTime(route, departureTimeISO, tISO)
      const bPos = getIcebergPositionAtTime(berg, tISO)
      if (sPos && bPos) {
        const d = haversineNm(sPos.lat, sPos.lon, bPos.lat, bPos.lon)
        if (d < minDist) {
          minDist = d
          closestTimeMs = t
          closestShipPos = sPos
          closestBergPos = bPos
        }
      }
    }

    if (minDist !== Infinity) {
      results.push({
        icebergId: berg.id,
        icebergName: berg.name || berg.id,
        minDistanceNm: Math.round(minDist * 10) / 10,
        timeISO: new Date(closestTimeMs).toISOString(),
        shipPos: closestShipPos,
        bergPos: closestBergPos,
      })
    }
  }

  results.sort((a, b) => a.minDistanceNm - b.minDistanceNm)
  return results
}