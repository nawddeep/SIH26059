const API_BASE = '/api'

async function json(res) {
  const text = await res.text()
  let data
  try {
    data = JSON.parse(text)
  } catch {
    throw new Error(`Unexpected response (${res.status})`)
  }
  if (!res.ok) {
    const msg = (data && data.detail) ? (Array.isArray(data.detail) ? data.detail.map(d => d.msg).join(', ') : data.detail) : `Request failed (${res.status})`
    throw new Error(msg)
  }
  return data
}

export async function searchPorts(q) {
  const res = await fetch(`${API_BASE}/ports?q=${encodeURIComponent(q)}`)
  return json(res)
}

export async function geocode(lat, lon) {
  const res = await fetch(`${API_BASE}/geocode?lat=${lat}&lon=${lon}`)
  return json(res)
}

export async function computeRoute(waypoints, speedKnots, departureTimeUTC, optimizeFor, vesselType = 'cargo', draftMeters = 10.0, iceClass = 'none') {
  const res = await fetch(`${API_BASE}/route`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ waypoints, speedKnots, departureTimeUTC, optimizeFor, vesselType, draftMeters, iceClass }),
  })
  return json(res)
}

export async function fetchEcas() {
  const res = await fetch(`${API_BASE}/ecas`)
  return json(res)
}

export async function fetchChokepoints() {
  const res = await fetch(`${API_BASE}/chokepoints`)
  return json(res)
}

export async function fetchIcebergs(startTime = '', endTime = '', stepHours = 1.0) {
  const params = new URLSearchParams()
  if (startTime) params.set('startTime', startTime)
  if (endTime) params.set('endTime', endTime)
  if (stepHours) params.set('stepHours', String(stepHours))
  const res = await fetch(`${API_BASE}/icebergs?${params.toString()}`)
  return json(res)
}

export async function fetchSeaIce() {
  const res = await fetch(`${API_BASE}/sea-ice`)
  return json(res)
}

export async function fetchIceRisk(polarClass = 'PC4') {
  const res = await fetch(`${API_BASE}/ice-risk?polarClass=${encodeURIComponent(polarClass)}`)
  return json(res)
}

export async function fetchOceanCurrents() {
  const res = await fetch(`${API_BASE}/ocean-currents`)
  return json(res)
}

export async function fetchWind() {
  const res = await fetch(`${API_BASE}/wind`)
  return json(res)
}

export async function fetchWeatherHeatmap() {
  const res = await fetch(`${API_BASE}/weather-heatmap`)
  return json(res)
}

export async function fetchModelStatus() {
  const res = await fetch(`${API_BASE}/model-status`)
  return json(res)
}

export async function fetchRouteWeather(routeId = '', departureTime = '', speedKnots = 15) {
  const params = new URLSearchParams()
  if (routeId) params.set('routeId', routeId)
  if (departureTime) params.set('departureTime', departureTime)
  if (speedKnots) params.set('speedKnots', String(speedKnots))
  const res = await fetch(`${API_BASE}/route-weather?${params.toString()}`)
  return json(res)
}