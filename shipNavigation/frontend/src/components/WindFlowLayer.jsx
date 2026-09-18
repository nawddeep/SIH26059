import React, { useEffect, useRef, useState } from 'react'
import { fetchWind } from '../api'

// Open-Meteo bills every COORDINATE as an API call, not every request, so a
// 20x13 grid cost 260 calls per pan and exhausted the hourly quota within a
// minute. The field is interpolated anyway, so a coarse grid costs little
// visually and buys roughly twenty times as many refreshes.
const GRID_COLS = 10
const GRID_ROWS = 6
const FETCH_DEBOUNCE_MS = 700
const MIN_FETCH_GAP_MS = 15000
// Panning inside the area already covered should not spend quota, so a refetch
// only happens once the view has left the fetched box or this much time passed.
const REFRESH_AFTER_MS = 10 * 60 * 1000

const MAX_AGE_FRAMES = 110
const TRAIL_FADE = 0.94
// Screen pixels a 20 m/s wind advances per frame. Motion is normalised to the
// viewport rather than to degrees: a fixed step in degrees is a couple of pixels
// when zoomed out and hundreds when zoomed in, which is what turned the field
// into solid streaks at close range.
const PX_PER_FRAME_AT_REF = 1.6
const REF_SPEED_MS = 20
// One streak per this many square pixels, so the field keeps the same visual
// density at every zoom instead of crowding as the view narrows.
const PX2_PER_PARTICLE = 1400
const MIN_PARTICLES = 350
const MAX_PARTICLES = 2200
// Heatmap is interpolated at this fraction of screen resolution and scaled up;
// the browser's bilinear filtering supplies the smooth gradient for free.
const FIELD_DOWNSCALE = 10

// Wind speed (km/h) -> colour, following the Zoom Earth scale: near-calm reads
// as deep indigo, trade-wind strength as teal/green, gale force as yellow-red.
const RAMP = [
  [0, [26, 18, 74]],
  [10, [40, 42, 120]],
  [20, [40, 84, 150]],
  [30, [38, 130, 152]],
  [45, [52, 168, 133]],
  [60, [120, 198, 98]],
  [75, [206, 222, 92]],
  [90, [244, 216, 78]],
  [105, [240, 160, 66]],
  [120, [226, 104, 58]],
  [150, [200, 60, 58]],
]

function rampColor(kmh) {
  if (kmh <= RAMP[0][0]) return RAMP[0][1]
  for (let i = 1; i < RAMP.length; i++) {
    if (kmh <= RAMP[i][0]) {
      const [x0, c0] = RAMP[i - 1]
      const [x1, c1] = RAMP[i]
      const t = (kmh - x0) / (x1 - x0)
      return [
        c0[0] + (c1[0] - c0[0]) * t,
        c0[1] + (c1[1] - c0[1]) * t,
        c0[2] + (c1[2] - c0[2]) * t,
      ]
    }
  }
  return RAMP[RAMP.length - 1][1]
}

function buildGrid(bounds) {
  const s = bounds.getSouth()
  const n = bounds.getNorth()
  const w = bounds.getWest()
  const e = bounds.getEast()
  // Inset slightly so edge samples are inside the view being interpolated.
  const latSpan = n - s
  const lonSpan = e - w
  const lats = []
  const lons = []
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const lat = s + (latSpan * r) / (GRID_ROWS - 1)
      const lon = w + (lonSpan * c) / (GRID_COLS - 1)
      lats.push(Math.max(-89.5, Math.min(89.5, lat)).toFixed(3))
      lons.push((((lon + 180) % 360 + 360) % 360 - 180).toFixed(3))
    }
  }
  return { lats, lons }
}

async function fetchWindGrid(bounds, signal) {
  const { lats, lons } = buildGrid(bounds)
  const url =
    'https://api.open-meteo.com/v1/forecast' +
    `?latitude=${lats.join(',')}` +
    `&longitude=${lons.join(',')}` +
    '&current=wind_speed_10m,wind_direction_10m' +
    '&wind_speed_unit=ms'

  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`Open-Meteo ${res.status}`)
  const body = await res.json()
  const rows = Array.isArray(body) ? body : [body]

  return rows
    .map((r) => {
      const cur = r.current
      if (!cur) return null
      const spd = cur.wind_speed_10m
      const dir = cur.wind_direction_10m
      if (typeof spd !== 'number' || typeof dir !== 'number') return null
      // Meteorological convention: direction is where the wind blows FROM, so
      // the motion vector is the negated components.
      const rad = (dir * Math.PI) / 180
      return {
        lat: r.latitude,
        lon: r.longitude,
        u: -spd * Math.sin(rad),
        v: -spd * Math.cos(rad),
        speed: spd,
      }
    })
    .filter(Boolean)
}

/**
 * The app's own ERA5 layer, used when Open-Meteo is unreachable or out of
 * quota. It only covers the Southern Ocean, but that is the operating area this
 * planner exists for, so a working Antarctic field beats an empty globe.
 */
async function fetchLocalWind() {
  const fc = await fetchWind()
  return (fc.features || [])
    .map((f) => ({
      lat: f.geometry.coordinates[1],
      lon: f.geometry.coordinates[0],
      u: f.properties.u,
      v: f.properties.v,
      speed: Math.hypot(f.properties.u, f.properties.v),
    }))
    .filter((p) => Number.isFinite(p.u) && Number.isFinite(p.v))
}

/**
 * Zoom Earth style wind layer: an interpolated speed field with animated
 * streaklines over it.
 *
 * Live global winds come straight from Open-Meteo rather than the app's own
 * /api/wind, which only covers the Southern Ocean reanalysis domain and would
 * leave the rest of the world blank.
 */
export default function WindFlowLayer({ map, visible }) {
  const fieldCanvasRef = useRef(null)
  const flowCanvasRef = useRef(null)
  const dataRef = useRef(null)
  const rafRef = useRef(0)
  const particlesRef = useRef([])
  const coveredRef = useRef(null)
  const [status, setStatus] = useState('idle')

  // ---- fetch wind for the current view -----------------------------------
  useEffect(() => {
    if (!map || !visible) return
    let timer = 0
    let controller = null
    let lastFetchAt = 0

    const load = () => {
      controller?.abort()
      controller = new AbortController()
      lastFetchAt = Date.now()
      const b = map.getBounds()
      coveredRef.current = {
        s: b.getSouth(), n: b.getNorth(), w: b.getWest(), e: b.getEast(), at: Date.now(),
      }
      setStatus('loading')

      fetchWindGrid(b, controller.signal)
        .then((pts) => {
          if (!pts.length) throw new Error('no wind points')
          dataRef.current = pts
          particlesRef.current = []
          setStatus('ok')
        })
        .catch((err) => {
          if (err.name === 'AbortError') return
          if (dataRef.current) { setStatus('stale'); return }
          // Nothing on screen yet, so fall back to the local ERA5 field rather
          // than leaving the layer blank.
          return fetchLocalWind()
            .then((pts) => {
              if (!pts.length) throw new Error('no local wind')
              dataRef.current = pts
              particlesRef.current = []
              setStatus('local')
            })
            .catch(() => setStatus('error'))
        })
    }

    // Debounce alone still lets a series of deliberate pans fire a request each.
    // Open-Meteo is a free service and starts refusing under that, which is what
    // makes the field go stale, so hold a hard floor between calls too.
    // Only spend quota when the view has actually left the area already fetched,
    // or the data has aged out. Panning within the covered box is free.
    const needsRefetch = () => {
      const c = coveredRef.current
      if (!c) return true
      if (Date.now() - c.at > REFRESH_AFTER_MS) return true
      const b = map.getBounds()
      return b.getSouth() < c.s || b.getNorth() > c.n || b.getWest() < c.w || b.getEast() > c.e
    }

    const schedule = () => {
      clearTimeout(timer)
      if (!needsRefetch()) return
      const since = Date.now() - lastFetchAt
      timer = setTimeout(load, Math.max(FETCH_DEBOUNCE_MS, MIN_FETCH_GAP_MS - since))
    }

    load()
    map.on('moveend', schedule)
    return () => {
      clearTimeout(timer)
      controller?.abort()
      map.off('moveend', schedule)
    }
  }, [map, visible])

  // ---- render -------------------------------------------------------------
  useEffect(() => {
    const fieldCanvas = fieldCanvasRef.current
    const flowCanvas = flowCanvasRef.current
    if (!map || !fieldCanvas || !flowCanvas) return

    const fieldCtx = fieldCanvas.getContext('2d')
    const flowCtx = flowCanvas.getContext('2d')
    if (!fieldCtx || !flowCtx) return

    if (!visible) {
      fieldCtx.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height)
      flowCtx.clearRect(0, 0, flowCanvas.width, flowCanvas.height)
      particlesRef.current = []
      return
    }

    let width = 0
    let height = 0
    let dpr = 1
    const scratch = document.createElement('canvas')
    const scratchCtx = scratch.getContext('2d')

    const resize = () => {
      const rect = map.getContainer().getBoundingClientRect()
      width = Math.max(1, Math.floor(rect.width))
      height = Math.max(1, Math.floor(rect.height))
      dpr = window.devicePixelRatio || 1
      for (const c of [fieldCanvas, flowCanvas]) {
        c.width = Math.floor(width * dpr)
        c.height = Math.floor(height * dpr)
        c.style.width = `${width}px`
        c.style.height = `${height}px`
      }
      fieldCtx.setTransform(dpr, 0, 0, dpr, 0, 0)
      flowCtx.setTransform(dpr, 0, 0, dpr, 0, 0)
      scratch.width = Math.max(2, Math.ceil(width / FIELD_DOWNSCALE))
      scratch.height = Math.max(2, Math.ceil(height / FIELD_DOWNSCALE))
      fieldDirty = true
    }

    // Inverse-distance weighting over the sample grid. The samples are a few
    // degrees apart, so nearest-neighbour alone would show visible facets.
    const sample = (lon, lat) => {
      const pts = dataRef.current
      if (!pts) return null
      let su = 0, sv = 0, sw = 0
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i]
        let dLon = p.lon - lon
        if (dLon > 180) dLon -= 360
        else if (dLon < -180) dLon += 360
        const dLat = p.lat - lat
        const d2 = dLon * dLon + dLat * dLat
        if (d2 < 1e-6) return { u: p.u, v: p.v }
        // Inverse-square, not inverse-fourth: a sharper falloff pulls each cell
        // toward its single nearest sample and the field renders as bullseyes
        // around the grid points rather than a continuous flow.
        const w = 1 / (d2 + 0.6)
        su += p.u * w; sv += p.v * w; sw += w
      }
      return sw === 0 ? null : { u: su / sw, v: sv / sw }
    }

    // Redrawing the field costs an unproject plus an IDW pass per cell, which is
    // far too much to run on every frame of a drag. Streaks keep animating at
    // full rate; the colour field lags by at most a few frames, which is not
    // perceptible while the map is moving.
    let fieldDirty = true
    let sinceFieldDraw = 0
    const FIELD_FRAME_INTERVAL = 4
    const markDirty = () => { fieldDirty = true }

    const drawField = () => {
      if (!dataRef.current || !scratchCtx) return
      const sw = scratch.width
      const sh = scratch.height
      const img = scratchCtx.createImageData(sw, sh)
      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const px = (x / (sw - 1)) * width
          const py = (y / (sh - 1)) * height
          const ll = map.unproject([px, py])
          const wind = sample(ll.lng, ll.lat)
          const i = (y * sw + x) * 4
          if (!wind) { img.data[i + 3] = 0; continue }
          const kmh = Math.hypot(wind.u, wind.v) * 3.6
          const [r, g, b] = rampColor(kmh)
          img.data[i] = r
          img.data[i + 1] = g
          img.data[i + 2] = b
          // Uniform alpha here; overall translucency is set in CSS so the
          // basemap stays readable underneath and is tunable in one place.
          img.data[i + 3] = 255
        }
      }
      scratchCtx.putImageData(img, 0, 0)
      fieldCtx.clearRect(0, 0, width, height)
      fieldCtx.imageSmoothingEnabled = true
      fieldCtx.imageSmoothingQuality = 'high'
      fieldCtx.drawImage(scratch, 0, 0, sw, sh, 0, 0, width, height)
    }

    const seed = (p) => {
      p.x = Math.random() * width
      p.y = Math.random() * height
      p.age = Math.floor(Math.random() * MAX_AGE_FRAMES)
      return p
    }

    const step = () => {
      if (dataRef.current) {
        const b = map.getBounds()
        let lonSpan = b.getEast() - b.getWest()
        if (lonSpan <= 0) lonSpan += 360
        const degPerPx = lonSpan / Math.max(width, 1)

        sinceFieldDraw += 1
        if (fieldDirty && sinceFieldDraw >= FIELD_FRAME_INTERVAL) {
          drawField()
          fieldDirty = false
          sinceFieldDraw = 0
        }

        const target = Math.round(
          Math.min(MAX_PARTICLES, Math.max(MIN_PARTICLES, (width * height) / PX2_PER_PARTICLE)),
        )
        if (particlesRef.current.length !== target) {
          particlesRef.current = Array.from({ length: target }, () =>
            seed({ x: 0, y: 0, age: 0 }))
        }

        // Fade rather than clear: the residue is what draws the streaks.
        flowCtx.globalCompositeOperation = 'destination-out'
        flowCtx.fillStyle = `rgba(0,0,0,${1 - TRAIL_FADE})`
        flowCtx.fillRect(0, 0, width, height)
        flowCtx.globalCompositeOperation = 'source-over'
        flowCtx.strokeStyle = 'rgba(255,255,255,0.85)'
        flowCtx.lineWidth = 1.15
        flowCtx.lineCap = 'round'
        flowCtx.beginPath()

        for (const p of particlesRef.current) {
          if (p.age > MAX_AGE_FRAMES) { seed(p); continue }
          const ll = map.unproject([p.x, p.y])
          const wind = sample(ll.lng, ll.lat)
          if (!wind) { seed(p); continue }

          // Advance in geographic space so streaks follow the real field, then
          // project back: moving in screen space would distort with latitude.
          // The step is sized from the current degrees-per-pixel so a streak is
          // the same length on screen whatever the zoom.
          const lonScale = 1 / Math.max(Math.cos((ll.lat * Math.PI) / 180), 0.2)
          const stepDeg = degPerPx * PX_PER_FRAME_AT_REF / REF_SPEED_MS
          const nextLon = ll.lng + wind.u * stepDeg * lonScale
          const nextLat = ll.lat + wind.v * stepDeg
          if (nextLat > 89 || nextLat < -89) { seed(p); continue }

          const next = map.project([nextLon, nextLat])
          p.age += 1

          if (next.x < -20 || next.x > width + 20 || next.y < -20 || next.y > height + 20) {
            seed(p); continue
          }
          if (Math.hypot(next.x - p.x, next.y - p.y) < 25) {
            flowCtx.moveTo(p.x, p.y)
            flowCtx.lineTo(next.x, next.y)
          }
          p.x = next.x
          p.y = next.y
        }
        flowCtx.stroke()
      }
      rafRef.current = requestAnimationFrame(step)
    }

    resize()
    map.on('resize', resize)
    map.on('move', markDirty)
    rafRef.current = requestAnimationFrame(step)

    return () => {
      cancelAnimationFrame(rafRef.current)
      map.off('resize', resize)
      map.off('move', markDirty)
      fieldCtx.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height)
      flowCtx.clearRect(0, 0, flowCanvas.width, flowCanvas.height)
    }
  }, [map, visible])

  return (
    <>
      <canvas ref={fieldCanvasRef} className={`wind-field-canvas ${visible ? 'on' : ''}`} />
      <canvas ref={flowCanvasRef} className={`wind-flow-canvas ${visible ? 'on' : ''}`} />
      {visible && (
        <div className="wind-legend">
          <div className="wind-legend-head">
            <span>Wind speed</span>
            <span className="wind-legend-src">
              {status === 'loading' ? 'loading…'
                : status === 'error' ? 'unavailable'
                : status === 'stale' ? 'Open-Meteo · last good'
                : status === 'local' ? 'ERA5 · Southern Ocean'
                : 'Open-Meteo'}
            </span>
          </div>
          <div className="wind-legend-bar" />
          <div className="wind-legend-ticks">
            {[0, 20, 40, 60, 80, 100, 120].map((v) => <span key={v}>{v}</span>)}
          </div>
          <div className="wind-legend-unit">km/h</div>
        </div>
      )}
    </>
  )
}
