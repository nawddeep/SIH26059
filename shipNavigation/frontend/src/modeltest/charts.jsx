/**
 * Inline SVG/canvas chart primitives for the model dashboard.
 *
 * Hand-rolled rather than pulled from a charting library: the page already
 * hand-rolls its own CSS, and the three forms here (a swept curve, a packed
 * grid, a vector field) are each small enough that a dependency would cost
 * more than it saved.
 *
 * Shared rules, applied by every component below:
 *   - one 2px line for the series that matters, hairlines for context
 *   - solid hairline grid, never dashed
 *   - a crosshair or per-mark tooltip, with a hit target bigger than the mark
 *   - a table view twin, so no value is reachable only by hovering
 */
import React, { useEffect, useMemo, useRef, useState } from 'react'

const W = 560
const H = 268
const M = { top: 16, right: 20, bottom: 40, left: 58 }
const PW = W - M.left - M.right
const PH = H - M.top - M.bottom

/** Round axis bounds out to clean numbers so ticks read 0 / 0.5 / 1. */
function niceTicks(min, max, count = 5) {
  if (!(max > min)) return { lo: 0, hi: 1, ticks: [0, 0.5, 1] }
  const raw = (max - min) / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].find((s) => s * mag >= raw) * mag
  const lo = Math.floor(min / step) * step
  const hi = Math.ceil(max / step) * step
  const ticks = []
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(Number(v.toFixed(10)))
  return { lo, hi, ticks }
}

function TableToggle({ open, onClick, rows }) {
  return (
    <button type="button" className="mt-tableview-btn" onClick={onClick} aria-expanded={open}>
      {open ? 'Hide' : 'Show'} table ({rows})
    </button>
  )
}

/* ------------------------------------------------------------------ curve */
/**
 * One emphasised curve over a ladder of context curves.
 *
 * The context lines are the same quantity for the other hulls. They are not a
 * second series competing for identity - they are the backdrop that makes the
 * selected hull's position legible - so they stay hairline grey and unlabelled
 * rather than taking categorical hues.
 */
export function CurveChart({
  x, values, context = [], label, yLabel, xLabel = 'Sea-ice concentration',
  marker, unit = '', decimals = 2, color = 'var(--viz-series)',
}) {
  const [hover, setHover] = useState(null)
  const [table, setTable] = useState(false)
  const svgRef = useRef(null)

  const all = useMemo(() => {
    const flat = [...values, ...context.flatMap((c) => c.values)]
    return niceTicks(Math.min(0, ...flat), Math.max(...flat))
  }, [values, context])

  const sx = (v) => M.left + ((v - x[0]) / (x[x.length - 1] - x[0])) * PW
  const sy = (v) => M.top + PH - ((v - all.lo) / (all.hi - all.lo)) * PH
  const path = (vals) => vals.map((v, i) => `${i ? 'L' : 'M'}${sx(x[i]).toFixed(2)},${sy(v).toFixed(2)}`).join('')

  const onMove = (e) => {
    const r = svgRef.current.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    const t = (px - M.left) / PW
    const i = Math.max(0, Math.min(x.length - 1, Math.round(t * (x.length - 1))))
    setHover(i)
  }

  const fmt = (v) => `${v.toFixed(decimals)}${unit}`

  return (
    <figure className="mt-chart">
      <figcaption className="mt-chart-title">{label}</figcaption>
      <svg
        ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="mt-svg" role="img"
        aria-label={`${label}. ${yLabel} against ${xLabel}.`}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}
      >
        {all.ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + PW} y1={sy(t)} y2={sy(t)} className="mt-grid" />
            <text x={M.left - 8} y={sy(t)} className="mt-axis-t" textAnchor="end" dominantBaseline="middle">
              {t}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={sx(t)} y={M.top + PH + 22} className="mt-axis-t" textAnchor="middle">
            {t}
          </text>
        ))}
        <line
          x1={M.left} x2={M.left + PW} y1={M.top + PH} y2={M.top + PH} className="mt-axis-line"
        />

        {context.map((c) => (
          <path key={c.label} d={path(c.values)} className="mt-line-ctx" />
        ))}
        <path d={path(values)} className="mt-line" style={{ stroke: color }} />

        {/* Direct label at the line end - identity never rests on colour alone. */}
        <text
          x={M.left + PW - 2} y={Math.max(M.top + 10, sy(values[values.length - 1]) - 9)}
          className="mt-line-label" textAnchor="end"
        >
          {marker?.seriesLabel}
        </text>

        {marker && (
          <g>
            <circle cx={sx(marker.x)} cy={sy(marker.y)} r={6.5} className="mt-marker-ring" />
            <circle cx={sx(marker.x)} cy={sy(marker.y)} r={4.5} style={{ fill: color }} />
            <text
              x={sx(marker.x) + (marker.x > 0.75 ? -10 : 10)}
              y={sy(marker.y) - 10}
              className="mt-marker-label"
              textAnchor={marker.x > 0.75 ? 'end' : 'start'}
            >
              {fmt(marker.y)}
            </text>
          </g>
        )}

        {hover !== null && (
          <g>
            <line x1={sx(x[hover])} x2={sx(x[hover])} y1={M.top} y2={M.top + PH} className="mt-crosshair" />
            <circle cx={sx(x[hover])} cy={sy(values[hover])} r={6.5} className="mt-marker-ring" />
            <circle cx={sx(x[hover])} cy={sy(values[hover])} r={4} style={{ fill: color }} />
          </g>
        )}

        <text x={M.left + PW / 2} y={H - 4} className="mt-axis-l" textAnchor="middle">{xLabel}</text>
        <text x={12} y={M.top + PH / 2} className="mt-axis-l" textAnchor="middle"
              transform={`rotate(-90 12 ${M.top + PH / 2})`}>{yLabel}</text>
      </svg>

      {hover !== null && (
        <div className="mt-tip-inline">
          <strong>{fmt(values[hover])}</strong>
          <span>at SIC {x[hover].toFixed(3)}</span>
        </div>
      )}

      <TableToggle open={table} onClick={() => setTable(!table)} rows={x.length} />
      {table && (
        <div className="mt-tablewrap">
          <table className="mt-table">
            <thead><tr><th>SIC</th><th>{yLabel}</th></tr></thead>
            <tbody>
              {x.map((v, i) => (
                <tr key={v}><td className="num">{v.toFixed(3)}</td><td className="num">{fmt(values[i])}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  )
}

/* ---------------------------------------------------------------- heatmap */
// Sequential: one hue, light -> dark. Land gets a neutral grey so it never
// reads as "ice-free ocean", which is a different fact about the same cell.
const SIC_RAMP = ['#eef5fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
const LAND_FILL = [214, 213, 208]

function rampRgb(t) {
  const p = Math.max(0, Math.min(1, t)) * (SIC_RAMP.length - 1)
  const i = Math.min(SIC_RAMP.length - 2, Math.floor(p))
  const f = p - i
  const a = SIC_RAMP[i], b = SIC_RAMP[i + 1]
  const ch = (h, o) => parseInt(h.slice(o, o + 2), 16)
  return [
    Math.round(ch(a, 1) + (ch(b, 1) - ch(a, 1)) * f),
    Math.round(ch(a, 3) + (ch(b, 3) - ch(a, 3)) * f),
    Math.round(ch(a, 5) + (ch(b, 5) - ch(a, 5)) * f),
  ]
}

/**
 * The 332x316 forecast field, painted to a canvas.
 *
 * 104,912 cells is two orders of magnitude past what SVG rects handle, so the
 * grid arrives as one byte per cell and goes straight into ImageData.
 */
export function SicHeatmap({ field }) {
  const canvasRef = useRef(null)
  const [hover, setHover] = useState(null)
  const [table, setTable] = useState(false)

  const cells = useMemo(() => {
    const bin = atob(field.grid)
    const a = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i)
    return a
  }, [field.grid])

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const { width: w, height: h, land, scale } = field
    cv.width = w
    cv.height = h
    const ctx = cv.getContext('2d')
    const img = ctx.createImageData(w, h)
    for (let i = 0; i < cells.length; i++) {
      const o = i * 4
      if (cells[i] === land) {
        img.data[o] = LAND_FILL[0]; img.data[o + 1] = LAND_FILL[1]; img.data[o + 2] = LAND_FILL[2]
      } else {
        const [r, g, b] = rampRgb(cells[i] / scale)
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b
      }
      img.data[o + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [cells, field])

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const col = Math.floor(((e.clientX - r.left) / r.width) * field.width)
    const row = Math.floor(((e.clientY - r.top) / r.height) * field.height)
    if (col < 0 || row < 0 || col >= field.width || row >= field.height) return setHover(null)
    const v = cells[row * field.width + col]
    setHover({ row, col, land: v === field.land, sic: v / field.scale })
  }

  const legend = [0, 0.2, 0.4, 0.6, 0.8, 1]
  const s = field.stats

  return (
    <figure className="mt-chart">
      <figcaption className="mt-chart-title">
        Predicted concentration field · {field.date}
      </figcaption>
      <div className="mt-heatwrap" onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        <canvas
          ref={canvasRef} className="mt-heatcanvas"
          role="img" aria-label={`Predicted sea-ice concentration for ${field.date}, ${field.width} by ${field.height} grid.`}
        />
        {hover && (
          <div className="mt-heattip">
            <strong>{hover.land ? 'Land' : hover.sic.toFixed(3)}</strong>
            <span>row {hover.row}, col {hover.col}</span>
          </div>
        )}
      </div>

      <div className="mt-legend">
        <span className="mt-legend-l">0</span>
        <div className="mt-legend-bar" aria-hidden="true">
          {legend.map((t) => {
            const [r, g, b] = rampRgb(t)
            return <span key={t} style={{ background: `rgb(${r},${g},${b})` }} />
          })}
        </div>
        <span className="mt-legend-l">1 concentration</span>
        <span className="mt-legend-land"><i aria-hidden="true" /> land</span>
      </div>

      <TableToggle open={table} onClick={() => setTable(!table)} rows={6} />
      {table && (
        <table className="mt-table">
          <thead><tr><th>Measure</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Mean concentration over ocean</td><td className="num">{s.meanOcean}</td></tr>
            <tr><td>Peak concentration</td><td className="num">{s.max}</td></tr>
            <tr><td>Cells at or above 0.15</td><td className="num">{s.iceCells.toLocaleString()}</td></tr>
            <tr><td>Ocean cells in grid</td><td className="num">{s.oceanCells.toLocaleString()}</td></tr>
            <tr><td>Ice area</td><td className="num">{s.iceAreaMkm2} M km²</td></tr>
            <tr><td>Ice extent</td><td className="num">{s.extentMkm2} M km²</td></tr>
          </tbody>
        </table>
      )}
    </figure>
  )
}

/* ----------------------------------------------------------- drift field */
const VW = 560
const VH = 300
const VM = { top: 16, right: 18, bottom: 40, left: 52 }

/**
 * Predicted 24 h displacement per berg, on a lon/lat plane.
 *
 * Stationary bergs are drawn, not dropped. The first stage of this model is a
 * moving/stationary gate and most fixes are grounded, so a field showing only
 * the arrows would misrepresent what the model spends its time saying.
 */
export function DriftField({ data }) {
  const [hover, setHover] = useState(null)
  const [table, setTable] = useState(false)
  const svgRef = useRef(null)
  const bergs = data.bergs

  const bounds = useMemo(() => {
    const lats = bergs.map((b) => b.lat), lons = bergs.map((b) => b.lon)
    const pad = (a) => {
      const lo = Math.min(...a), hi = Math.max(...a)
      const m = Math.max((hi - lo) * 0.12, 1.5)
      return [lo - m, hi + m]
    }
    return { lat: pad(lats), lon: pad(lons) }
  }, [bergs])

  const pw = VW - VM.left - VM.right
  const ph = VH - VM.top - VM.bottom
  const sx = (lon) => VM.left + ((lon - bounds.lon[0]) / (bounds.lon[1] - bounds.lon[0])) * pw
  const sy = (lat) => VM.top + ph - ((lat - bounds.lat[0]) / (bounds.lat[1] - bounds.lat[0])) * ph

  // Scale the longest arrow to a readable length rather than to true distance:
  // at this zoom a 3 km displacement is sub-pixel. The legend says so.
  const maxDrift = Math.max(data.maxDriftKm24h, 0.001)
  const ARROW_MAX = 46
  const arrow = (b) => {
    const len = (b.driftKm24h / maxDrift) * ARROW_MAX
    const th = Math.atan2(b.uMs, b.vMs)
    return { dx: Math.sin(th) * len, dy: -Math.cos(th) * len, len }
  }

  const latTicks = niceTicks(bounds.lat[0], bounds.lat[1], 4).ticks.filter((t) => t >= bounds.lat[0] && t <= bounds.lat[1])
  const lonTicks = niceTicks(bounds.lon[0], bounds.lon[1], 4).ticks.filter((t) => t >= bounds.lon[0] && t <= bounds.lon[1])

  return (
    <figure className="mt-chart">
      <figcaption className="mt-chart-title">
        Predicted 24 h drift · {data.date} · {data.movingCount} of {data.count} gated as moving
      </figcaption>
      <svg ref={svgRef} viewBox={`0 0 ${VW} ${VH}`} className="mt-svg" role="img"
           aria-label={`Predicted iceberg drift vectors for ${data.date}.`}>
        <defs>
          <marker id="mt-arrowhead" viewBox="0 0 8 8" refX="6" refY="4"
                  markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,1 L7,4 L0,7 z" fill="var(--viz-series)" />
          </marker>
        </defs>

        {latTicks.map((t) => (
          <g key={`la${t}`}>
            <line x1={VM.left} x2={VM.left + pw} y1={sy(t)} y2={sy(t)} className="mt-grid" />
            <text x={VM.left - 8} y={sy(t)} className="mt-axis-t" textAnchor="end" dominantBaseline="middle">
              {t}°
            </text>
          </g>
        ))}
        {lonTicks.map((t) => (
          <text key={`lo${t}`} x={sx(t)} y={VM.top + ph + 22} className="mt-axis-t" textAnchor="middle">{t}°</text>
        ))}
        <line x1={VM.left} x2={VM.left + pw} y1={VM.top + ph} y2={VM.top + ph} className="mt-axis-line" />

        {bergs.filter((b) => !b.moving).map((b, i) => (
          <circle key={`s${b.bergId}${i}`} cx={sx(b.lon)} cy={sy(b.lat)} r={3.5} className="mt-berg-still" />
        ))}

        {bergs.filter((b) => b.moving).map((b, i) => {
          const a = arrow(b)
          return (
            <g key={`m${b.bergId}${i}`}>
              <line
                x1={sx(b.lon)} y1={sy(b.lat)} x2={sx(b.lon) + a.dx} y2={sy(b.lat) + a.dy}
                className="mt-berg-vec" markerEnd="url(#mt-arrowhead)"
              />
              <circle cx={sx(b.lon)} cy={sy(b.lat)} r={4.5} className="mt-marker-ring" />
              <circle cx={sx(b.lon)} cy={sy(b.lat)} r={3} className="mt-berg-move" />
            </g>
          )
        })}

        {/* Hit targets last and transparent: 24px beats a 7px dot for pointing at. */}
        {bergs.map((b, i) => (
          <circle
            key={`h${b.bergId}${i}`} cx={sx(b.lon)} cy={sy(b.lat)} r={12}
            fill="transparent"
            onPointerEnter={() => setHover(b)} onPointerLeave={() => setHover(null)}
          />
        ))}

        <text x={VM.left + pw / 2} y={VH - 4} className="mt-axis-l" textAnchor="middle">Longitude</text>
        <text x={11} y={VM.top + ph / 2} className="mt-axis-l" textAnchor="middle"
              transform={`rotate(-90 11 ${VM.top + ph / 2})`}>Latitude</text>
      </svg>

      <div className="mt-vec-legend">
        <span><i className="k-move" aria-hidden="true" /> moving — arrow points along predicted drift</span>
        <span><i className="k-still" aria-hidden="true" /> gated stationary</span>
        <span className="mt-vec-scale">longest arrow = {data.maxDriftKm24h} km/24 h (lengths scaled to fit)</span>
      </div>

      {hover && (
        <div className="mt-tip-inline">
          <strong>{hover.driftKm24h} km / 24 h</strong>
          <span>
            berg {hover.bergId} · {hover.moving ? `bearing ${hover.bearingDeg}°` : 'stationary'} ·
            {' '}{hover.lat}°, {hover.lon}° · {hover.sizeNm} nm
          </span>
        </div>
      )}

      <TableToggle open={table} onClick={() => setTable(!table)} rows={bergs.length} />
      {table && (
        <div className="mt-tablewrap">
          <table className="mt-table">
            <thead>
              <tr><th>Berg</th><th>Lat</th><th>Lon</th><th>State</th><th>Drift km/24 h</th><th>Bearing</th></tr>
            </thead>
            <tbody>
              {bergs.map((b, i) => (
                <tr key={`t${b.bergId}${i}`}>
                  <td>{b.bergId}</td>
                  <td className="num">{b.lat}</td>
                  <td className="num">{b.lon}</td>
                  <td>{b.moving ? 'moving' : 'stationary'}</td>
                  <td className="num">{b.driftKm24h}</td>
                  <td className="num">{b.moving ? `${b.bearingDeg}°` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  )
}
