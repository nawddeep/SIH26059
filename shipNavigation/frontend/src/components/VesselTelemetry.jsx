import React, { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'

/**
 * Live vessel telemetry forwarded by the shipboard gateway.
 *
 * Shows where the ship is and what its instruments report. It is explicitly NOT
 * an input to the forecast, drift or routing models - the telemetry is current
 * while the processed ice archive ends 2018-12-31, so anything derived from
 * both would pair a real position with an eight-year-old environment. The strip
 * says "display only" for that reason, and the wording should stay.
 */
const POLL_MS = 2000

function Field({ label, value, unit, alert }) {
  if (value === null || value === undefined) return null
  return (
    <div className={`vt-field${alert ? ' vt-alert' : ''}`}>
      <span className="vt-field-l">{label}</span>
      <span className="vt-field-v">{value}{unit && <em>{unit}</em>}</span>
    </div>
  )
}

export default function VesselTelemetry({ map }) {
  const [data, setData] = useState(null)
  const [reachable, setReachable] = useState(true)
  const markerRef = useRef(null)

  useEffect(() => {
    let alive = true
    const tick = () => {
      fetch('/api/telemetry/live')
        .then((r) => r.json())
        .then((d) => { if (alive) { setData(d); setReachable(true) } })
        .catch(() => { if (alive) setReachable(false) })
    }
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [])

  // Drive a marker on the existing map rather than adding a layer: one moving
  // point does not justify a source, and a DOM marker rotates cleanly.
  useEffect(() => {
    if (!map) return
    const pos = data?.position
    if (!pos) return

    if (!markerRef.current) {
      const el = document.createElement('div')
      el.className = 'vt-marker'
      el.innerHTML = '<span class="vt-marker-hull"></span><span class="vt-marker-pulse"></span>'
      markerRef.current = new maplibregl.Marker({ element: el })
        .setLngLat([pos.lon, pos.lat])
        .addTo(map)
    } else {
      markerRef.current.setLngLat([pos.lon, pos.lat])
    }
    const hull = markerRef.current.getElement().querySelector('.vt-marker-hull')
    if (hull && pos.courseDeg != null) hull.style.transform = `rotate(${pos.courseDeg}deg)`
  }, [map, data])

  useEffect(() => () => { if (markerRef.current) markerRef.current.remove() }, [])

  if (!reachable || !data || !data.connected) {
    return (
      <div className="vt-strip vt-offline">
        <span className="vt-dot off" />
        <span className="vt-title">Shipboard gateway</span>
        <span className="vt-msg">
          no telemetry received — start the gateway and run the shore listener with
          {' '}<code>--forward</code>
        </span>
      </div>
    )
  }

  const p = data.position
  const w = data.weather
  const e = data.engine
  const stale = data.lastSeen
    ? (Date.now() - new Date(data.lastSeen).getTime()) / 1000
    : null

  return (
    <div className="vt-strip">
      <div className="vt-head">
        <span className={`vt-dot ${stale != null && stale > 15 ? 'stale' : 'on'}`} />
        <span className="vt-title">{data.vesselName || 'Vessel'}</span>
        {data.mmsi && <span className="vt-mmsi">MMSI {data.mmsi}</span>}
        <span className="vt-count">{data.recordsReceived} records</span>
        {stale != null && stale > 15 && (
          <span className="vt-stalemsg">last update {Math.round(stale)}s ago</span>
        )}
      </div>

      <div className="vt-fields">
        {p && <Field label="Position" value={`${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}`} />}
        {p && <Field label="Speed" value={p.speedKn} unit=" kn" />}
        {p && <Field label="Course" value={p.courseDeg} unit="°" />}
        {w && <Field label="Wind" value={w.wind_speed_kn} unit=" kn"
                     alert={(w.alerts || []).includes('gale_force_wind')} />}
        {w && <Field label="Visibility" value={w.visibility_km} unit=" km"
                     alert={(w.alerts || []).includes('severe_visibility')} />}
        {w && <Field label="Air" value={w.air_temp_c} unit="°C" />}
        {e && <Field label="RPM" value={e.rpm} />}
        {e && <Field label="Fuel" value={e.fuel_remaining_pct} unit="%" />}
        {e && <Field label="Engine" value={e.engine_temp_c} unit="°C"
                     alert={(e.alerts || []).includes('engine_overheat')} />}
      </div>

      {data.alerts.length > 0 && (
        <div className="vt-alerts">
          {data.alerts.slice(0, 3).map((a, i) => (
            <span key={i} className="vt-alert-tag">
              {a.alert.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}

      <p className="vt-note">
        Forwarded by the shipboard gateway over the satellite uplink · display only,
        not an input to the models
      </p>
    </div>
  )
}
