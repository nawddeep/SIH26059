import React, { useEffect } from 'react'
import { MapPin, Ship, Navigation, Play, Pause, AlertTriangle, Clock, CloudLightning } from 'lucide-react'
import { fmtHours, formatCoord } from '../utils'
import { usePlanner } from '../store'

function Row({ label, value, sub, icon }) {
  return (
    <div className="res-row">
      <span className="res-icon">{icon}</span>
      <span className="res-label">{label}</span>
      <span className="res-value">
        {value}
        {sub ? <em>{sub}</em> : null}
      </span>
    </div>
  )
}

/**
 * Why the router chose this track.
 *
 * The backend already computes all of this while costing the route; without it
 * on screen the reader has to reverse-engineer the reasoning from a polyline,
 * which is the job a decision-support system exists to do for them.
 *
 * The caveat is not decoration. Every route here is costed against historical
 * reanalysis, and anyone acting on one needs to know that before they act.
 */
function RouteExplanation({ exp }) {
  const [open, setOpen] = React.useState(true)
  const risk = exp.maxIceRisk ?? 0
  const tone = risk >= 0.66 ? 'crit' : risk >= 0.33 ? 'warn' : 'ok'

  return (
    <div className={`route-why ${tone}`}>
      <button type="button" className="route-why-head" onClick={() => setOpen(!open)}>
        <span className="route-why-title">Why this route</span>
        <span className="route-why-toggle">{open ? '−' : '+'}</span>
      </button>

      {open && (
        <>
          <p className="route-why-reason">{exp.reason}</p>

          <div className="route-why-grid">
            <div><dt>Peak ice risk</dt><dd className={`risk-${tone}`}>{risk.toFixed(2)}</dd></div>
            <div><dt>Track in ice</dt>
                 <dd>{Math.round((exp.fractionOfTrackInIce ?? 0) * 100)}%</dd></div>
            <div><dt>Ice fuel penalty</dt><dd>{exp.maxIceFuelPenalty}×</dd></div>
            <div><dt>Icebergs tracked</dt><dd>{exp.icebergsTracked}</dd></div>
            {exp.closestIcebergNm != null && (
              <div><dt>Closest berg</dt>
                   <dd className={exp.icebergIntersections ? 'risk-crit' : ''}>
                     {exp.closestIcebergNm} nm</dd></div>
            )}
            <div><dt>Hull class</dt><dd>{exp.vesselIceClass}</dd></div>
          </div>

          <p className="route-why-driver">{exp.dominantCostTerm}</p>
          <p className="route-why-caveat">{exp.caveat}</p>
        </>
      )}
    </div>
  )
}


export default function ResultsPanel({ route, loading, error }) {
  const {
    departureTimeUTC,
    sliderTimeISO, setSliderTimeISO,
    isPlaying, setIsPlaying,
    closestApproaches, routeWeather,
  } = usePlanner()

  const startMs = new Date(departureTimeUTC).getTime()
  const totalHours = route ? route.totalDurationHours : 1
  const endMs = startMs + totalHours * 3600000

  const currentMs = sliderTimeISO ? new Date(sliderTimeISO).getTime() : startMs
  const elapsedHours = Math.max(0, Math.min(totalHours, (currentMs - startMs) / 3600000))
  const progressPct = Math.round((elapsedHours / totalHours) * 100) || 0

  const weatherWarnings = React.useMemo(() => {
    if (!routeWeather || !routeWeather.samples) return []
    return routeWeather.samples.filter(s => s.hazardLevel === 'severe' || s.hazardLevel === 'moderate').slice(0, 4)
  }, [routeWeather])

  useEffect(() => {
    let timer = null
    if (isPlaying && route) {
      timer = setInterval(() => {
        setSliderTimeISO(prev => {
          const curr = prev ? new Date(prev).getTime() : startMs
          const next = curr + 0.5 * 3600000 // 30-min step per tick
          if (next >= endMs) {
            setIsPlaying(false)
            return new Date(endMs).toISOString()
          }
          return new Date(next).toISOString()
        })
      }, 150)
    }
    return () => clearInterval(timer)
  }, [isPlaying, route, startMs, endMs, setSliderTimeISO, setIsPlaying])

  if (error) {
    return (
      <div className="results results-error">
        <div className="res-title">Route</div>
        <p className="error-msg">{error}</p>
      </div>
    )
  }

  if (loading && !route) {
    return (
      <div className="results results-loading">
        <div className="res-title">Route</div>
        <div className="skeleton-line" />
        <div className="skeleton-line short" />
        <div className="skeleton-line" />
        <div className="skeleton-line short" />
      </div>
    )
  }

  if (!route) return null

  const places = route.legs.map(l => l.from.label)
  places.push(route.legs[route.legs.length - 1].to.label)

  function handleSliderChange(e) {
    const hours = Number(e.target.value)
    const target = new Date(startMs + hours * 3600000).toISOString()
    setSliderTimeISO(target)
  }

  return (
    <div className="results">
      <div className="res-title">Route</div>

      <div className="res-route-stops">
        {places.map((p, i) => (
          <div key={i} className={`res-stop ${i === 0 ? 'origin' : i === places.length - 1 ? 'dest' : ''}`}>
            <div className="res-stop-badge">
              <span className="res-stop-num">{i + 1}</span>
              {i < places.length - 1 && <span className="res-stop-line" />}
            </div>
            <span className="res-stop-name">{p}</span>
          </div>
        ))}
      </div>

      <Row icon={<Ship size={15} />} label="Distance" value={`${route.totalDistanceNm.toLocaleString()} nm`} sub="total geodesic" />
      <Row icon={<MapPin size={15} />} label="Distance in ECA" value={route.totalDistanceInEcaNm ? `${route.totalDistanceInEcaNm.toLocaleString()} nm` : '0 nm'} sub={route.totalDistanceInEcaNm ? 'Emission control areas' : undefined} />
      <Row icon={<span style={{ fontSize: 13 }}>🧊</span>} label="Distance in Sea Ice" value={route.totalDistanceInSeaIceNm ? `${route.totalDistanceInSeaIceNm.toLocaleString()} nm` : '0 nm'} sub={route.maxSeaIceConcentrationPct ? `Max conc: ${route.maxSeaIceConcentrationPct}%` : 'Open water'} />
      {route.meanPolarisRisk !== undefined && route.maxPolarisRisk > 0 && (
        <Row
          icon={<span style={{ fontSize: 13 }}>📊</span>}
          label="POLARIS ice risk"
          value={`${route.meanPolarisRisk} mean`}
          sub={`Peak ${route.maxPolarisRisk} · ${route.vessel?.iceClass === 'none' ? 'unclassed hull' : route.vessel?.iceClass}`}
        />
      )}
      {route.iceDataSource && (
        <div className={`ice-source ${route.iceDataSource.startsWith('synthetic') ? 'synthetic' : 'live'}`}>
          {route.iceDataSource.startsWith('synthetic')
            ? `Ice costing used ${route.iceDataSource} — not the trained forecast`
            : `Ice costing: ${route.iceDataSource}`}
        </div>
      )}
      {route.explanation && <RouteExplanation exp={route.explanation} />}
      {route.estimatedFuelTons !== undefined && (
        <Row
          icon={<span style={{ fontSize: 13 }}>⛽</span>}
          label="Estimated Fuel Burn"
          value={`${route.estimatedFuelTons.toLocaleString()} MT`}
          sub={route.maxIceFuelPenalty > 1.05
            ? `Peak ice penalty ${route.maxIceFuelPenalty}× open water`
            : 'Metric tons HFO/MGO'}
        />
      )}
      {route.safetyScore !== undefined && (
        <Row icon={<span style={{ fontSize: 13 }}>🛡️</span>} label="Safety Score" value={`${route.safetyScore} / 100`} sub={route.safetyScore >= 80 ? 'Optimal Clearance' : route.safetyScore >= 50 ? 'Moderate Clearance' : 'High Risk Hazard'} />
      )}
      {route.currentAssistanceKnots !== undefined && (
        <Row icon={<span style={{ fontSize: 13 }}>🌊</span>} label="Current Assistance" value={`${route.currentAssistanceKnots >= 0 ? '+' : ''}${route.currentAssistanceKnots} kn`} sub="Ocean current net boost" />
      )}
      <Row icon={<Navigation size={15} />} label="Duration" value={fmtHours(route.totalDurationHours)} />
      <div className="res-row">
        <span className="res-icon">🕒</span>
        <span className="res-label">Estimated arrival (UTC)</span>
        <span className="res-value">{new Date(route.etaUTC).toLocaleString('en-GB', { timeZone: 'UTC', dateStyle: 'medium', timeStyle: 'short' })} UTC</span>
      </div>

      <div className="voyage-slider-section">
        <div className="slider-header">
          <button
            type="button"
            className="play-btn"
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause simulation' : 'Play voyage simulation'}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <div className="slider-time-info">
            <span className="slider-time-label">
              <Clock size={12} /> {new Date(sliderTimeISO || departureTimeUTC).toLocaleString('en-GB', { timeZone: 'UTC', dateStyle: 'short', timeStyle: 'short' })} UTC
            </span>
            <span className="slider-pct">{progressPct}%</span>
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={totalHours}
          step={0.25}
          value={elapsedHours}
          onChange={handleSliderChange}
          className="voyage-slider"
        />
        <div className="slider-ticks">
          <span>Dep</span>
          <span>+{Math.round(totalHours / 2)}h</span>
          <span>ETA</span>
        </div>
      </div>

      {closestApproaches && closestApproaches.length > 0 && (
        <div className="closest-approach-box">
          <div className="ca-title">
            <AlertTriangle size={14} className="ca-icon" /> Iceberg Closest Approach
          </div>
          {closestApproaches.map(app => (
            <div key={app.icebergId} className="ca-item">
              <span className="ca-name">🔺 {app.icebergName}</span>
              <span className="ca-value">
                <strong>{app.minDistanceNm} nm</strong>
                <em>{new Date(app.timeISO).toLocaleString('en-GB', { timeZone: 'UTC', dateStyle: 'short', timeStyle: 'short' })} UTC</em>
              </span>
            </div>
          ))}
        </div>
      )}

      {weatherWarnings && weatherWarnings.length > 0 && (
        <div className="closest-approach-box" style={{ background: '#fff7ed', borderColor: '#fed7aa' }}>
          <div className="ca-title" style={{ color: '#c2410c' }}>
            <CloudLightning size={14} className="ca-icon" /> Route Weather Hazards
          </div>
          {weatherWarnings.map((w, idx) => (
            <div key={idx} className="ca-item" style={{ borderBottomColor: '#ffedd5' }}>
              <span className="ca-name">
                {w.hazardLevel === 'severe' ? '🔴' : '🟡'} {w.windSpeedKn >= 34 ? 'Gale' : w.waveHeightM >= 4 ? 'Rough seas' : 'Moderate weather'} near {formatCoord(w.lat, w.lon)}
              </span>
              <span className="ca-value">
                <strong>{w.windSpeedKn} kn / {w.waveHeightM} m</strong>
                <em style={{ color: '#9a3412' }}>
                  Day {Math.floor(w.elapsedHours / 24) + 1} ({w.dataSource === 'forecast' ? 'forecast' : 'modeled estimate — beyond forecast range'})
                </em>
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="res-row crossing-row">
        <span className="res-icon">⛵</span>
        <span className="res-label">Crossing</span>
        <span className="res-value crossing-value">
          {route.crossings && route.crossings.length ? (
            route.crossings.map(c => <span key={c} className="crossing-chip">{c}</span>)
          ) : (
            <em className="none">— no named crossings —</em>
          )}
        </span>
      </div>

      <div className="res-per-leg">
        {route.legs.map((leg, i) => (
          <div key={i} className="res-leg">
            <span className="res-leg-names">{leg.from.label} → {leg.to.label}</span>
            <span className="res-leg-meta">
              {leg.distanceNm.toLocaleString()} nm · {fmtHours(leg.durationHours)}
              {leg.crossings.length ? ` · ${leg.crossings.join(', ')}` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}