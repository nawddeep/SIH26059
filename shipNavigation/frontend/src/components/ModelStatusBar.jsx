import React, { useEffect, useState } from 'react'
import { BrainCircuit, ChevronDown, CircleAlert, CircleCheck, CircleSlash } from 'lucide-react'
import { fetchModelStatus } from '../api'

const LABELS = {
  seaIce: 'Sea-ice forecast',
  icebergDrift: 'Iceberg drift',
  polarisRisk: 'POLARIS risk',
}

function Row({ name, info }) {
  const live = !!info.live
  return (
    <div className="model-row">
      <span className={`model-dot ${live ? 'live' : 'down'}`} />
      <div className="model-row-main">
        <span className="model-row-name">{LABELS[name] || name}</span>
        <span className="model-row-detail">
          {live ? info.model : (info.error || 'unavailable — using synthetic data')}
        </span>
      </div>
    </div>
  )
}

/**
 * Every model-backed endpoint falls back to synthetic data rather than failing,
 * so a map that renders is not evidence the models loaded. This asks the
 * backend to actually run them and shows which answer you are looking at.
 */
export default function ModelStatusBar() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetchModelStatus()
      .then((s) => { if (!cancelled) setStatus(s) })
      .catch((e) => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [])

  if (error) {
    return (
      <div className="model-status down">
        <CircleSlash size={14} />
        <span className="model-status-label">Model status unavailable</span>
      </div>
    )
  }

  if (!status) {
    return (
      <div className="model-status loading">
        <BrainCircuit size={14} />
        <span className="model-status-label">Checking models…</span>
      </div>
    )
  }

  const components = status.components || {}
  const total = Object.keys(components).length
  const liveCount = (status.liveComponents || []).length
  const allLive = !!status.allLive

  return (
    <div className={`model-status ${allLive ? 'live' : 'degraded'}`}>
      <button className="model-status-head" onClick={() => setOpen(!open)}>
        {allLive ? <CircleCheck size={14} /> : <CircleAlert size={14} />}
        <span className="model-status-label">
          {allLive ? 'Trained models live' : `${liveCount}/${total} models live`}
        </span>
        {status.forecastDate && (
          <span className="model-status-date">{status.forecastDate}</span>
        )}
        <ChevronDown size={13} className={`model-caret ${open ? 'open' : ''}`} />
      </button>

      {open && (
        <div className="model-status-body">
          {Object.entries(components).map(([name, info]) => (
            <Row key={name} name={name} info={info} />
          ))}

          {components.seaIce?.live && (
            <div className="model-metrics">
              <span>Ice cover <b>{components.seaIce.iceCoveredPct}%</b></span>
              <span>Mean SIC <b>{components.seaIce.meanConcentration}%</b></span>
              {components.seaIce.gridShape && (
                <span>Grid <b>{components.seaIce.gridShape.join('×')}</b></span>
              )}
            </div>
          )}

          <a className="model-tests-link" href="/model-tests.html" target="_blank" rel="noreferrer">
            Open model test dashboard →
          </a>

          {!allLive && (
            <p className="model-status-hint">
              Layers shown for offline models are simulated. Start the backend with
              the model venv (see RUN.md) to use the trained forecasts.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
