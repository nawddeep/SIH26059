/**
 * The interactive half of the dashboard.
 *
 * Everything here is the exported .pkl answering a question the reader asked.
 * The evidence panels below it replay a fixed sample and prove the artifact
 * loads; this panel proves it responds - change a hull class or a date and the
 * numbers come back from `predict()` on the file in model_exports/.
 *
 * Each card carries its own control row because each card is a different model
 * with a different domain: a Polar Class means nothing to the sea-ice grid, and
 * the drift archive has 364 usable days against the forecaster's eleven years.
 * One shared filter row would have to be the intersection of all of that, which
 * is empty.
 */
import React, { useEffect, useRef, useState } from 'react'
import { CurveChart, DriftField, SicHeatmap } from './charts'

const CLASSES = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'UNCLASSED']

async function get(url) {
  const r = await fetch(url)
  const d = await r.json()
  if (d.error) throw new Error(d.error)
  return d
}

function Stat({ label, value, unit, tone }) {
  return (
    <div className={`mt-xstat ${tone || ''}`}>
      <span className="mt-xstat-v">{value}{unit && <em>{unit}</em>}</span>
      <span className="mt-xstat-l">{label}</span>
    </div>
  )
}

/** Risk as a meter: the track is a lighter step of the fill's own ramp. */
function RiskMeter({ risk }) {
  const pct = Math.round(risk * 100)
  const tone = risk >= 0.66 ? 'crit' : risk >= 0.33 ? 'warn' : 'ok'
  const word = risk >= 0.66 ? 'Elevated' : risk >= 0.33 ? 'Moderate' : 'Low'
  return (
    <div className="mt-meter-wrap">
      <div className="mt-meter-head">
        <span className="mt-xstat-l">POLARIS risk index</span>
        <span className={`mt-meter-word ${tone}`}>{word} · {risk.toFixed(4)}</span>
      </div>
      <div className="mt-meter" role="meter" aria-valuenow={pct} aria-valuemin="0" aria-valuemax="100">
        <i className={tone} style={{ width: `${Math.max(2, pct)}%` }} />
      </div>
    </div>
  )
}

function Err({ message, what }) {
  return (
    <p className="mt-caveat">
      <strong>{what} unavailable.</strong> <code>{message}</code> — start the backend with
      the model venv (see RUN.md), and build the artifacts with
      {' '}<code>backend/scripts/export_model_pickles.py</code>.
    </p>
  )
}

/* ------------------------------------------------- deterministic artifacts */
function IceConditions() {
  const [polarClass, setPolarClass] = useState('PC4')
  const [sic, setSic] = useState(0.7)
  const [distanceKm, setDistanceKm] = useState(25)
  const [curves, setCurves] = useState(null)
  const [point, setPoint] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let live = true
    setBusy(true)
    get(`/api/model-explorer/curves?distanceKm=${distanceKm}`)
      .then((d) => live && setCurves(d))
      .catch((e) => live && setErr(e.message))
      .finally(() => live && setBusy(false))
    return () => { live = false }
  }, [distanceKm])

  // The marker tracks the slider off the swept curve, so dragging is instant;
  // this call fills the readout with the artifact's answer for the exact value.
  useEffect(() => {
    const t = setTimeout(() => {
      get(`/api/model-explorer/point?sic=${sic}&polarClass=${polarClass}&distanceKm=${distanceKm}`)
        .then(setPoint)
        .catch((e) => setErr(e.message))
    }, 120)
    return () => clearTimeout(t)
  }, [sic, polarClass, distanceKm])

  if (err) return <section className="mt-card"><h2>Ice conditions</h2><Err message={err} what="Explorer" /></section>
  if (!curves) return <section className="mt-card"><p className="mt-loading">Loading curves from the artifacts…</p></section>

  const sel = curves.series[polarClass]
  const others = CLASSES.filter((c) => c !== polarClass).map((c) => ({ label: c, values: curves.series[c].risk }))
  const at = (arr) => {
    const i = Math.round(sic * (curves.sic.length - 1))
    return arr[Math.max(0, Math.min(arr.length - 1, i))]
  }

  return (
    <section className={`mt-card mt-explorer ${busy ? 'busy' : ''}`}>
      <header className="mt-card-head">
        <div>
          <h2>Ice conditions — risk and fuel</h2>
          <p className="mt-arch">
            <code>polaris_risk.pkl</code> and <code>fuel_consumption.pkl</code>, evaluated
            live. Move the slider and every figure below is that artifact's own answer.
          </p>
        </div>
        <span className="mt-kind deterministic">formula</span>
      </header>

      <div className="mt-controls">
        <label>
          <span>Vessel Polar Class</span>
          <select value={polarClass} onChange={(e) => setPolarClass(e.target.value)}>
            {CLASSES.map((c) => <option key={c} value={c}>{c === 'UNCLASSED' ? 'Unclassed' : c}</option>)}
          </select>
        </label>
        <label className="wide">
          <span>Sea-ice concentration <b>{sic.toFixed(2)}</b></span>
          <input
            type="range" min="0" max="1" step="0.01" value={sic}
            onChange={(e) => setSic(Number(e.target.value))}
          />
        </label>
        <label>
          <span>Cell length (km)</span>
          <input
            type="number" min="1" max="500" step="1" value={distanceKm}
            onChange={(e) => setDistanceKm(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
          />
        </label>
      </div>

      {point && (
        <>
          <RiskMeter risk={point.risk} />
          <div className="mt-xstats">
            <Stat label={`Fuel burn over ${point.distanceKm} km`} value={point.fuelTonnes} unit=" t" />
            <Stat label="Attainable speed" value={point.speedKnots} unit=" kn" />
            <Stat label="Transit time" value={point.transitHours} unit=" h" />
            <Stat
              label="Versus open water" value={`${point.fuelPenaltyX}×`}
              tone={point.fuelPenaltyX >= 2 ? 'warn' : ''}
            />
          </div>
        </>
      )}

      <div className="mt-chart-grid">
        <CurveChart
          x={curves.sic} values={sel.risk} context={others}
          label="Navigational risk rises with ice, faster for a weaker hull"
          yLabel="POLARIS risk index" unit="" decimals={3}
          marker={{ x: sic, y: at(sel.risk), seriesLabel: polarClass }}
        />
        <CurveChart
          x={curves.sic} values={sel.fuelTonnes}
          context={CLASSES.filter((c) => c !== polarClass).map((c) => ({ label: c, values: curves.series[c].fuelTonnes }))}
          label={`Fuel burn over a ${curves.distanceKm} km cell`}
          yLabel="Tonnes" unit=" t" decimals={2}
          marker={{ x: sic, y: at(sel.fuelTonnes), seriesLabel: polarClass }}
        />
        <CurveChart
          x={curves.sic} values={sel.speedKnots}
          context={CLASSES.filter((c) => c !== polarClass).map((c) => ({ label: c, values: curves.series[c].speedKnots }))}
          label="Attainable speed collapses as concentration climbs"
          yLabel="Knots" unit=" kn" decimals={2}
          marker={{ x: sic, y: at(sel.speedKnots), seriesLabel: polarClass }}
        />
      </div>

      <p className="mt-source">
        Grey lines are the same quantity for the other seven hull classes, for scale.
        Source <code>{curves.source}</code>.
      </p>
    </section>
  )
}

/* --------------------------------------------------------- sea ice field */
function SeaIcePanel() {
  const [date, setDate] = useState('2018-12-20')
  const [field, setField] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const last = useRef(null)

  useEffect(() => {
    let live = true
    setBusy(true); setErr('')
    get(`/api/model-explorer/seaice?date=${date}`)
      .then((d) => { if (live) { setField(d); last.current = d } })
      .catch((e) => live && setErr(e.message))
      .finally(() => live && setBusy(false))
    return () => { live = false }
  }, [date])

  const shown = field || last.current

  return (
    <section className={`mt-card mt-explorer ${busy ? 'busy' : ''}`}>
      <header className="mt-card-head">
        <div>
          <h2>Sea-ice forecaster — run it on a date</h2>
          <p className="mt-arch">
            <code>seaice_forecast.pkl</code> takes the seven days before the date you pick
            and returns the next day's concentration over the whole 332×316 grid.
          </p>
        </div>
        <span className="mt-kind trained">trained</span>
      </header>

      <div className="mt-controls">
        <label>
          <span>Forecast date</span>
          <input
            type="date" value={date} min="2008-01-07" max="2018-12-31"
            onChange={(e) => e.target.value && setDate(e.target.value)}
          />
        </label>
        <p className="mt-control-note">
          The processed archive runs 2008-01-01 to 2018-12-31, so every date here is
          historical reanalysis — this is not a live feed.
        </p>
      </div>

      {err && <Err message={err} what="Forecast" />}
      {shown && <SicHeatmap field={shown} />}
      {shown && (
        <div className="mt-xstats">
          <Stat label="Mean concentration over ocean" value={shown.stats.meanOcean} />
          <Stat label="Peak concentration" value={shown.stats.max} />
          <Stat label="Ice extent" value={shown.stats.extentMkm2} unit=" M km²" />
          <Stat label="Ice area" value={shown.stats.iceAreaMkm2} unit=" M km²" />
        </div>
      )}
      {shown && <p className="mt-source">Source <code>{shown.source}</code>. {shown.note}</p>}
    </section>
  )
}

/* ----------------------------------------------------------- drift field */
function DriftPanel() {
  const [date, setDate] = useState('')
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const last = useRef(null)

  useEffect(() => {
    let live = true
    setBusy(true); setErr('')
    get(`/api/model-explorer/drift?date=${date}&limit=400`)
      .then((d) => { if (live) { setData(d); last.current = d } })
      .catch((e) => live && setErr(e.message))
      .finally(() => live && setBusy(false))
    return () => { live = false }
  }, [date])

  const shown = data || last.current

  return (
    <section className={`mt-card mt-explorer ${busy ? 'busy' : ''}`}>
      <header className="mt-card-head">
        <div>
          <h2>Iceberg drift — run it on a day of fixes</h2>
          <p className="mt-arch">
            <code>iceberg_drift.pkl</code> gates each berg moving or stationary, then
            predicts 24 h displacement for the ones it gated as moving.
          </p>
        </div>
        <span className="mt-kind trained">trained</span>
      </header>

      <div className="mt-controls">
        <label>
          <span>Day of NIC fixes</span>
          <select value={date} onChange={(e) => setDate(e.target.value)}>
            <option value="">Most recent crowded day</option>
            {(shown?.availableDates || []).slice().reverse().map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <p className="mt-control-note">
          Only days carrying eight or more simultaneous fixes are listed — fewer than
          that is a scatter of dots, not a field.
        </p>
      </div>

      {err && <Err message={err} what="Drift field" />}
      {shown && <DriftField data={shown} />}
      {shown && (
        <div className="mt-xstats">
          <Stat label="Bergs on this day" value={shown.count} />
          <Stat label="Gated as moving" value={`${shown.movingCount}/${shown.count}`} />
          <Stat label="Mean drift, moving bergs" value={shown.meanMovingDriftKm24h} unit=" km" />
          <Stat label="Largest predicted drift" value={shown.maxDriftKm24h} unit=" km" />
        </div>
      )}
      {shown && (
        <p className="mt-source">
          Source <code>{shown.source}</code>. Most tracked bergs are grounded or
          fast-locked on any given day, so a majority gated stationary is the expected
          result, not a failure.
        </p>
      )}
    </section>
  )
}

export default function Explorer() {
  return (
    <>
      <h2 className="mt-section-head">Run the artifacts</h2>
      <p className="mt-section-note">
        Each panel loads its <code>.pkl</code> from <code>model_exports/</code> and calls
        <code> predict()</code> on the inputs you choose. Nothing below is precomputed.
      </p>
      <IceConditions />
      <SeaIcePanel />
      <DriftPanel />
    </>
  )
}
