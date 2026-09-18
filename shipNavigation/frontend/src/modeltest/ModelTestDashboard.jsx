import React, { useEffect, useState } from 'react'
import Explorer from './Explorer'

const API = '/api/model-tests'

function Check({ check }) {
  return (
    <li className={`mt-check ${check.passed ? 'pass' : 'fail'}`}>
      <span className="mt-check-mark">{check.passed ? '✓' : '✕'}</span>
      <span className="mt-check-body">
        <span className="mt-check-name">{check.name}</span>
        <span className="mt-check-detail">{check.detail}</span>
      </span>
    </li>
  )
}

function Artifact({ label, a }) {
  if (!a) return null
  if (!a.exists) {
    return <div className="mt-artifact missing">{label}: not found at {a.path}</div>
  }
  return (
    <div className="mt-artifact">
      <div className="mt-artifact-row">
        <span className="mt-k">{label}</span>
        <code className="mt-path">{a.path}</code>
      </div>
      <div className="mt-artifact-meta">
        <span>{a.sizeMB} MB</span>
        <span>modified {a.modified.replace('T', ' ')}</span>
        {a.sha256 && <span className="mt-sha">SHA-256 {a.sha256}</span>}
      </div>
    </div>
  )
}

function SeaIceTable({ ev }) {
  return (
    <table className="mt-table">
      <thead>
        <tr>
          <th>Horizon</th><th>n</th><th>Model</th><th>Persistence</th><th>Climatology</th><th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        {ev.rows.map((r) => (
          <tr key={r.horizon}>
            <td>{r.horizon}</td>
            <td className="num">{r.nSamples}</td>
            <td className="num strong">{r.modelMaeIce}</td>
            <td className="num">{r.persistenceMaeIce}</td>
            <td className="num">{r.climatologyMaeIce}</td>
            <td>
              <span className={`mt-tag ${r.beatsPersistence ? 'good' : 'warn'}`}>
                {r.beatsPersistence ? 'beats persistence' : 'below persistence'}
              </span>
              <span className={`mt-tag ${r.beatsClimatology ? 'good' : 'warn'}`}>
                {r.beatsClimatology ? 'beats climatology' : 'below climatology'}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function DriftTable({ ev }) {
  return (
    <table className="mt-table">
      <thead>
        <tr>
          <th>Variant</th><th>RMS 24 h (km)</th><th>Median (km)</th><th>Within 10 km</th><th>Skill vs constant</th>
        </tr>
      </thead>
      <tbody>
        {ev.rows.map((r) => (
          <tr key={r.variant} className={r.shipped ? 'shipped' : ''}>
            <td>{r.variant.replace(/_/g, ' ')}{r.shipped && <span className="mt-tag good">shipped</span>}</td>
            <td className="num strong">{r.rmsErrorKm24h}</td>
            <td className="num">{r.medianErrorKm24h}</td>
            <td className="num">{r.within10kmPct}%</td>
            <td className="num">{r.skillVsConstant}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Component({ c }) {
  return (
    <section className="mt-card">
      <header className="mt-card-head">
        <div>
          <h2>{c.name}</h2>
          <p className="mt-arch">{c.architecture} · {c.framework}</p>
        </div>
        <div className="mt-badges">
          <span className={`mt-kind ${c.kind}`}>
            {c.kind === 'trained' ? 'trained model' : 'deterministic module'}
          </span>
          <span className={`mt-live ${c.live ? 'on' : 'off'}`}>
            {c.live ? 'loaded' : 'unavailable'}
          </span>
        </div>
      </header>

      <dl className="mt-io">
        <div><dt>Input</dt><dd>{c.inputs}</dd></div>
        <div><dt>Output</dt><dd>{c.output}</dd></div>
        {c.trainingData && <div><dt>Training data</dt><dd>{c.trainingData}</dd></div>}
        {c.parameters && <div><dt>Parameters</dt><dd>{c.parameters.toLocaleString()}</dd></div>}
        {c.features && <div><dt>Features</dt><dd>{c.features.length}: {c.features.slice(0, 6).join(', ')}…</dd></div>}
      </dl>

      {c.note && <p className="mt-note">{c.note}</p>}

      <Artifact label="Artifact" a={c.artifact} />
      <Artifact label="Dataset" a={c.dataset} />

      <h3 className="mt-sub">Checks run just now</h3>
      <ul className="mt-checks">
        {c.checks.map((ch, i) => <Check key={i} check={ch} />)}
      </ul>

      {c.evaluation && (
        <>
          <h3 className="mt-sub">Held-out evaluation</h3>
          <p className="mt-metric">{c.evaluation.metric}</p>
          {c.key === 'seaIce' ? <SeaIceTable ev={c.evaluation} /> : <DriftTable ev={c.evaluation} />}
          <p className="mt-source">Source: <code>{c.evaluation.source}</code></p>
        </>
      )}

      {c.caveat && <p className="mt-caveat"><strong>Known limitation.</strong> {c.caveat}</p>}
    </section>
  )
}

function Out({ v }) {
  if (v === null || v === undefined) return <span>—</span>
  if (typeof v !== 'object') return <code>{String(v)}</code>
  return (
    <span className="mt-out">
      {Object.entries(v).map(([k, val]) => (
        <span key={k} className="mt-out-pair">
          <em>{k}</em> <code>{Array.isArray(val) ? val.join('×') : String(val)}</code>
        </span>
      ))}
    </span>
  )
}

function Exports({ ex }) {
  if (!ex) return null
  if (!ex.available) {
    return (
      <section className="mt-card">
        <h2>Exported model files</h2>
        <p className="mt-caveat">Not built yet. {ex.hint}</p>
      </section>
    )
  }
  const loaded = ex.artifacts.filter((a) => a.loaded).length
  return (
    <section className="mt-card mt-exports">
      <header className="mt-card-head">
        <div>
          <h2>Exported model files (.pkl)</h2>
          <p className="mt-arch">
            Each file is loaded from disk and run when this page opens — the output
            below comes from the distributed artifact, not from the running server.
          </p>
        </div>
        <span className={`mt-live ${loaded === ex.artifacts.length ? 'on' : 'off'}`}>
          {loaded}/{ex.artifacts.length} load &amp; run
        </span>
      </header>

      <p className="mt-note">{ex.note}</p>
      {ex.loaderError && <p className="mt-caveat">{ex.loaderError}</p>}

      <div className="mt-export-grid">
        {ex.artifacts.map((a) => (
          <article key={a.key} className={`mt-export ${a.loaded ? '' : 'bad'}`}>
            <div className="mt-export-head">
              <code className="mt-export-file">{a.file}</code>
              <span className={`mt-kind ${a.kind}`}>{a.kind === 'trained' ? 'trained' : 'formula'}</span>
            </div>
            <p className="mt-export-name">{a.name}</p>

            <dl className="mt-export-meta">
              <div><dt>Size</dt><dd>{a.artifact.sizeMB} MB</dd></div>
              <div><dt>Requires</dt><dd>{a.requires.join(', ')}</dd></div>
              <div><dt>Trained on</dt><dd>{a.source}</dd></div>
              {a.loaded && <div><dt>Load</dt><dd>{a.loadMs} ms · predict {a.predictMs} ms</dd></div>}
            </dl>

            {a.loaded ? (
              <div className="mt-export-run">
                <div className="mt-io-row"><span className="mt-k">Input</span><span>{a.sampleInput}</span></div>
                <div className="mt-io-row"><span className="mt-k">Output</span><Out v={a.sampleOutput} /></div>
              </div>
            ) : (
              <p className="mt-export-err">{a.error}</p>
            )}

            <div className="mt-export-hash">
              <span className={`mt-tag ${a.hashMatches ? 'good' : 'warn'}`}>
                {a.hashMatches ? 'hash verified' : 'hash changed since export'}
              </span>
              <code>{a.artifact.sha256}</code>
            </div>
          </article>
        ))}
      </div>

      <p className="mt-source">
        Directory <code>{ex.directory}/</code> · load with <code>{ex.loadWith}</code>
      </p>
    </section>
  )
}

export default function ModelTestDashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  const load = () => {
    setData(null)
    setError('')
    fetch(API)
      .then((r) => r.json())
      .then((d) => (d.error ? setError(d.error) : setData(d)))
      .catch((e) => setError(e.message))
  }

  useEffect(load, [])

  if (error) {
    return (
      <div className="mt-wrap">
        <div className="mt-error">
          <h1>Model tests unavailable</h1>
          <p><code>{error}</code></p>
          <p>Start the backend with the model environment — see RUN.md — then reload.</p>
        </div>
      </div>
    )
  }

  if (!data) return <div className="mt-wrap"><p className="mt-loading">Loading models and running checks…</p></div>

  const s = data.summary
  const allPassed = s.checksPassed === s.checksRun

  return (
    <div className="mt-wrap">
      <header className="mt-header">
        <div>
          <h1>Model test dashboard</h1>
          <p className="mt-sub-title">
            Antarctic navigation decision support · every figure below is measured
            when this page loads or read from the evaluation file the training run wrote
          </p>
        </div>
        <button className="mt-rerun" onClick={load}>Re-run checks</button>
      </header>

      <div className="mt-summary">
        <div className="mt-stat">
          <span className="mt-stat-v">{s.trainedModels}</span>
          <span className="mt-stat-l">trained models</span>
        </div>
        <div className="mt-stat">
          <span className="mt-stat-v">{s.deterministicModules}</span>
          <span className="mt-stat-l">deterministic modules</span>
        </div>
        <div className="mt-stat">
          <span className="mt-stat-v">{s.live}/{s.components}</span>
          <span className="mt-stat-l">loaded successfully</span>
        </div>
        <div className={`mt-stat ${allPassed ? 'good' : 'warn'}`}>
          <span className="mt-stat-v">{s.checksPassed}/{s.checksRun}</span>
          <span className="mt-stat-l">checks passed</span>
        </div>
      </div>

      <p className="mt-generated">
        Generated {data.generatedAt} · model root <code>{data.modelRoot}</code>
      </p>

      <Explorer />

      <h2 className="mt-section-head">Evidence</h2>
      <p className="mt-section-note">
        Fixed samples and held-out scores: proof each artifact loads and runs, and
        what it measured against its baselines when it was trained.
      </p>

      <Exports ex={data.exports} />

      {data.components.map((c) => <Component key={c.key} c={c} />)}

      <footer className="mt-footer">
        <p>
          Two components are trained models with fitted weights on disk; the other
          two implement published formulas and have no learned parameters, so they
          are verified by property checks rather than an accuracy score.
        </p>
        <p>
          Verify any artifact yourself with{' '}
          <code>shasum -a 256 &lt;path&gt;</code> against the hash shown above.
        </p>
      </footer>
    </div>
  )
}
