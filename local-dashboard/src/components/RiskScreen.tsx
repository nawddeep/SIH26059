import { useState } from 'react'
import { useSimulation } from '../data/simulationStore'

export function RiskScreen() {
  const { state } = useSimulation()
  const { riskCategories, compositeRiskScore, compositeRiskTrend, compositeRiskDelta, vessel, hazards } = state

  const [selectedCategoryId, setSelectedCategoryId] = useState<string>('iceberg-encounter')
  const [verifiedItems, setVerifiedItems] = useState<Set<string>>(new Set(['SECONDARY NAVIGATION INTEGRITY']))

  const selectedCategory =
    riskCategories.find((rc) => rc.id === selectedCategoryId) || riskCategories[0]

  const toggleVerification = (title: string) => {
    setVerifiedItems((prev) => {
      const next = new Set(prev)
      if (next.has(title)) next.delete(title)
      else next.add(title)
      return next
    })
  }

  const verifications = [
    { title: 'ICE-042 CORRIDOR REVIEW', action: 'Inspect radar and optical bearing against ICE-042 drift model.', status: 'URGENT' },
    { title: 'PACK ICE PRESSURE WATCH', action: 'Monitor engine load and hull resistance in 84% pack ice.', status: 'ACTIVE' },
    { title: 'SONAR-014 RETURN VERIFICATION', action: 'Confirm forward acoustic return classification (Depth −31m).', status: 'ACTIVE' },
    { title: 'SECONDARY NAVIGATION INTEGRITY', action: 'Cross-reference GNSS fixes against radar landmarks.', status: 'ROUTINE' },
  ]

  return (
    <div className="demo-page risk-page">
      {/* Top Banner */}
      <div className="page-header-banner">
        <div>
          <span className="eyebrow">INTEGRATED MARITIME HAZARD INDEX & RISK DECOMPOSITION · {vessel.name}</span>
          <h2>OPERATIONAL RISK MATRIX / {vessel.name} EXPEDITION SECTOR</h2>
        </div>
        <div className="banner-stats">
          <div>
            <small>COMPOSITE RISK INDEX</small>
            <strong className={compositeRiskScore > 75 ? 'critical' : compositeRiskScore > 50 ? 'warning' : ''}>
              {compositeRiskScore} / 100
            </strong>
          </div>
          <div>
            <small>RISK TREND</small>
            <strong className={compositeRiskTrend === 'INCREASING' ? 'critical' : 'nominal'}>
              {compositeRiskDelta} ({compositeRiskTrend})
            </strong>
          </div>
          <div>
            <small>RISK CLASSIFICATION</small>
            <span className={`badge ${compositeRiskScore > 70 ? 'critical' : 'warning'}`}>
              {compositeRiskScore > 70 ? 'HIGH RISK ADVISORY' : 'MODERATE RISK'}
            </span>
          </div>
          <div>
            <small>DATA SOURCE</small>
            <span className="mode-tag">● SIMULATION</span>
          </div>
        </div>
      </div>

      <div className="demo-grid-layout">
        {/* Left Column: Risk Gauge & Category Breakdown */}
        <div className="demo-column main-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">HAZARD VECTOR BREAKDOWN · CLICK ROW TO INSPECT CONTRIBUTING FACTORS</span>
                <h2>SECTOR RISK FACTORS (EVALUATED AGAINST SCIENTIFIC BASELINES)</h2>
              </div>
              <span className="fresh">{riskCategories.length} VECTORS MONITORED</span>
            </div>

            <div className="risk-matrix-table-box">
              <table className="demo-table clickable-table">
                <thead>
                  <tr>
                    <th>HAZARD CATEGORY</th>
                    <th>RISK SCORE</th>
                    <th>WEIGHT</th>
                    <th>CLASSIFICATION</th>
                    <th>6H TREND</th>
                    <th>PRIMARY CONTRIBUTING FACTOR</th>
                  </tr>
                </thead>
                <tbody>
                  {riskCategories.map((rc) => {
                    const isSel = rc.id === selectedCategoryId
                    return (
                      <tr
                        key={rc.id}
                        className={isSel ? 'selected-row' : ''}
                        onClick={() => setSelectedCategoryId(rc.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td>
                          <b>{rc.name}</b>
                        </td>
                        <td>
                          <div className="score-cell">
                            <strong className={rc.score > 75 ? 'critical' : rc.score > 50 ? 'warning' : ''}>
                              {rc.score}
                            </strong>
                            <div className="mini-progress">
                              <div
                                className={`fill ${
                                  rc.score > 75 ? 'critical' : rc.score > 50 ? 'warning' : 'nominal'
                                }`}
                                style={{ width: `${rc.score}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td>{(rc.weight * 100).toFixed(0)}%</td>
                        <td>
                          <span className={`status-pill ${rc.level.toLowerCase()}`}>{rc.level}</span>
                        </td>
                        <td>
                          <small>{rc.trend}</small>
                        </td>
                        <td>
                          <span className="risk-desc">{rc.desc}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Selected Hazard Factor Deep Dive */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">VECTOR DECOMPOSITION · {selectedCategory.name}</span>
                <h2>CONTRIBUTING HAZARD PARAMETERS & MITIGATION</h2>
              </div>
              <span className={`status-pill ${selectedCategory.level.toLowerCase()}`}>
                {selectedCategory.level} ({selectedCategory.score}/100)
              </span>
            </div>
            <div className="risk-detail-body">
              <p>
                <b>DESCRIPTION:</b> {selectedCategory.desc}
              </p>
              <div className="risk-factors-tags">
                <span className="eyebrow">ACTIVE METRICS:</span>
                {selectedCategory.contributingFactors.map((f) => (
                  <span className="metric-tag" key={f}>
                    {f}
                  </span>
                ))}
              </div>
              <div className="mitigation-box">
                <span className="eyebrow">RECOMMENDED MITIGATION ACTION:</span>
                <p>{selectedCategory.mitigation}</p>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Active Hazards & Verification Items */}
        <div className="demo-column side-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">OPEN HAZARDS</span>
                <h2>ACTIVE MARITIME ALERTS</h2>
              </div>
              <span className="alert-dot">▲</span>
            </div>
            <div className="hazard-list">
              {hazards.map((h) => (
                <article className={`hazard ${h.level.toLowerCase()}`} key={h.title}>
                  <div>
                    <b>{h.level}</b>
                    <strong>{h.title}</strong>
                  </div>
                  <p>{h.message}</p>
                  <button className="text-btn">{h.action} →</button>
                </article>
              ))}
            </div>
          </section>

          {/* Interactive Bridge Verification Checklist */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">BRIDGE VERIFICATION PROTOCOLS</span>
                <h2>DECISION CHECKLIST</h2>
              </div>
              <span className="fresh">
                {verifiedItems.size} / {verifications.length} COMPLETE
              </span>
            </div>
            <div className="verification-list">
              {verifications.map((v) => {
                const isVerified = verifiedItems.has(v.title)
                return (
                  <div
                    key={v.title}
                    className={`verification-item clickable-row ${isVerified ? 'verified-item' : ''}`}
                    onClick={() => toggleVerification(v.title)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="verif-head">
                      <b>{v.title}</b>
                      <span className={`status-pill ${isVerified ? 'nominal' : v.status.toLowerCase()}`}>
                        {isVerified ? '✓ VERIFIED' : v.status}
                      </span>
                    </div>
                    <p>{v.action}</p>
                  </div>
                )
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
