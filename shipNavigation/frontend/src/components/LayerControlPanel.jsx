import React, { useState } from 'react'
import { Layers, Waves, Wind, Compass, ShieldAlert, Navigation, ChevronRight, Eye, EyeOff } from 'lucide-react'
import { usePlanner } from '../store'

export default function LayerControlPanel() {
  const [isOpen, setIsOpen] = useState(true)
  const {
    showSeaIce, setShowSeaIce,
    showIceRisk, setShowIceRisk,
    polarClass, setPolarClass,
    showOceanCurrents, setShowOceanCurrents,
    showWind, setShowWind,
    showWeatherHeatmap, setShowWeatherHeatmap,
    showStorms, setShowStorms,
    weatherSubFilters, toggleWeatherSubFilter,
    showEca, setShowEca,
    showChokepoints, setShowChokepoints,
    showChevrons, setShowChevrons,
    showWindFlow, setShowWindFlow,
    showSeamarks, setShowSeamarks,
  } = usePlanner()

  const activeCount = [
    showSeaIce, showIceRisk, polarClass, showOceanCurrents, showWind, showWeatherHeatmap, showStorms,
    showEca, showChokepoints, showChevrons, showWindFlow, showSeamarks
  ].filter(Boolean).length

  return (
    <div className={`right-layer-bar ${isOpen ? 'open' : 'collapsed'}`}>
      <button
        type="button"
        className="layer-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? 'Hide Map Data Layers' : 'Show Map Data Layers'}
      >
        <Layers size={18} />
        <span className="layer-btn-text">Layers</span>
        <span className="layer-badge">{activeCount}</span>
        <ChevronRight size={14} className={`chevron-icon ${isOpen ? 'open' : ''}`} />
      </button>

      {isOpen && (
        <div className="layer-panel-content">
          <div className="layer-panel-header">
            <span className="lp-title">Map Data Selection</span>
            <div className="lp-quick-actions">
              <button
                type="button"
                className="lp-quick-btn"
                onClick={() => {
                  setShowSeaIce(true)
                  setShowOceanCurrents(true)
                  setShowWind(true)
                  setShowWeatherHeatmap(true)
                  setShowStorms(true)
                  setShowEca(true)
                  setShowChokepoints(true)
                  setShowChevrons(true)
                  setShowWindFlow(true)
                  setShowSeamarks(true)
                }}
              >
                All
              </button>
              <button
                type="button"
                className="lp-quick-btn"
                onClick={() => {
                  setShowSeaIce(false)
                  setShowOceanCurrents(false)
                  setShowWind(false)
                  setShowWeatherHeatmap(false)
                  setShowStorms(false)
                  setShowEca(false)
                  setShowChokepoints(false)
                  setShowChevrons(false)
                  setShowWindFlow(false)
                  setShowSeamarks(false)
                }}
              >
                None
              </button>
            </div>
          </div>

          <div className="layer-list">
            <label className="layer-item">
              <span className="layer-item-icon wind-flow"><Wind size={15} /></span>
              <span className="layer-item-info">
                <span className="layer-name">Wind Flow</span>
                <span className="layer-sub">Animated streamlines &middot; Open-Meteo live</span>
              </span>
              <input
                type="checkbox"
                checked={showWindFlow}
                onChange={(e) => setShowWindFlow(e.target.checked)}
              />
            </label>

            <label className="layer-item">
              <span className="layer-item-icon currents"><Waves size={15} /></span>
              <span className="layer-item-info">
                <span className="layer-name">Ocean Currents</span>
                <span className="layer-sub">Copernicus flow vectors (u, v)</span>
              </span>
              <input
                type="checkbox"
                checked={showOceanCurrents}
                onChange={(e) => setShowOceanCurrents(e.target.checked)}
              />
            </label>

            <label className="layer-item">
              <span className="layer-item-icon ice"><ShieldAlert size={15} /></span>
              <span className="layer-item-info">
                <span className="layer-name">POLARIS Ice Risk</span>
                <span className="layer-sub">IMO risk index &middot; {polarClass}</span>
              </span>
              <input
                type="checkbox"
                checked={showIceRisk}
                onChange={(e) => setShowIceRisk(e.target.checked)}
              />
            </label>

            {showIceRisk && (
              <label className="layer-item" style={{ paddingLeft: 30 }}>
                <span className="layer-item-info">
                  <span className="layer-name">Vessel Polar Class</span>
                  <span className="layer-sub">Lower class = stronger icebreaker</span>
                </span>
                <select
                  value={polarClass}
                  onChange={(e) => setPolarClass(e.target.value)}
                  style={{ fontSize: 12, padding: '2px 4px' }}
                >
                  {['PC1','PC2','PC3','PC4','PC5','PC6','PC7','UNCLASSED'].map(pc => (
                    <option key={pc} value={pc}>{pc}</option>
                  ))}
                </select>
              </label>
            )}

            <label className="layer-item">
              <span className="layer-item-icon eca"><ShieldAlert size={15} /></span>
              <span className="layer-item-info">
                <span className="layer-name">ECA Zones</span>
                <span className="layer-sub">Emission Control Areas</span>
              </span>
              <input
                type="checkbox"
                checked={showEca}
                onChange={(e) => setShowEca(e.target.checked)}
              />
            </label>

            <label className="layer-item">
              <span className="layer-item-icon chokepoint"><Navigation size={15} /></span>
              <span className="layer-item-info">
                <span className="layer-name">Chokepoints</span>
                <span className="layer-sub">Straits, canals & capes</span>
              </span>
              <input
                type="checkbox"
                checked={showChokepoints}
                onChange={(e) => setShowChokepoints(e.target.checked)}
              />
            </label>

            <div className="layer-item-group">
              <label className="layer-item">
                <span className="layer-item-icon wind" style={{ background: '#fef2f2', color: '#dc2626' }}>⛈️</span>
                <span className="layer-item-info">
                  <span className="layer-name">Storms & Weather</span>
                  <span className="layer-sub">Open-Meteo live hazard forecast</span>
                </span>
                <input
                  type="checkbox"
                  checked={showStorms}
                  onChange={(e) => setShowStorms(e.target.checked)}
                />
              </label>
              {showStorms && (
                <div className="weather-sub-filters">
                  {[
                    ['wind', '💨 Wind'],
                    ['waves', '🌊 Waves'],
                    ['storms', '⚡ Storms'],
                    ['rain', '🌧️ Rain'],
                    ['hazards', '🔴 Severe'],
                  ].map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      className={`sub-filter-chip ${weatherSubFilters[key] ? 'active' : ''}`}
                      onClick={() => toggleWeatherSubFilter(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <label className="layer-item">
              <span className="layer-item-icon seamarks"><Compass size={15} /></span>
              <span className="layer-item-info">
                <span className="layer-name">Seamarks</span>
                <span className="layer-sub">Buoys, lights & depth contours</span>
              </span>
              <input
                type="checkbox"
                checked={showSeamarks}
                onChange={(e) => setShowSeamarks(e.target.checked)}
              />
            </label>

            <label className="layer-item">
              <span className="layer-item-icon chevrons">➢</span>
              <span className="layer-item-info">
                <span className="layer-name">Route Chevrons</span>
                <span className="layer-sub">Directional route markers</span>
              </span>
              <input
                type="checkbox"
                checked={showChevrons}
                onChange={(e) => setShowChevrons(e.target.checked)}
              />
            </label>
          </div>
        </div>
      )}
    </div>
  )
}
