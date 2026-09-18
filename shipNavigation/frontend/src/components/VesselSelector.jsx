import React, { useState, useRef, useEffect } from 'react'
import { Ship, ChevronDown, Check, Anchor, Gauge, ShieldAlert } from 'lucide-react'
import { usePlanner } from '../store'

export const VESSEL_TYPES = [
  { id: 'cargo', label: 'Cargo Ships', defaultSpeed: 15, defaultDraft: 10.5, icon: '📦' },
  { id: 'tanker', label: 'Tankers', defaultSpeed: 14, defaultDraft: 14.5, icon: '🛢️' },
  { id: 'passenger', label: 'Passenger Ships', defaultSpeed: 20, defaultDraft: 7.5, icon: '🛳️' },
  { id: 'service', label: 'Service Ships', defaultSpeed: 12, defaultDraft: 5.0, icon: '⚓' },
  { id: 'fishing', label: 'Fishing', defaultSpeed: 10, defaultDraft: 3.5, icon: '🎣' },
  { id: 'icebreaker', label: 'Icebreakers', defaultSpeed: 13, defaultDraft: 9.0, icon: '🧊' },
  { id: 'research', label: 'Non-commercial Ships', defaultSpeed: 13, defaultDraft: 6.0, icon: '🔬' },
  { id: 'other', label: 'All Others', defaultSpeed: 12, defaultDraft: 6.0, icon: '🚤' },
]

export const ICE_CLASSES = [
  { id: 'none', label: 'Standard (No Ice Class)' },
  { id: 'pc1_pc7', label: 'Polar Class (PC1–PC7)' },
  { id: 'icebreaker', label: 'Heavy Icebreaker' },
]

export default function VesselSelector() {
  const {
    vesselType, setVesselType,
    draftMeters, setDraftMeters,
    iceClass, setIceClass,
    setSpeedKnots,
  } = usePlanner()

  const [open, setOpen] = useState(false)
  const dropdownRef = useRef(null)

  const activeVessel = VESSEL_TYPES.find(v => v.id === vesselType) || VESSEL_TYPES[0]

  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  function selectVessel(v) {
    setVesselType(v.id)
    setDraftMeters(v.defaultDraft)
    setSpeedKnots(v.defaultSpeed)
    if (v.id === 'icebreaker' && iceClass === 'none') {
      setIceClass('icebreaker')
    }
    setOpen(false)
  }

  return (
    <section className="section vessel-section">
      <div className="section-head">
        <span className="section-title">Vessel & Specs</span>
        <span className="section-hint">{activeVessel.label}</span>
      </div>

      <div className="vessel-dropdown-container" ref={dropdownRef}>
        <button
          type="button"
          className="vessel-select-trigger"
          onClick={() => setOpen(!open)}
        >
          <span className="vessel-icon">{activeVessel.icon}</span>
          <span className="vessel-label">{activeVessel.label}</span>
          <ChevronDown size={16} className={`vessel-arrow ${open ? 'open' : ''}`} />
        </button>

        {open && (
          <div className="vessel-dropdown-menu">
            {VESSEL_TYPES.map((v) => (
              <button
                key={v.id}
                type="button"
                className={`vessel-option ${v.id === vesselType ? 'active' : ''}`}
                onClick={() => selectVessel(v)}
              >
                <span className="v-check">{v.id === vesselType ? <Check size={14} /> : null}</span>
                <span className="v-icon">{v.icon}</span>
                <span className="v-name">{v.label}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="vessel-specs-row">
        <div className="spec-field">
          <label className="spec-label">
            <Anchor size={12} /> Draft
          </label>
          <div className="spec-input-wrap">
            <input
              type="number"
              min={1.0}
              max={28.0}
              step={0.5}
              value={draftMeters}
              onChange={(e) => setDraftMeters(Number(e.target.value))}
              className="spec-input"
            />
            <span className="spec-unit">m</span>
          </div>
        </div>

        <div className="spec-field">
          <label className="spec-label">
            <ShieldAlert size={12} /> Ice Class
          </label>
          <select
            className="spec-select"
            value={iceClass}
            onChange={(e) => setIceClass(e.target.value)}
          >
            {ICE_CLASSES.map((ic) => (
              <option key={ic.id} value={ic.id}>{ic.label}</option>
            ))}
          </select>
        </div>
      </div>
    </section>
  )
}
