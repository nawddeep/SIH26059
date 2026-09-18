import React, { useEffect, useRef, useState } from 'react'
import { GripVertical, X, Navigation } from 'lucide-react'
import { searchPorts } from '../api'
import { flagEmoji, parseCoordLabel } from '../utils'

export default function WaypointRow({ index, waypoint, canRemove, onChange, onRemove, onDragStart, onDragOver, onDrop, dragOver }) {
  const [query, setQuery] = useState(waypoint.label || '')
  const [open, setOpen] = useState(false)
  const [options, setOptions] = useState([])
  const [coordOption, setCoordOption] = useState(null)
  const timer = useRef(null)
  const rowRef = useRef(null)

  useEffect(() => setQuery(waypoint.label || ''), [waypoint.id, waypoint.label])

  useEffect(() => {
    return () => clearTimeout(timer.current)
  }, [])

  function handleQuery(text) {
    setQuery(text)
    setOpen(text.length > 0)
    clearTimeout(timer.current)
    timer.current = setTimeout(async () => {
      if (!text.trim()) { setOptions([]); setCoordOption(null); return }
      const parsed = parseCoordLabel(text)
      setCoordOption(parsed)
      try {
        const res = await searchPorts(text.trim())
        setOptions(res || [])
      } catch {
        setOptions([])
      }
    }, 220)
  }

  function select(opt) {
    onChange({ label: opt.name, countryCode: opt.countryCode || '', lat: opt.lat, lon: opt.lon, isPort: !!opt.isPort })
    setOpen(false)
  }

  function selectCoord(parsed) {
    let label = `${parsed.lat.toFixed(2)}°${parsed.lat >= 0 ? 'N' : 'S'} ${parsed.lon.toFixed(2)}°${parsed.lon >= 0 ? 'E' : 'W'}`
    onChange({ label, countryCode: '', lat: parsed.lat, lon: parsed.lon, isPort: false })
    setOpen(false)
  }

  const placed = typeof waypoint.lat === 'number'

  return (
    <div
      ref={rowRef}
      className={`waypoint-row ${dragOver ? 'drag-over' : ''}`}
      draggable
      onDragStart={(e) => { e.dataTransfer.setData('text/plain', index); onDragStart(index) }}
      onDragOver={(e) => { e.preventDefault(); onDragOver(index) }}
      onDrop={(e) => { e.preventDefault(); onDrop(index) }}
    >
      <span className="wp-num">{index + 1}</span>
      <span className="wp-flag">{placed ? flagEmoji(waypoint.countryCode) : <span className="wp-flag-empty">·</span>}</span>

      <div className="wp-main">
        <input
          className="wp-input"
          value={query}
          placeholder={index === 0 ? 'Origin — search port or lat,lon…' : 'Destination or open-ocean coord…'}
          onChange={(e) => handleQuery(e.target.value)}
          onFocus={() => query && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && options.length) select(options[0])
            else if (e.key === 'Enter' && coordOption) selectCoord(coordOption)
          }}
        />
        {placed && (
          <span className="wp-coords">
            {waypoint.lat.toFixed(2)}°, {waypoint.lon.toFixed(2)}°
          </span>
        )}
        {open && (options.length > 0 || coordOption) && (
          <div className="autocomplete">
            {options.map((o) => (
              <button key={`${o.name}-${o.lat}-${o.lon}`} className="ac-item" onMouseDown={(e) => { e.preventDefault(); select(o) }}>
                <span className="ac-flag">{flagEmoji(o.countryCode)}</span>
                <span className="ac-name">{o.name}</span>
                <span className="ac-cc">{o.countryCode}</span>
                <span className="ac-coord">{o.lat.toFixed(2)}°, {o.lon.toFixed(2)}°</span>
              </button>
            ))}
            {coordOption && (
              <button className="ac-item ac-coord" onMouseDown={(e) => { e.preventDefault(); selectCoord(coordOption) }}>
                <Navigation size={13} />
                <span className="ac-name">Use coordinate {coordOption.lat.toFixed(2)}°, {coordOption.lon.toFixed(2)}°</span>
              </button>
            )}
          </div>
        )}
      </div>

      {canRemove && (
        <button className="wp-remove" tabIndex={-1} onMouseDown={(e) => { e.stopPropagation(); onRemove() }} title="Remove point">
          <X size={15} />
        </button>
      )}
      <span className="wp-grip"><GripVertical size={16} /></span>
    </div>
  )
}