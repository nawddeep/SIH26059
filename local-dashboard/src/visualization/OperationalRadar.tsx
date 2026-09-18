import { useEffect, useRef } from 'react'
import type { OperationalSnapshot } from '../types/operational'

type Props = { state: OperationalSnapshot; selectedId: string; onSelect: (id: string) => void }

export function OperationalRadar({ state, selectedId, onSelect }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const node = canvas.current, context = node?.getContext('2d'); if (!node || !context) return
    let frame = 0; const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const resize = () => { const r = node.getBoundingClientRect(), d = window.devicePixelRatio || 1; node.width = Math.max(1, r.width * d); node.height = Math.max(1, r.height * d); context.setTransform(d, 0, 0, d, 0, 0) }
    const observer = new ResizeObserver(resize); observer.observe(node); resize()
    const draw = (time: number) => {
      const width = node.clientWidth, height = node.clientHeight, cx = width / 2, cy = height / 2, radius = Math.min(width, height) * .405, sweep = reduced ? -Math.PI / 2 : time / 1450 - Math.PI / 2
      context.clearRect(0, 0, width, height); context.fillStyle = '#061012'; context.fillRect(0, 0, width, height); context.save(); context.translate(cx, cy)
      context.strokeStyle = 'rgba(100,172,169,.45)'; context.lineWidth = 1; [.25,.5,.75,1].forEach((f) => { context.beginPath(); context.arc(0, 0, radius * f, 0, Math.PI * 2); context.stroke() })
      context.strokeStyle = 'rgba(78,133,137,.5)'; for (let i = 0; i < 16; i++) { const a = i * Math.PI * 2 / 16; context.beginPath(); context.moveTo(0, 0); context.lineTo(Math.cos(a) * radius, Math.sin(a) * radius); context.stroke() }
      const fade = context.createRadialGradient(0,0,0,0,0,radius); fade.addColorStop(0,'rgba(86,197,190,.22)'); fade.addColorStop(1,'rgba(86,197,190,0)'); context.fillStyle = fade; context.beginPath(); context.moveTo(0,0); context.arc(0,0,radius,sweep-.32,sweep); context.closePath(); context.fill(); context.strokeStyle='#9adfda'; context.lineWidth=1.2; context.beginPath(); context.moveTo(0,0); context.lineTo(Math.cos(sweep)*radius,Math.sin(sweep)*radius); context.stroke()
      context.strokeStyle='#7ebda5'; context.setLineDash([6,5]); context.beginPath(); context.moveTo(-radius*.55,radius*.66); context.lineTo(0,0); context.lineTo(radius*.35,-radius*.73); context.stroke(); context.setLineDash([])
      state.icebergs.forEach((ice) => { const x=(ice.x-50)*radius/48,y=(ice.y-57)*radius/48,active=ice.id===selectedId; context.strokeStyle=ice.risk==='high'?'#d46b4e':'#d8a164'; context.fillStyle='#091719'; context.lineWidth=active?2.4:1.4; context.beginPath(); context.arc(x,y,active?7:5,0,Math.PI*2); context.fill(); context.stroke(); context.beginPath(); context.moveTo(x,y); context.lineTo(x+Math.cos(ice.heading*Math.PI/180)*17,y+Math.sin(ice.heading*Math.PI/180)*17); context.stroke(); context.fillStyle=ice.risk==='high'?'#e09a78':'#e6bb7d'; context.font='10px SFMono-Regular, Consolas, monospace'; context.fillText(ice.id,x+9,y-8) })
      state.radar.forEach((c) => { const x=(c.x-50)*radius/48,y=(c.y-57)*radius/48; context.strokeStyle='#84d2d0'; context.beginPath(); context.arc(x,y,3,0,Math.PI*2); context.stroke() }); state.sonar.forEach((c) => { const x=(c.x-50)*radius/48,y=(c.y-57)*radius/48; context.fillStyle=c.returnStrength==='strong'?'#d89b55':'#7eb9c0'; context.beginPath(); context.arc(x,y,c.returnStrength==='strong'?4:2.5,0,Math.PI*2); context.fill() })
      context.fillStyle='#dceae5'; context.beginPath(); context.moveTo(0,-12); context.lineTo(7,10); context.lineTo(0,6); context.lineTo(-7,10); context.closePath(); context.fill(); context.fillStyle='#afc9c4'; context.font='10px SFMono-Regular, Consolas, monospace'; context.fillText('N',-3,-radius-12); context.fillText('S',-3,radius+17); ['6 NM','12 NM','18 NM','24 NM'].forEach((label,i)=>context.fillText(label,5,-radius*(.25+i*.25))); context.restore(); if (!reduced) frame=requestAnimationFrame(draw)
    }; frame=requestAnimationFrame(draw); return () => { cancelAnimationFrame(frame); observer.disconnect() }
  }, [state, selectedId])
  return <div className="radar-console" aria-label="Animated IMPALA operational radar"><canvas ref={canvas} className="radar-canvas" onClick={(event) => { const r=event.currentTarget.getBoundingClientRect(),x=(event.clientX-r.left)/r.width*100,y=(event.clientY-r.top)/r.height*100,target=state.icebergs.reduce((best,item)=>Math.hypot(item.x-x,item.y-y)<Math.hypot(best.x-x,best.y-y)?item:best,state.icebergs[0]); if(target) onSelect(target.id) }} /><div className="radar-readout radar-nw">SIMULATED X-BAND / 24 NM<br /><span>HDG {state.vessel.heading}° · SOG {state.vessel.sog} KT</span></div><div className="radar-readout radar-se">RV BHARATI / CENTRAL TRACK<br /><span>SWEEP ACTIVE · SONAR FWD</span></div></div>
}
