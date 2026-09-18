import React from 'react'
import ReactDOM from 'react-dom/client'
import { PlannerProvider } from './store'
import App from './App'
import './styles.css'

// NOTE: intentionally no <StrictMode> wrapper. StrictMode double-invokes the
// MapView mount effect in dev, creating a second, temp MapLibre instance whose
// async `load` throws when it calls addSource() on the first (already-removed)
// map — killing the route/waypoints/overlay layers. The map is created
// imperatively once; priming a fresh map per page load is fine.
ReactDOM.createRoot(document.getElementById('root')).render(
  <PlannerProvider>
    <App />
  </PlannerProvider>,
)