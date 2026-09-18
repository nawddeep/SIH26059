/**
 * IMPALA Operational API Client
 * Connects frontend to the FastAPI backend with graceful fallback to local simulation.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8600'

export interface BackendHealth {
  status: string
  simulation_mode: boolean
  service?: string
  disclaimer?: string
}

async function safeFetch<T>(endpoint: string, options?: RequestInit): Promise<T | null> {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 2500)

    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    })

    clearTimeout(timeoutId)

    if (!res.ok) {
      console.warn(`[API] HTTP ${res.status} on ${endpoint}`)
      return null
    }

    return await res.json()
  } catch {
    // Graceful fallback on network error or timeout
    return null
  }
}

export async function checkBackendHealth(): Promise<BackendHealth | null> {
  return safeFetch<BackendHealth>('/api/health')
}

export async function fetchDashboardSnapshot(refresh = false): Promise<any | null> {
  return safeFetch(`/api/dashboard?refresh=${refresh}`)
}

export async function fetchVesselTelemetry(): Promise<any | null> {
  return safeFetch('/api/vessel')
}

export async function fetchIcebergs(): Promise<any | null> {
  return safeFetch('/api/icebergs')
}

export async function fetchIcebergById(id: string): Promise<any | null> {
  return safeFetch(`/api/icebergs/${encodeURIComponent(id)}`)
}

export async function fetchAisContacts(): Promise<any | null> {
  return safeFetch('/api/contacts/ais')
}

export async function fetchRadarContacts(): Promise<any | null> {
  return safeFetch('/api/contacts/radar')
}

export async function fetchEnvironment(): Promise<any | null> {
  return safeFetch('/api/environment')
}
