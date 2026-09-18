# Bridge console (local dashboard)

A bridge-console view — vessel state, AIS, ice, icebergs, routes, risk, sensors —
for the Antarctic navigation system in this repository.

## What changed when this was brought in

**Its simulation backend was dropped, not imported.** The original
`backend/app/services/simulation.py` generated vessel speed, wind and contact
bearings with `random.random()`. This repository removed its own simulated data
layers for the same reason, and re-importing one through a dashboard would have
undone that.

The console is now served by the real backend at
`shipNavigation/backend/app/bridge_console.py`, on **port 8600**:

| endpoint | real source |
|---|---|
| `/api/vessel` | shipboard telemetry gateway |
| `/api/contacts/ais` | telemetry gateway AIS feed |
| `/api/environment` | ship weather station + U-Net+ConvLSTM sea-ice forecast |
| `/api/icebergs` | trained two-stage drift model |
| `/api/dashboard` | all of the above |
| `/api/contacts/radar` | **none — returns empty with a reason** |

**There is no radar feed in this system.** That endpoint returns an empty list
and says so. A console drawing invented radar contacts is worse than one showing
none, because the officer cannot tell which is which.

**The unused 102 MB Qwen3-8B LoRA adapter was not brought in.** Nothing in this
frontend or backend references it.

> **Status:** the console shows real telemetry and real model output. It is not
> connected to live ship hardware, and the sea-ice field is historical
> reanalysis ending 2018-12-31, so weather and ice are not contemporaneous.

## Running it

```bash
# 1. the real backend
cd ../shipNavigation && ./start.sh          # serves :8600

# 2. this console
cd ../local-dashboard && npm install && npm run dev
```

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       IMPALA FRONTEND                       │
│  React 19 + TypeScript + MapLibre GL JS + Custom Design Sys │
│  (Overview Radar, Map, Ice, Icebergs, Routes, Risk, Sensors)│
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON (Port 8000)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND SERVICE                  │
│       Python 3.11+ / Uvicorn API (`backend/app/main.py`)    │
│  - /api/health          - /api/vessel       - /api/icebergs │
│  - /api/contacts/ais    - /api/contacts/radar               │
│  - /api/environment     - /api/dashboard                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 SIMULATION & DATA LAYER                     │
│  Deterministic Kinematics, Metocean & Cryosphere Physics    │
│  `backend/data/` (vessel.json, icebergs.json, contacts.json)│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Quick Start (Running Full Stack)

### Step 1: Start the Backend (FastAPI)
```bash
# Option A: From root using virtual environment
.venv/bin/uvicorn backend.app.main:app --reload --port 8000

# Option B: From backend folder
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
* **API Index**: `http://localhost:8000/`
* **Health Check**: `http://localhost:8000/api/health`
* **Swagger API Docs**: `http://localhost:8000/docs`

---

### Step 2: Start the Frontend (Vite + React)
In a separate terminal:
```bash
npm install
npm run dev
```
* **Bridge Dashboard UI**: `http://localhost:5180` (or `http://localhost:5173`)

---

## 3. Key Backend Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status |
| `GET` | `/api/dashboard` | Consolidated telemetry snapshot for UI (`?refresh=true` triggers simulation tick) |
| `GET` | `/api/vessel` | Own-vessel `RV BHARATI` position, SOG, COG, heading, engine status |
| `GET` | `/api/contacts/ais` | Tracked AIS maritime vessels with CPA/TCPA |
| `GET` | `/api/contacts/radar` | X-band radar acoustic & target contacts |
| `GET` | `/api/icebergs` | Full tracked iceberg registry with 72h predicted trajectories |
| `GET` | `/api/icebergs/{id}` | Detailed target dossier (e.g. `ICE-042`) |
| `GET` | `/api/environment` | Wind, wave, sea ice concentration, and composite risk metrics |

---

## 4. Production Build

To build the client bundle for air-gapped deployment:
```bash
npm run build
```
Generates self-contained static assets in `dist/`.
