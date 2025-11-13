# Multimodal Flight Assistant

`app.py` is a one-file FastAPI app that:

- streams **X-Plane 12** telemetry via your REST/WebSocket bridge server,
- opens an **OpenAI Realtime** voice session in the browser (uses a single multimodal model)
- lets you **attach cockpit screenshots** (server-side) for vision, while using a tool call for numeric telemetry or runs both automatically at intervals you set.

> **Note**: server-side screenshots use Windows APIs (`mss`, `pygetwindow`, `pywin32`).  
> On macOS/Linux the `/screenshot` route will return `503`.

---

## Prerequisites

- **X-Plane 12** with your telemetry bridge server running  
  (REST/WebSocket server exposing sim variables on `http://127.0.0.1:8086` or similar).
- **Python 3.10+** on Windows (for screenshot support).
- An **OpenAI API key** with access to Realtime models.
- A modern desktop browser (Chrome/Edge) with microphone access enabled.

---

## Quickstart

```bash
# clone
git clone https://github.com/vrtnis/multimodal-voice-assistant.git
cd multimodal-voice-assistant

# create & activate venv (PowerShell on Windows)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt


### Set environment variables (PowerShell)

```powershell
$Env:OPENAI_API_KEY = "sk-..."           # required
# If your X-Plane REST bridge is not on 127.0.0.1:8086, set:
# $Env:XPLANE_HOST = "127.0.0.1"
# $Env:XPLANE_PORT = "8111"
# Optional: choose a Realtime model (vision-capable)
# $Env:OPENAI_REALTIME_MODEL = "gpt-realtime"

### Run the server
uvicorn app:app --reload

Open http://127.0.0.1:8000, click Connect, then Snap & Send to attach a cockpit screenshot or select autosend every n seconds.
