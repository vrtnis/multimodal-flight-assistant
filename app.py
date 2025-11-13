# app.py  — server-side screenshot edition

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import base64
import io

import requests
import websockets
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# NEW: server-side screenshot libs (Windows)
try:
    from PIL import Image
    import mss
    import pygetwindow as gw
except Exception:
    # Dont hard-fail on import so the app still runs without screenshots
    Image = None
    mss = None
    gw = None

HOST = os.environ.get("XPLANE_HOST", "127.0.0.1")
PORT = int(os.environ.get("XPLANE_PORT", "8086"))
BASE = f"http://{HOST}:{PORT}"
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
WEB_SEARCH_MODEL = os.environ.get("OPENAI_WEB_SEARCH_MODEL", "gpt-5")
WEB_SEARCH_CONNECT_TIMEOUT = float(os.environ.get("OPENAI_WEB_SEARCH_CONNECT_TIMEOUT", "10"))
WEB_SEARCH_READ_TIMEOUT = float(os.environ.get("OPENAI_WEB_SEARCH_READ_TIMEOUT", "60"))

# --- screenshot tuning (env overridable) ---
XPLANE_WINDOW_TITLES = [
    t.strip() for t in os.environ.get("XPLANE_WINDOW_TITLES", "X-System,X-Plane").split(",") if t.strip()
]
SCREENSHOT_JPEG_QUALITY = int(os.environ.get("SCREENSHOT_JPEG_QUALITY", "85"))

XPLANE_EXCLUDE_TITLES = [
    t.strip().lower() for t in os.environ.get(
        "XPLANE_EXCLUDE_TITLES",
        "Voice Copilot,Chrome,Edge,Firefox,Visual Studio Code,PowerShell,Command Prompt,Terminal"
    ).split(",") if t.strip()
]
XPLANE_EXACT_TITLES = [
    t.strip() for t in os.environ.get(
        "XPLANE_EXACT_TITLES",
        "X-System,X-Plane,X-Plane 12"
    ).split(",") if t.strip()
]

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("xplane.telemetry")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

LOGGER = _build_logger()

WANTED = {
    # position / attitude
    "lat_deg": "sim/flightmodel/position/latitude",
    "lon_deg": "sim/flightmodel/position/longitude",
    "alt_ft": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
    "hdg_deg": "sim/flightmodel/position/psi",
    # fuel & performance
    "fuel_qty_kg": "sim/cockpit2/fuel/fuel_quantity",
    "fuel_flow_kg_s": "sim/cockpit2/engine/indicators/fuel_flow_kg_sec",
    "gs_ms": "sim/flightmodel/position/groundspeed",
    # radios / avionics (G1000-friendly)
    "com1_hz": "sim/cockpit2/radios/actuators/com1_frequency_hz_833",
    "nav1_hz": "sim/cockpit2/radios/actuators/nav1_frequency_hz",
    "hsi_source": "sim/cockpit2/radios/actuators/HSI_source_select_pilot",
}

# replace your existing _pick_api_root() with this:

def _pick_api_root() -> str:
    """
    Find the X-Plane REST API by probing hosts/ports and set BASE/PORT globals.
    Tries /api/capabilities and selects v2 if present, else v1.
    Env:
      XPLANE_HOST            e.g., "127.0.0.1" (already used)
      XPLANE_HOST_CANDIDATES e.g., "127.0.0.1,localhost"
      XPLANE_PORT            e.g., "8086" (already used)
      XPLANE_PORT_CANDIDATES e.g., "8086,8111,51000,8680"
    """
    import requests  # local import to keep startup quick
    global HOST, PORT, BASE

    # Build candidate lists
    host_candidates = [HOST]
    extra_hosts = os.environ.get("XPLANE_HOST_CANDIDATES", "")
    if extra_hosts:
        host_candidates += [h.strip() for h in extra_hosts.split(",") if h.strip()]
    # add localhost variants if not already present
    for h in ("127.0.0.1", "localhost"):
        if h not in host_candidates:
            host_candidates.append(h)

    port_candidates = []
    raw_ports = os.environ.get("XPLANE_PORT_CANDIDATES", f"{PORT},8111,51000,8680")
    for piece in raw_ports.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            port_candidates.append(int(piece))
        except ValueError:
            pass

    last_err = None
    for h in host_candidates:
        for p in port_candidates:
            base = f"http://{h}:{p}"
            try:
                r = requests.get(f"{base}/api/capabilities", headers=HEADERS, timeout=2.5)
                r.raise_for_status()
                data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                versions = (data.get("api") or {}).get("versions", [])
                version = "v2" if "v2" in versions else "v1"
                # Lock in the working endpoint for the rest of the process
                HOST = h
                PORT = p
                BASE = base
                LOGGER.info("X-Plane API discovered at %s (root /api/%s)", BASE, version)
                return f"/api/{version}"
            except Exception as e:
                last_err = e
                LOGGER.debug("Probe failed at %s: %s", base, e)

    # Fall back to original host/port to keep error path consistent
    raise HTTPException(
        status_code=503,
        detail={
            "error": "Could not discover X-Plane REST API",
            "hint": (
                "Ensure your X-Plane REST/WebSocket plugin is running. "
                "Set XPLANE_HOST / XPLANE_PORT or provide XPLANE_PORT_CANDIDATES. "
                f"Last probe error: {last_err}"
            ),
        },
    )


def _fetch_ids(root: str) -> Dict[str, int]:
    url = f"{BASE}{root}/datarefs"
    params: List[tuple[str, str]] = [("fields", "id,name,value_type")]
    for name in WANTED.values():
        params.append(("filter[name]", name))
    response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json().get("data", response.json())
    by_name = {entry["name"]: entry for entry in payload if isinstance(entry, dict)}
    ids = {key: by_name[name]["id"] for key, name in WANTED.items() if name in by_name}
    missing = [name for name in WANTED.values() if name not in by_name]
    if missing:
        LOGGER.warning("Missing datarefs (continuing without them): %s", missing)
    return ids

@dataclass(slots=True)
class TelemetryState:
    values: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    def as_dict(self) -> Dict[str, Any]:
        return {"values": self.values, "timestamp": self.timestamp}
    def as_sentence(self, detail: str = "basic") -> str:
        if not self.values:
            return "No telemetry has been received yet."
        lat = self.values.get("lat_deg")
        lon = self.values.get("lon_deg")
        alt = self.values.get("alt_ft")
        hdg = self.values.get("hdg_deg")
        parts = []
        if lat is not None and lon is not None:
            lat_dir = "N" if lat >= 0 else "S"
            lon_dir = "E" if lon >= 0 else "W"
            parts.append(f"position {abs(lat):.4f}° {lat_dir}, {abs(lon):.4f}° {lon_dir}")
        if alt is not None:
            parts.append(f"altitude {alt:.0f} feet")
        if hdg is not None:
            parts.append(f"heading {hdg:.0f}°")
        if not parts:
            return "Telemetry values are currently unavailable."
        summary = "Current flight status: " + ", ".join(parts) + "."
        if detail != "full":
            return summary
        extras = []
        ft_kg = self.values.get("fuel_total_kg")
        end_hr = self.values.get("endurance_hr")
        rng_nm = self.values.get("range_nm")
        if isinstance(ft_kg, (int, float)):
            gal = ft_kg * 2.20462 / 6.0
            extras.append(f"Fuel {ft_kg:.0f} kg (~{gal:.0f} gal).")
        if isinstance(end_hr, (int, float)) and end_hr > 0:
            endurance = f"Endurance ~{end_hr:.1f} hr"
            if isinstance(rng_nm, (int, float)):
                endurance += f" (range ~{rng_nm:.0f} nm)"
            extras.append(endurance + ".")
        com1 = self.values.get("com1_mhz")
        nav1 = self.values.get("nav1_mhz")
        src = self.values.get("hsi_source")
        if com1 or nav1 or src is not None:
            src_map = {0: "NAV1", 1: "NAV2", 2: "GPS1", 3: "GPS2"}
            cdi = src_map.get(int(src), "unknown") if isinstance(src, (int, float)) else None
            radios = []
            if com1:
                radios.append(f"COM1 {com1}")
            if nav1:
                radios.append(f"NAV1 {nav1}")
            if cdi:
                radios.append(f"CDI {cdi}")
            if radios:
                extras.append("Radios: " + ", ".join(radios) + ".")
        return summary + (" " + " ".join(extras) if extras else "")

class TelemetryManager:
    def __init__(self) -> None:
        self._state = TelemetryState()
        self._task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        self._update_count = 0
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="xplane-telemetry")
    async def _run(self) -> None:
        LOGGER.info("Starting telemetry background task – probing X-Plane API at %s", BASE)
        try:
            root = _pick_api_root()
            ids = _fetch_ids(root)
            LOGGER.info("Using telemetry API root %s with datarefs %s", root, list(ids.values()))
        except Exception as exc:
            LOGGER.error("Unable to start telemetry: %s", exc, exc_info=True)
            return
        ws_url = f"ws://{HOST}:{PORT}{root}"
        while True:
            try:
                LOGGER.info("Connecting to telemetry stream at %s", ws_url)
                async with websockets.connect(ws_url) as ws:
                    LOGGER.info("Telemetry stream connected; subscribing to datarefs")
                    subscribe = {
                        "req_id": 1,
                        "type": "dataref_subscribe_values",
                        "params": {"datarefs": [{"id": i} for i in ids.values()]},
                    }
                    await ws.send(json.dumps(subscribe))
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("type") != "dataref_update_values":
                            LOGGER.debug("Ignoring telemetry frame with type %s", message.get("type"))
                            continue
                        data = message.get("data", {})
                        mapped = {key: data.get(str(identifier)) for key, identifier in ids.items()}
                        def _sum_if_list(value: Any) -> Optional[float]:
                            if isinstance(value, list):
                                total = 0.0
                                any_found = False
                                for item in value:
                                    if isinstance(item, (int, float)):
                                        total += float(item)
                                        any_found = True
                                return total if any_found else None
                            if isinstance(value, (int, float)):
                                return float(value)
                            return None
                        fuel_total_kg = _sum_if_list(mapped.get("fuel_qty_kg"))
                        flow_kg_s = _sum_if_list(mapped.get("fuel_flow_kg_s"))
                        gs_value = mapped.get("gs_ms")
                        if isinstance(gs_value, (int, float)):
                            mapped["gs_kts"] = float(gs_value) * 1.94384
                        if fuel_total_kg is not None:
                            mapped["fuel_total_kg"] = fuel_total_kg
                        if flow_kg_s and flow_kg_s > 0:
                            endurance_hr = fuel_total_kg / (flow_kg_s * 3600.0) if fuel_total_kg else None
                            if endurance_hr:
                                mapped["endurance_hr"] = endurance_hr
                                if isinstance(mapped.get("gs_kts"), (int, float)):
                                    mapped["range_nm"] = mapped["gs_kts"] * endurance_hr
                        def _mhz_str(hz: Any, places: int = 3) -> Optional[str]:
                            if isinstance(hz, (int, float)):
                                return f"{(hz or 0) / 1_000_000:.{places}f}"
                            return None
                        com1_hz = mapped.get("com1_hz")
                        nav1_hz = mapped.get("nav1_hz")
                        com1_str = _mhz_str(com1_hz, 3)
                        nav1_str = _mhz_str(nav1_hz, 2)
                        if com1_str is not None:
                            mapped["com1_mhz"] = com1_str
                        if nav1_str is not None:
                            mapped["nav1_mhz"] = nav1_str
                        async with self._lock:
                            self._state = TelemetryState(values=mapped, timestamp=time.time())
                            self._update_count += 1
                            if self._update_count == 1 or self._update_count % 50 == 0:
                                LOGGER.info("Telemetry update %s received: %s", self._update_count, mapped)
            except Exception as exc:
                LOGGER.warning("Telemetry stream error: %s", exc, exc_info=True)
                await asyncio.sleep(2)
    async def latest(self) -> TelemetryState:
        async with self._lock:
            if not self._state.values:
                LOGGER.debug("Telemetry requested but no updates have been received yet")
            return TelemetryState(values=dict(self._state.values), timestamp=self._state.timestamp)

telemetry = TelemetryManager()

app = FastAPI(title="X-Plane Voice Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup() -> None:
    await telemetry.start()

@app.get("/status")
async def get_status(detail: str = Query("basic", pattern="^(basic|full)$")) -> JSONResponse:
    state = await telemetry.latest()
    return JSONResponse({"status": state.as_sentence(detail=detail), "raw": state.as_dict()})

# ---------- SERVER-SIDE SCREENSHOT HELPERS & ENDPOINT ----------

def _require_screenshot_libs():
    if not (Image and mss and gw):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Screenshot dependencies not installed",
                "hint": "pip install mss Pillow pygetwindow pywin32",
            },
        )

# ---- replace your existing grab_xplane_png_base64() with this ----
# put this helper above grab_xplane_png_base64()
def _pick_xplane_window():
    """
    Return the best pygetwindow window object for X-Plane:
    1) exact-title match (XPLANE_EXACT_TITLES)
    2) contains-title match (XPLANE_WINDOW_TITLES) but NOT in XPLANE_EXCLUDE_TITLES
    3) prefer visible, not minimized, largest area
    """
    _require_screenshot_libs()

    # 1) exact title match first
    try:
        all_windows = gw.getAllWindows()  # type: ignore[attr-defined]
    except Exception:
        all_windows = []

    def _is_visible(win):
        try:
            return bool(getattr(win, "isVisible", True))
        except Exception:
            return True

    def _is_minimized(win):
        try:
            return bool(getattr(win, "isMinimized", False))
        except Exception:
            return False

    def _area(win):
        try:
            w = max(0, int(win.right) - int(win.left))
            h = max(0, int(win.bottom) - int(win.top))
            return w * h
        except Exception:
            return 0

    exact = []
    for win in all_windows:
        try:
            title = (win.title or "").strip()
        except Exception:
            title = ""
        if not title:
            continue
        if title in XPLANE_EXACT_TITLES:
            exact.append(win)

    candidates = exact[:]

    # 2) else substring match, but exclude browser / our own UI titles
    if not candidates:
        low_exclude = tuple(XPLANE_EXCLUDE_TITLES)
        for win in all_windows:
            try:
                title = (win.title or "")
                tlow = title.lower()
            except Exception:
                continue
            if not title:
                continue
            # exclude obvious non-sim windows
            if any(excl in tlow for excl in low_exclude):
                continue
            if any(k.lower() in tlow for k in XPLANE_WINDOW_TITLES):
                candidates.append(win)

    # 3) score and pick largest visible
    if not candidates:
        return None

    scored = []
    for win in candidates:
        vis = _is_visible(win) and not _is_minimized(win)
        area = _area(win)
        score = (2_000_000 if vis else 0) + area  # visibility dominates, then size
        scored.append((score, win))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1] if scored else None


def grab_xplane_png_base64() -> str:
    """
    Capture X-Plane window and return a base64 JPEG that is small enough
    to send over the Realtime data channel without closing it.
    Size is controlled by envs: SCREENSHOT_MAX_WIDTH, SCREENSHOT_MAX_HEIGHT,
    SCREENSHOT_JPEG_QUALITY, SCREENSHOT_TARGET_BASE64_KB.
    """
    _require_screenshot_libs()

    # config
    max_w = int(os.environ.get("SCREENSHOT_MAX_WIDTH", "960") or "960")
    max_h = os.environ.get("SCREENSHOT_MAX_HEIGHT")
    max_h = int(max_h) if max_h and max_h.isdigit() else None
    base_quality = int(os.environ.get("SCREENSHOT_JPEG_QUALITY", "85") or "85")
    # budget for payload size (base64 bytes), keep well under typical DC limits
    target_kb = int(os.environ.get("SCREENSHOT_TARGET_BASE64_KB", "180") or "180")

    # find window
    candidates = []
    for title in XPLANE_WINDOW_TITLES:
        try:
            candidates = gw.getWindowsWithTitle(title)  # type: ignore[attr-defined]
        except Exception:
            candidates = []
        if candidates:
            break
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "X-Plane window not found",
                "hint": f"Ensure X-Plane is running (titles tried: {XPLANE_WINDOW_TITLES}). Use windowed/borderless.",
            },
        )
    win = _pick_xplane_window()
    if not win:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "X-Plane window not found",
                "hint": (
                    f"Tried exact={XPLANE_EXACT_TITLES} and contains={XPLANE_WINDOW_TITLES}, "
                    f"excluding={XPLANE_EXCLUDE_TITLES}. Ensure the sim is windowed/borderless."
                ),
            },
        )
    # ensure bounds
    if hasattr(win, "isMinimized") and win.isMinimized:
        win.restore()
        time.sleep(0.15)

    left, top, right, bottom = int(win.left), int(win.top), int(win.right), int(win.bottom)
    width = max(1, right - left)
    height = max(1, bottom - top)
    bbox = {"left": left, "top": top, "width": width, "height": height}

    # capture
    with mss.mss() as sct:  # type: ignore[call-arg]
        shot = sct.grab(bbox)
    img = Image.frombytes("RGB", shot.size, shot.rgb)  # type: ignore[arg-type]

    # downscale (preserve aspect)
    if max_w or max_h:
        # compute target box
        if max_h is None:
            # cap width only, let height follow
            ratio = max_w / float(img.width) if img.width else 1.0
            tgt = (max_w, max(1, int(img.height * ratio)))
        else:
            tgt = (max_w, max_h)
        img = img.copy()
        img.thumbnail(tgt, Image.LANCZOS)

    # try encoding and shrink until under budget
    quality = base_quality
    last_b64 = None
    for q in list(range(quality, 49, -5)) + [50, 45, 40]:
        buf = io.BytesIO()
        try:
            img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        except OSError:
            # some Pillow builds may not support progressive for certain sizes
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode("ascii")
        last_b64 = b64
        kb = len(b64)
        if kb <= target_kb * 1024:
            LOGGER.info("Screenshot encoded: %dx%d, q=%d, %d KB base64", img.width, img.height, q, kb // 1024)
            return b64

    # fallback: return the smallest we got
    kb = len(last_b64) if last_b64 else 0
    LOGGER.warning("Screenshot still large after shrinking (about %d KB base64); sending anyway.", kb // 1024)
    return last_b64 or ""


@app.post("/screenshot")
async def screenshot() -> JSONResponse:
    """
    Returns: { "image_b64": "<base64>", "mime_type": "image/jpeg" }
    """
    b64 = grab_xplane_png_base64()
    return JSONResponse({"image_b64": b64, "mime_type": "image/jpeg"})

# -------------------- OPENAI helpers & routes --------------------

def _get_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OPENAI_API_KEY is not configured",
                "hint": "Set the OPENAI_API_KEY environment variable before starting the server.",
            },
        )
    return api_key

def _perform_web_search(query: str) -> Dict[str, Any]:
    api_key = _get_api_key()
    LOGGER.info("Starting web search for query %r using model %s", query, WEB_SEARCH_MODEL)
    payload = {
        "model": WEB_SEARCH_MODEL,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": query}]}],
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"],
    }
    timeout = (WEB_SEARCH_CONNECT_TIMEOUT, WEB_SEARCH_READ_TIMEOUT)
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.HTTPError:
        try:
            detail = response.json()
        except Exception:
            detail = {"error": response.text}
        LOGGER.error("Web search HTTP error for %r: %s", query, detail, exc_info=True)
        raise HTTPException(status_code=response.status_code, detail=detail)
    except requests.Timeout as exc:
        LOGGER.error("Web search timed out for %r", query, exc_info=True)
        hint = "Try again or increase OPENAI_WEB_SEARCH_READ_TIMEOUT."
        raise HTTPException(status_code=504, detail={"error": "Web search timed out", "hint": hint}) from exc
    except requests.RequestException as exc:
        LOGGER.error("Web search request exception: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail={"error": "Failed to contact OpenAI", "hint": str(exc)}) from exc

    data = response.json()
    text = data.get("output_text")
    citations: List[Dict[str, str]] = []
    sources: List[Dict[str, str]] = []
    outputs = []
    for item in data.get("output", []):
        if item.get("type") == "web_search_call":
            action = item.get("action", {})
            for source in action.get("sources", []) or []:
                if isinstance(source, dict) and source.get("url"):
                    sources.append({"url": source.get("url"), "title": source.get("title")})
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") != "output_text":
                continue
            text_fragment = content.get("text")
            if text_fragment:
                outputs.append(text_fragment)
            for annotation in content.get("annotations", []):
                if annotation.get("type") == "url_citation" and annotation.get("url"):
                    citations.append({"url": annotation.get("url"), "title": annotation.get("title")})
    if not text:
        text = "\n".join(outputs)
    if not text:
        raise HTTPException(status_code=502, detail={"error": "Web search response did not contain any text."})
    return {"text": text, "citations": citations, "sources": sources}

def _mint_realtime_session() -> Dict[str, Any]:
    api_key = _get_api_key()
    session_cfg = {
        "model": REALTIME_MODEL,
        "voice": "alloy",
        "instructions": (
            "You are a helpful copilot. When the user asks about the flight, "
            "call the `get_flight_status` tool to retrieve the most recent telemetry "
            "and summarise it succinctly before replying."
        ),
        "tools": [
            {
                "type": "function",
                "name": "get_flight_status",
                "description": "Fetches the latest flight telemetry snapshot from X-Plane.",
                "parameters": {
                    "type": "object",
                    "properties": {"detail": {"type": "string", "enum": ["basic", "full"]}},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "search_web",
                "description": "Performs a web search for up-to-date aviation information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 3}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        ],
    }
    try:
        r = requests.post(
            "https://api.openai.com/v1/realtime/client_secrets",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"session": session_cfg},
            timeout=10,
        )
        r.raise_for_status()
        payload = r.json()
        return {"client_secret": payload.get("client_secret"), "model": REALTIME_MODEL}
    except requests.HTTPError:
        try:
            r = requests.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=session_cfg,
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            try:
                detail = r.json()
            except Exception:
                detail = {"error": r.text}
            raise HTTPException(status_code=r.status_code, detail=detail)
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail={"error": "Failed to contact OpenAI", "hint": str(exc)}) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail={"error": "Failed to contact OpenAI", "hint": str(exc)}) from exc

@app.post("/session")
async def create_session() -> JSONResponse:
    payload = _mint_realtime_session()
    return JSONResponse({"session": payload})

@app.get("/session", response_class=HTMLResponse)
async def session_page() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)

@app.get("/search")
async def search(query: str = Query(..., min_length=3, max_length=500)) -> JSONResponse:
    result = _perform_web_search(query)
    return JSONResponse({"result": result})

# ---------------------------- UI ----------------------------

HTML_PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>X-Plane Voice Copilot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
      body { margin: 0; padding: 1.5rem; display: grid; gap: 1.5rem; max-width: 1000px; }
      h1 { margin: 0; }
      button, label { font: inherit; }
      button { padding: 0.6rem 1rem; border-radius: 0.75rem; border: none; cursor: pointer; }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      .row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }
      #log { border: 1px solid rgba(128,128,128,0.4); border-radius: 0.75rem; padding: 1rem; height: 18rem; overflow-y: auto; white-space: pre-wrap; background: rgba(0,0,0,0.05); display: grid; gap: 0.75rem; }
      .log-entry-heading { font-weight: 600; }
      .search-result { white-space: normal; }
      .search-citations { margin: 0.5rem 0 0; padding-left: 1.5rem; }
      .search-citations-heading { margin-top: 0.75rem; font-weight: 600; }
      audio { width: 100%; }
      .status { font-size: 0.95rem; color: #666; }
      .muted { opacity: 0.7; }
    </style>
  </head>
  <body>
    <header>
      <h1>X-Plane Voice Copilot</h1>
      <p class="status">Press "Connect" and start speaking. Ask "How's the flight doing?" to hear telemetry readbacks with the outside view as context.</p>
    </header>

    <section class="row">
      <button id="connect">Connect</button>
      <audio id="assistant-audio" autoplay></audio>
    </section>

    <section>
      <h2>Visual context (server-side)</h2>
      <div class="row">
        <button id="snap">Snap &amp; Send</button>
        <label class="row muted">
          <input type="checkbox" id="autoSend" />
          Auto-send every <input type="number" id="intervalSec" min="2" value="5" style="width:3.5rem" />s
        </label>
        <label class="row muted">
          <input type="checkbox" id="contextOnly" checked />
          Context only (don’t reply on snap/auto)
        </label>
      </div>
      <div class="row muted">
        <small>Tip: Run X-Plane in windowed or borderless mode. Server grabs the "X-System"/"X-Plane" window and attaches it to the current turn.</small>
      </div>
    </section>

    <section>
      <h2>Conversation log</h2>
      <div id="log"></div>
    </section>

    <script>
      const logEl = document.getElementById('log');
      const connectButton = document.getElementById('connect');
      const audioEl = document.getElementById('assistant-audio');

      const snapBtn = document.getElementById('snap');
      const autoSendChk = document.getElementById('autoSend');
      const intervalSec = document.getElementById('intervalSec');
      const contextOnlyChk = document.getElementById('contextOnly');

      let pc, dc;
      const pendingCalls = {};
      let autoTimer = null;
      // Track the latest screenshot item id so we can replace it on next snap
      let lastScreenshotItemId = null;

      const coalesce = (v, f) => (v === undefined || v === null) ? f : v;
      const getNested = (obj, keys) => keys.reduce((o, k) => (o == null ? undefined : o[k]), obj);
      const logTimestamp = () => new Date().toLocaleTimeString();
      const appendLogNode = (node) => { logEl.appendChild(node); logEl.scrollTop = logEl.scrollHeight; };

      function log(msg) {
        const d = document.createElement('div');
        d.textContent = "[" + logTimestamp() + "] " + msg;
        appendLogNode(d);
      }
      function logError(message, error) {
        console.error(message, error);
        const em = getNested(error, ['error','message']);
        const detail = em ? (" (" + em + ")") : (typeof error === 'object' ? (" " + JSON.stringify(error)) : "");
        log(message + detail);
      }

      function sendEvent(payload) {
        if (!dc || dc.readyState !== 'open') {
          const state = dc && dc.readyState != null ? dc.readyState : 'absent';
          throw new Error("Data channel is not open (state: " + state + ")");
        }
        dc.send(JSON.stringify(payload));
        log("→ Sent event: " + payload.type);
      }

      async function fetchServerScreenshotB64() {
        const r = await fetch('/screenshot', { method: 'POST' });
        if (!r.ok) {
          let hint = '';
          try { const e = await r.json(); hint = e && e.detail ? (' ' + JSON.stringify(e.detail)) : ''; } catch (e2) {}
          throw new Error('Screenshot failed: HTTP ' + r.status + hint);
        }
        const data = await r.json();
        const b64 = getNested(data, ['image_b64']);
        if (!b64) throw new Error('No image_b64 in response.');
        return b64;
      }

      // ---------- SNAP BEHAVIORS ----------

      async function sendScreenshotTurn() {
        // Guard: if toggle is ON, treat as context-only even if this is called
        if (contextOnlyChk && contextOnlyChk.checked) {
          return sendScreenshotContextOnly();
        }

        try {
          const b64 = await fetchServerScreenshotB64();

          // Remove prior screenshot so only the newest image is in context
          if (lastScreenshotItemId) {
            sendEvent({ type: 'conversation.item.delete', item_id: lastScreenshotItemId });
            lastScreenshotItemId = null;
          }

          const note = [
            'NEW SNAPSHOT ' + new Date().toISOString() + '.',
            'Use THIS image ONLY for visual cues: outside scenery/runway/taxiway, horizon & pitch/attitude,',
            'clouds/visibility/weather, terrain/obstacles, lights/traffic, and any obvious warnings or signage.',
            'Do NOT estimate numeric flight parameters from pixels.',
            'For numbers (altitude, groundspeed, heading, fuel/endurance, radios), call the get_flight_status tool with {"detail":"full"}',
            'and treat its values as authoritative. Fuse your visual observations with telemetry into one concise answer.',
            'Ignore any prior images.'
          ].join(' ');

          sendEvent({
            type: 'conversation.item.create',
            item: {
              type: 'message',
              role: 'user',
              content: [
                { type: 'input_text', text: note },
                { type: 'input_image', image_url: 'data:image/jpeg;base64,' + b64 }
              ]
            }
          });

          // Give the server a beat to commit the new item, then request a reply
          setTimeout(() => {
            try { sendEvent({ type: 'response.create' }); }
            catch (e) { logError('response.create failed', e); }
          }, 150);

          log('Screenshot attached and response requested.');
        } catch (err) {
          logError('Failed to snap & send', err);
        }
      }

      async function sendScreenshotContextOnly() {
        try {
          const b64 = await fetchServerScreenshotB64();

          if (lastScreenshotItemId) {
            sendEvent({ type: 'conversation.item.delete', item_id: lastScreenshotItemId });
            lastScreenshotItemId = null;
          }

          const note =
            'Context image only. Do not reply or describe this image. ' +
            'Use it to ground the next answer if relevant (scenery/runway/horizon/weather/terrain/traffic). ' +
            'For numeric data, call get_flight_status (detail="full").';

          sendEvent({
            type: 'conversation.item.create',
            item: {
              type: 'message',
              role: 'user',
              content: [
                { type: 'input_text', text: note },
                { type: 'input_image', image_url: 'data:image/jpeg;base64,' + b64 }
              ]
            }
          });

          // No response.create here — purely staging context
          log('Context screenshot attached (no immediate reply).');
        } catch (err) {
          logError('Failed to snap (context only)', err);
        }
      }

      // ---------- UI WIRING ----------

      // Snap button respects the Context only toggle
      snapBtn.addEventListener('click', () => {
        if (contextOnlyChk.checked) {
          sendScreenshotContextOnly();
        } else {
          sendScreenshotTurn();
        }
      });

      // Auto-send also respects the Context only toggle
      function toggleAutoSend(on) {
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
        if (!on) return;
        const secs = Math.max(2, Number(intervalSec.value) || 5);
        autoTimer = setInterval(() => {
          if (contextOnlyChk.checked) {
            sendScreenshotContextOnly();   // attach image silently
          } else {
            sendScreenshotTurn();          // attach image and request a reply
          }
        }, secs * 1000);
        log("Auto-send enabled (every " + secs + "s).");
      }
      autoSendChk.addEventListener('change', (e) => toggleAutoSend(e.target.checked));

      // ---------- SEARCH HELPERS ----------

      function logSearchResult(result, query) {
        const entry = document.createElement('div');
        entry.classList.add('log-entry', 'search-result');
        const heading = document.createElement('div');
        heading.classList.add('log-entry-heading');
        heading.textContent = "[" + logTimestamp() + "] search_web result for \"" + query + "\"";
        entry.appendChild(heading);
        const summary = document.createElement('p');
        const summaryText = getNested(result, ['text']);
        summary.textContent = coalesce(summaryText, 'No summary text was returned.');
        entry.appendChild(summary);
        const citations = Array.isArray(result.citations) ? result.citations : [];
        if (citations.length) {
          const list = document.createElement('ol');
          list.classList.add('search-citations');
          const seen = new Set();
          citations.forEach((c) => {
            if (!c || !c.url || seen.has(c.url)) return;
            seen.add(c.url);
            const item = document.createElement('li');
            const link = document.createElement('a');
            link.href = c.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
            link.textContent = c.title || c.url;
            item.appendChild(link); list.appendChild(item);
          });
          if (list.childElementCount) {
            const h = document.createElement('div');
            h.classList.add('search-citations-heading');
            h.textContent = 'Sources';
            entry.appendChild(h); entry.appendChild(list);
          }
        }
        appendLogNode(entry);
      }

      function formatSearchResultForAssistant(result) {
        if (!result || !result.text) return 'I could not retrieve any search results.';
        const citations = Array.isArray(result.citations) ? result.citations : [];
        const seen = new Set(); const unique = [];
        citations.forEach((c) => { if (!c || !c.url || seen.has(c.url)) return; seen.add(c.url); unique.push({ title: c.title || c.url, url: c.url }); });
        if (!unique.length) return result.text;
        const lines = unique.map((c, i) => "[" + (i+1) + "] " + c.title + " - " + c.url);
        return result.text + "\\n\\nSources:\\n" + lines.join("\\n");
      }

      async function fetchTelemetry(detail) {
        detail = detail || "basic";
        log('Fetching latest telemetry...');
        const r = await fetch('/status?detail=' + encodeURIComponent(detail));
        if (!r.ok) throw new Error('Telemetry request failed: ' + r.status);
        const data = await r.json();
        if (!data || !data.status) throw new Error('Telemetry payload missing status string.');
        return data.status;
      }

      async function performWebSearch(query) {
        log('Searching the web for: ' + query);
        const r = await fetch('/search?query=' + encodeURIComponent(query));
        if (!r.ok) throw new Error('Web search failed: ' + r.status);
        const data = await r.json();
        if (!data || !data.result) throw new Error('Web search response missing result.');
        logSearchResult(data.result, query);
        return data.result;
      }

      const resolveToolCallId = (call) => call ? (call.id ?? call.tool_call_id ?? call.call_id ?? call.response_id ?? null) : null;
      const formatToolOutput = (toolCallId, text) => ({ tool_call_id: toolCallId, output: text });

      async function submitToolOutputs(responseId, calls) {
        const outputs = [];
        for (const call of calls) {
          const toolCallId = resolveToolCallId(call);
          if (!toolCallId) { log('Tool call x Missing tool_call_id.'); continue; }
          const fn = call.function;
          const functionName = (fn && fn.name != null) ? fn.name : call.name;
          if (functionName === 'get_flight_status') {
            try {
              let detail = 'full';
              let argSource = fn && fn.arguments != null ? fn.arguments : call.arguments;
              if (typeof argSource === 'string' && argSource.trim()) {
                try { const parsed = JSON.parse(argSource); if (parsed && typeof parsed.detail === 'string') detail = parsed.detail; } catch (e) {}
              }
              const status = await fetchTelemetry(detail);
              outputs.push(formatToolOutput(toolCallId, status));
              log('Tool call ok get_flight_status -> ' + status);
            } catch (error) {
              logError('Tool call x get_flight_status failed', error);
              outputs.push(formatToolOutput(toolCallId, 'Telemetry is currently unavailable. Please try again shortly.'));
            }
          } else if (functionName === 'search_web') {
            try {
              let query = '';
              let argSource = fn && fn.arguments != null ? fn.arguments : call.arguments;
              if (typeof argSource === 'string' && argSource.trim()) {
                try { const parsed = JSON.parse(argSource); if (parsed && typeof parsed.query === 'string') query = parsed.query; } catch (e) {}
              }
              if (!query) throw new Error('Missing web search query.');
              const result = await performWebSearch(query);
              outputs.push(formatToolOutput(toolCallId, formatSearchResultForAssistant(result)));
              log('Tool call ok search_web completed.');
            } catch (error) {
              logError('Tool call x search_web failed', error);
              outputs.push(formatToolOutput(toolCallId, 'I could not complete the web search. Please try again or adjust your request.'));
            }
          } else {
            const safeName = functionName || 'unknown';
            log('Tool call x Unsupported tool "' + safeName + '"');
            outputs.push(formatToolOutput(toolCallId, 'Unsupported tool: ' + safeName));
          }
        }
        if (!outputs.length) { log('No tool outputs to submit.'); return; }
        try {
          sendEvent({ type: 'response.submit_tool_outputs', response_id: responseId, tool_outputs: outputs });
        } catch (error) { logError('Failed to send tool outputs', error); }
      }

      // ---------- CONNECT / REALTIME ----------

      connectButton.addEventListener('click', connect);
      async function connect() {
        connectButton.disabled = true;
        try {
          log('Requesting session token...');
          const sessionResponse = await fetch('/session', { method: 'POST' });
          if (!sessionResponse.ok) { log('Failed to create session.'); return; }
          const sessionPayload = await sessionResponse.json();
          const session = sessionPayload && sessionPayload.session ? sessionPayload.session : undefined;
          let token;
          if (session && typeof session === 'object') {
            const secret = session.client_secret;
            token = (secret && typeof secret === 'object' && 'value' in secret) ? (secret.value || secret) : secret;
          }
          if (!token) { log('Session response was missing a client token.'); return; }

          log('Creating WebRTC peer connection...');
          pc = new RTCPeerConnection();
          pc.addEventListener('connectionstatechange', () => log('Peer connection state -> ' + pc.connectionState));
          pc.addEventListener('iceconnectionstatechange', () => log('ICE connection state -> ' + pc.iceConnectionState));
          pc.addEventListener('signalingstatechange', () => log('Signaling state -> ' + pc.signalingState));
          pc.addEventListener('icegatheringstatechange', () => log('ICE gathering state -> ' + pc.iceGatheringState));

          dc = pc.createDataChannel('oai-events');
          dc.addEventListener('open', () => {
            log('Data channel opened.');
            // Session-wide guard: images are context; use tool for numbers; no auto image descriptions.
            try {
              sendEvent({
                type: 'session.update',
                session: {
                  instructions:
                    'When an input_image arrives, treat it as visual context (scenery/runway/horizon/pitch/weather/traffic). ' +
                    'Do not describe an image unless the user asks. For numeric flight parameters, always call the get_flight_status tool (detail="full") and rely on it. ' +
                    'Prefer the most recent image; ignore earlier images unless explicitly asked to compare.'
                }
              });
            } catch (e) {
              logError('Failed to update session instructions', e);
            }
          });

          dc.addEventListener('close', () => log('Data channel closed.'));
          dc.addEventListener('error', (event) => logError('Data channel error', event));
          dc.addEventListener('message', async (event) => {
            try {
              const msg = JSON.parse(event.data);
              log('Received event: ' + msg.type);

              // Store id of any user message that contains an input_image (new screenshot)
              if (msg.type === 'conversation.item.created') {
                const it = msg.item || msg.conversation_item || {};
                const content = Array.isArray(it.content) ? it.content : [];
                const hasImage = content.some(p => p && p.type === 'input_image');
                if (it.role === 'user' && hasImage) {
                  lastScreenshotItemId = it.id || msg.item_id || null;
                  log('Stored last screenshot item id: ' + lastScreenshotItemId);
                }
              }

              if (
                msg.type === 'response.output_text.delta' ||
                msg.type === 'response.text.delta' ||
                msg.type === 'response.audio_transcript.delta'
              ) {
                if (msg.delta) {
                  const speaker = msg.type === 'response.audio_transcript.delta' ? 'Transcript' : 'Assistant';
                  log(speaker + ': ' + msg.delta);
                }
              } else if (msg.type === 'response.function_call_arguments.delta') {
                const id = (msg.call_id != null ? msg.call_id :
                           (msg.tool_call_id != null ? msg.tool_call_id :
                           (msg.id != null ? msg.id : null)));
                if (!id) { log('Function call delta missing call_id; ignoring.'); return; }
                if (!pendingCalls[id]) {
                  pendingCalls[id] = { name: (msg.name != null ? msg.name : null), args: '', item_id: (msg.item_id != null ? msg.item_id : null) };
                }
                const pending = pendingCalls[id];
                if (msg.name && !pending.name) pending.name = msg.name;
                if (typeof msg.delta === 'string') pending.args += msg.delta;
                if (msg.item_id && !pending.item_id) pending.item_id = msg.item_id;
              } else if (msg.type === 'response.function_call_arguments.done') {
                const callId = (msg.call_id != null ? msg.call_id :
                               (msg.tool_call_id != null ? msg.tool_call_id :
                               (msg.id != null ? msg.id : null)));
                if (!callId) { log('Function call completion missing call_id; cannot respond.'); return; }
                const record = pendingCalls[callId] || { name: (msg.name != null ? msg.name : null), args: '', item_id: (msg.item_id != null ? msg.item_id : null) };
                delete pendingCalls[callId];

                const functionName = (record.name != null ? record.name : (msg.name != null ? msg.name : null));
                let outputText = 'Telemetry is currently unavailable. Please try again shortly.';
                const parseArgs = () => { if (!record.args) return {}; try { return JSON.parse(record.args); } catch (e) { return {}; } };

                try {
                  if (functionName === 'get_flight_status') {
                    const args = parseArgs(); const detail = (args && typeof args.detail === 'string') ? args.detail : 'full';
                    outputText = await fetchTelemetry(detail);
                    log('Function call ok get_flight_status -> ' + outputText);
                  } else if (functionName === 'search_web') {
                    const args = parseArgs(); const query = (args && typeof args.query === 'string') ? args.query : '';
                    if (!query) throw new Error('Missing web search query.');
                    const result = await performWebSearch(query);
                    outputText = formatSearchResultForAssistant(result);
                    log('Function call ok search_web completed.');
                  } else {
                    const safeName = functionName || 'unknown';
                    log('Function call x Unsupported tool "' + safeName + '"');
                    outputText = 'Unsupported tool: ' + safeName;
                  }
                } catch (error) {
                  if (functionName === 'search_web') {
                    outputText = 'I could not complete the web search. Please try again or adjust your request.';
                    logError('Function call x search_web failed', error);
                  } else if (functionName === 'get_flight_status') {
                    outputText = 'Telemetry is currently unavailable. Please try again shortly.';
                    logError('Function call x get_flight_status failed', error);
                  } else {
                    outputText = 'Tool execution failed. Please try again shortly.';
                    logError('Function call x tool execution failed', error);
                  }
                }

                try {
                  const item = { type: 'function_call_output', call_id: callId, output: outputText };
                  const message = { type: 'conversation.item.create', item: item };
                  if (record.item_id != null) message.previous_item_id = record.item_id;
                  sendEvent(message);
                  sendEvent({ type: 'response.create', response: {} });
                } catch (error) {
                  logError('Failed to send function result', error);
                }
              } else if (msg.type === 'response.done' || (msg.type === 'response.completed' && msg.response && msg.response.status === 'completed')) {
                log('Assistant response completed.');
              } else if (msg.type === 'response.required_action' && msg.response && msg.response.required_action && msg.response.required_action.type === 'submit_tool_outputs') {
                const ra = msg.response.required_action;
                let calls = [];
                if (ra.submit_tool_outputs && Array.isArray(ra.submit_tool_outputs.tool_calls)) calls = ra.submit_tool_outputs.tool_calls;
                else if (Array.isArray(ra.tool_calls)) calls = ra.tool_calls;
                const responseId = (msg.response && msg.response.id != null) ? msg.response.id : (msg.response_id != null ? msg.response_id : null);
                await submitToolOutputs(responseId, calls);
              } else if (msg.type === 'response.error' || msg.type === 'error') {
                logError('Assistant reported an error', msg);
              }
            } catch (err) {
              logError('Failed to process message', err);
              console.error('Raw message data:', event.data);
            }
          });

          pc.addEventListener('track', (event) => {
            const stream = event.streams[0];
            audioEl.srcObject = stream;
          });

          log('Requesting microphone access...');
          const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

          log('Creating SDP offer...');
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);

          log('Contacting OpenAI Realtime API...');
          const sdpResponse = await fetch('https://api.openai.com/v1/realtime?model=' + encodeURIComponent(session.model), {
            method: 'POST',
            body: offer.sdp,
            headers: { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/sdp' },
          });
          if (!sdpResponse.ok) { log('Failed to negotiate WebRTC session (status ' + sdpResponse.status + ').'); return; }
          const answer = { type: 'answer', sdp: await sdpResponse.text() };
          await pc.setRemoteDescription(answer);
          log('Connected - start talking!');
        } catch (error) {
          logError('Connection failed', error);
          if (pc) { try { pc.close(); } catch (e) {} }
        } finally {
          if (!pc || pc.connectionState === 'failed' || pc.connectionState === 'closed') connectButton.disabled = false;
        }
      }
    </script>
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)

__all__ = ["app", "telemetry", "TelemetryManager", "TelemetryState"]
