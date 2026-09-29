import os
import cv2
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import SystemConfig
from video_streamer import VideoStreamManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_videos")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize configuration and streaming manager
config = SystemConfig.get_instance(os.path.join(BASE_DIR, "config.yaml"))
stream_manager = VideoStreamManager(config)

app = FastAPI(title="Crowd Sentinel - Stampede Early-Warning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    # Auto-load the first available video if exists
    videos = get_video_files()
    if videos:
        # Prefer calm_crowd.mp4 by default or the first video
        default_video = next((v for v in videos if "calm" in v["filename"].lower()), videos[0])
        full_path = os.path.join(SAMPLE_DIR, default_video["filename"])
        stream_manager.set_video(full_path)
        stream_manager.start()

def get_video_files():
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_list = []
    if not os.path.exists(SAMPLE_DIR):
        return video_list

    for fname in sorted(os.listdir(SAMPLE_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            fpath = os.path.join(SAMPLE_DIR, fname)
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            
            # Quick probe for duration and resolution
            duration_sec = 0.0
            resolution = "Unknown"
            try:
                probe_cap = cv2.VideoCapture(fpath)
                if probe_cap.isOpened():
                    fc = probe_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps = probe_cap.get(cv2.CAP_PROP_FPS) or 20.0
                    w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration_sec = round(fc / fps, 1) if fps > 0 else 0.0
                    resolution = f"{w}x{h}"
                    probe_cap.release()
            except Exception:
                pass

            tag = "Demo Clip"
            if "calm" in fname.lower():
                tag = "Normal / Calm Flow"
            elif "dense" in fname.lower() or "risky" in fname.lower():
                tag = "High-Risk Surge / Crush"
            elif "vtest" in fname.lower():
                tag = "Pedestrian Benchmark"

            video_list.append({
                "filename": fname,
                "label": fname.replace("_", " ").replace(".mp4", "").replace(".avi", "").title(),
                "size_mb": size_mb,
                "duration_sec": duration_sec,
                "resolution": resolution,
                "tag": tag
            })
    return video_list

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Crowd Safety Early Warning System running. Static index.html not found.</h1>"

@app.get("/api/videos")
async def api_list_videos():
    return {
        "videos": get_video_files(),
        "current_video": os.path.basename(stream_manager.current_video_path) if stream_manager.current_video_path else None
    }

class VideoSelectRequest(BaseModel):
    video: str

@app.post("/api/control/play")
async def api_play(req: Optional[VideoSelectRequest] = None):
    if req and req.video:
        video_path = os.path.join(SAMPLE_DIR, req.video)
        if os.path.exists(video_path):
            stream_manager.set_video(video_path)
    stream_manager.start()
    return {"status": "playing", "current_video": os.path.basename(stream_manager.current_video_path) if stream_manager.current_video_path else None}

@app.post("/api/control/pause")
async def api_pause():
    stream_manager.pause()
    return {"status": "paused"}

@app.post("/api/control/restart")
async def api_restart():
    stream_manager.restart()
    return {"status": "restarted"}

@app.post("/api/videos/upload")
async def api_upload_video(file: UploadFile = File(...)):
    filename = file.filename
    dest_path = os.path.join(SAMPLE_DIR, filename)
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"status": "uploaded", "filename": filename}

@app.get("/api/config")
async def api_get_config():
    return config.data

class ConfigUpdateRequest(BaseModel):
    preset: Optional[str] = None
    moderate_count: Optional[int] = None
    dense_count: Optional[int] = None
    risky_count: Optional[int] = None
    turbulence_threshold: Optional[float] = None

@app.post("/api/config")
async def api_update_config(req: ConfigUpdateRequest):
    if req.preset == "sensitive":
        config.set("risk_engine.density.moderate_count", 3)
        config.set("risk_engine.density.dense_count", 6)
        config.set("risk_engine.density.risky_count", 10)
        config.set("risk_engine.turbulence.risky_threshold", 65.0)
    elif req.preset == "balanced":
        config.set("risk_engine.density.moderate_count", 4)
        config.set("risk_engine.density.dense_count", 8)
        config.set("risk_engine.density.risky_count", 14)
        config.set("risk_engine.turbulence.risky_threshold", 80.0)
    elif req.preset == "relaxed":
        config.set("risk_engine.density.moderate_count", 6)
        config.set("risk_engine.density.dense_count", 12)
        config.set("risk_engine.density.risky_count", 18)
        config.set("risk_engine.turbulence.risky_threshold", 90.0)

    if req.moderate_count is not None:
        config.set("risk_engine.density.moderate_count", req.moderate_count)
    if req.dense_count is not None:
        config.set("risk_engine.density.dense_count", req.dense_count)
    if req.risky_count is not None:
        config.set("risk_engine.density.risky_count", req.risky_count)
    if req.turbulence_threshold is not None:
        config.set("risk_engine.turbulence.risky_threshold", req.turbulence_threshold)

    return {"status": "updated", "config": config.data}

@app.get("/api/email-status")
async def api_email_status():
    """Returns the current email alerter configuration and dispatch log."""
    alerter = stream_manager.email_alerter
    return {
        "smtp_configured": alerter.is_smtp_ready,
        "recipient": alerter.recipient,
        "cooldown_seconds": alerter.cooldown,
        "recent_dispatches": alerter.email_log
    }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await stream_manager.connect_client(websocket)
    try:
        while True:
            # Handle incoming client commands over websocket if any
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                action = data.get("action")
                if action == "play":
                    video = data.get("video")
                    if video:
                        vpath = os.path.join(SAMPLE_DIR, video)
                        if os.path.exists(vpath):
                            stream_manager.set_video(vpath)
                    stream_manager.start()
                elif action == "pause":
                    stream_manager.pause()
                elif action == "restart":
                    stream_manager.restart()
            except Exception as e:
                print(f"Error handling ws message: {e}")
    except WebSocketDisconnect:
        stream_manager.disconnect_client(websocket)
    except Exception:
        stream_manager.disconnect_client(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
