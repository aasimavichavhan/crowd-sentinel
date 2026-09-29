import cv2
import base64
import time
import asyncio
import os
import numpy as np
from typing import Dict, List, Optional, Set
from fastapi import WebSocket

from config import SystemConfig
from detector import PersonDetector
from optical_flow import OpticalFlowTurbulence
from risk_engine import RiskEngine
from visualizer import ZoneVisualizer

class VideoStreamManager:
    def __init__(self, config_obj: SystemConfig):
        self.config = config_obj
        self.detector = PersonDetector(
            model_name=self.config.get("detection.model_name", "yolov8n.pt"),
            conf_thresh=float(self.config.get("detection.confidence_threshold", 0.25))
        )
        self.flow_engine = OpticalFlowTurbulence(
            pyr_scale=float(self.config.get("optical_flow.pyr_scale", 0.5)),
            levels=int(self.config.get("optical_flow.levels", 3)),
            winsize=int(self.config.get("optical_flow.winsize", 15)),
            iterations=int(self.config.get("optical_flow.iterations", 3)),
            poly_n=int(self.config.get("optical_flow.poly_n", 5)),
            poly_sigma=float(self.config.get("optical_flow.poly_sigma", 1.2)),
            turbulence_gain=float(self.config.get("optical_flow.turbulence_gain", 12.0)),
            min_motion_threshold=float(self.config.get("optical_flow.min_motion_threshold", 0.3))
        )
        self.risk_engine = RiskEngine(self.config)
        self.visualizer = ZoneVisualizer()

        self.current_video_path: Optional[str] = None
        self.is_playing: bool = False
        self.is_looping: bool = True
        self.clients: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.worker_task: Optional[asyncio.Task] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_index = 0
        self.total_frames = 0
        self.fps_measured = 0.0

    async def connect_client(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        # Send current status
        status = {
            "type": "status",
            "is_playing": self.is_playing,
            "current_video": os.path.basename(self.current_video_path) if self.current_video_path else None
        }
        await websocket.send_json(status)

    def disconnect_client(self, websocket: WebSocket):
        self.clients.discard(websocket)

    async def broadcast(self, data: dict):
        if not self.clients:
            return
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send_json(data)
            except Exception:
                dead_clients.add(client)
        for dead in dead_clients:
            self.clients.discard(dead)

    def set_video(self, video_path: str):
        self.current_video_path = video_path
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.flow_engine.reset()
        self.risk_engine.reset()
        self.frame_index = 0

        if os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.cap = None
            self.total_frames = 0

    def start(self):
        if self.worker_task is None or self.worker_task.done():
            self.is_playing = True
            self.worker_task = asyncio.create_task(self._process_stream())
        else:
            self.is_playing = True

    def pause(self):
        self.is_playing = False

    def restart(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = 0
        self.flow_engine.reset()
        self.risk_engine.reset()
        self.is_playing = True

    async def _process_stream(self):
        rows = int(self.config.get("grid.rows", 3))
        cols = int(self.config.get("grid.cols", 3))
        target_width = int(self.config.get("detection.target_width", 640))
        target_fps = float(self.config.get("detection.target_fps", 20.0))
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0.05

        last_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        while True:
            try:
                if not self.is_playing or self.cap is None or not self.cap.isOpened():
                    await asyncio.sleep(0.1)
                    continue

                loop_start = time.time()

                # Read next frame
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_looping:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.flow_engine.reset()
                        self.frame_index = 0
                        ret, frame = self.cap.read()
                        if not ret:
                            await asyncio.sleep(0.2)
                            continue
                    else:
                        self.is_playing = False
                        await self.broadcast({"type": "video_ended"})
                        await asyncio.sleep(0.2)
                        continue

                self.frame_index += 1

                # Resize frame for uniform fast CPU processing
                h, w = frame.shape[:2]
                if w != target_width:
                    scale = target_width / float(w)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (target_width, new_h))

                # Step 1: YOLO Person Detection
                det_res = self.detector.detect_and_assign_zones(frame, rows=rows, cols=cols)

                # Step 2: Optical Flow & Turbulence
                flow_res = self.flow_engine.compute_flow(frame, rows=rows, cols=cols)

                # Step 3: Risk Engine & Predictive Extrapolation
                current_sec = (self.frame_index / target_fps)
                risk_eval = self.risk_engine.evaluate_zones(
                    zone_counts=det_res["zone_counts"],
                    zone_turbulence=flow_res["zone_turbulence"],
                    rows=rows,
                    cols=cols,
                    timestamp_sec=current_sec
                )

                # FPS calculation
                fps_counter += 1
                now = time.time()
                if (now - fps_timer) >= 1.0:
                    self.fps_measured = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # Step 4: Render Visual HUD
                annotated = self.visualizer.draw_overlay(
                    frame=frame,
                    zone_eval=risk_eval,
                    detections=det_res["detections"],
                    fps=self.fps_measured,
                    flow=flow_res.get("flow", None)
                )

                # Step 5: Encode to JPEG base64
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64_frame = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

                payload = {
                    "type": "frame",
                    "frame": b64_frame,
                    "frame_idx": self.frame_index,
                    "total_frames": self.total_frames,
                    "fps": round(self.fps_measured, 1),
                    "current_time_sec": round(current_sec, 1),
                    "total_people": det_res["total_count"],
                    "average_risk": risk_eval["average_risk"],
                    "highest_risk": risk_eval["overall_tier"],
                    "zones": risk_eval["zones"],
                    "new_alerts": risk_eval["new_alerts"],
                    "recent_alerts": risk_eval["recent_alerts"]
                }

                await self.broadcast(payload)

                # Maintain target framerate pacing
                elapsed = time.time() - loop_start
                sleep_time = max(0.005, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                print(f"Error in video processing loop: {e}")
                await asyncio.sleep(0.5)
