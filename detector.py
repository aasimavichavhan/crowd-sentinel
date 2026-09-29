import numpy as np
from typing import List, Dict, Tuple, Any

class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.25):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.model = None

    def _ensure_model_loaded(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)

    def detect_and_assign_zones(
        self, 
        frame: np.ndarray, 
        rows: int = 3, 
        cols: int = 3
    ) -> Dict[str, Any]:
        """
        Runs YOLOv8n person detection on frame and assigns detected persons
        to zones based on their ground position (bottom-center of bounding box).
        """
        self._ensure_model_loaded()
        h, w = frame.shape[:2]
        cell_h = h / rows
        cell_w = w / cols

        # Run inference: class 0 is 'person' in COCO
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.conf_thresh,
            verbose=False,
            device='cpu'
        )

        detections = []
        zone_counts = {(r, c): 0 for r in range(rows) for c in range(cols)}
        zone_boxes = {(r, c): [] for r in range(rows) for c in range(cols)}

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                
                # Bottom-center represents ground position of person
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                # Determine zone
                r = int(foot_y / cell_h)
                c = int(foot_x / cell_w)
                r = max(0, min(rows - 1, r))
                c = max(0, min(cols - 1, c))

                detection_info = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": conf,
                    "foot": [float(foot_x), float(foot_y)],
                    "zone": (r, c)
                }
                detections.append(detection_info)
                zone_counts[(r, c)] += 1
                zone_boxes[(r, c)].append(detection_info)

        return {
            "total_count": len(detections),
            "detections": detections,
            "zone_counts": zone_counts,
            "zone_boxes": zone_boxes,
            "frame_shape": (h, w),
            "cell_dims": (cell_h, cell_w)
        }
