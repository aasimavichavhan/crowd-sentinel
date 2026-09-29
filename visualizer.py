import cv2
import numpy as np
import time
from typing import Dict, List, Any

# BGR Color Palette
COLOR_MAP = {
    "Normal": (46, 204, 113),     # Soft Emerald Green
    "Moderate": (24, 206, 241),    # Amber / Golden Yellow
    "Dense": (30, 144, 255),       # Deep Orange
    "Risky": (40, 40, 238)         # Crimson Red
}

ALPHA_MAP = {
    "Normal": 0.08,
    "Moderate": 0.16,
    "Dense": 0.24,
    "Risky": 0.38
}

class ZoneVisualizer:
    def __init__(self):
        self.pulse_phase = 0.0

    def draw_overlay(
        self,
        frame: np.ndarray,
        zone_eval: Dict[str, Any],
        detections: List[Dict[str, Any]],
        fps: float = 0.0,
        flow: np.ndarray = None,
        draw_flow_vectors: bool = False
    ) -> np.ndarray:
        """
        Draws the complete crowd safety HUD on the video frame:
        - Semi-transparent color-coded zone tiles
        - Zone boundary lines and headers
        - Density & turbulence stats per zone
        - 30-second predictive bottleneck warnings
        - Modern person bounding boxes / foot markers
        - Privacy assurance watermark & top stats HUD
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()
        overlay = frame.copy()

        zones = zone_eval.get("zones", [])
        if not zones:
            return canvas

        # Determine rows and cols from zones
        rows = max(z["row"] for z in zones) + 1
        cols = max(z["col"] for z in zones) + 1

        cell_h = h / rows
        cell_w = w / cols

        self.pulse_phase += 0.2

        # 1. Fill zone tiles with risk color tint
        for z in zones:
            r = z["row"]
            c = z["col"]
            risk_level = z["risk_level"]
            color = COLOR_MAP.get(risk_level, (120, 120, 120))
            alpha = ALPHA_MAP.get(risk_level, 0.1)

            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w) if c < cols - 1 else w
            y2 = int((r + 1) * cell_h) if r < rows - 1 else h

            # For Risky, add a subtle pulse effect
            if risk_level == "Risky":
                pulse_factor = 0.12 * (0.5 + 0.5 * np.sin(self.pulse_phase))
                alpha = min(0.55, alpha + pulse_factor)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Blend tint overlay onto canvas
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # 2. Draw person detections (clean, subtle boxes with privacy emphasis)
        for det in detections:
            bbox = det["bbox"]
            zone_rc = det.get("zone", (0, 0))
            # Find zone risk level
            z_match = next((z for z in zones if z["row"] == zone_rc[0] and z["col"] == zone_rc[1]), None)
            box_color = COLOR_MAP.get(z_match["risk_level"] if z_match else "Normal", (200, 200, 200))

            bx1, by1, bx2, by2 = map(int, bbox)
            # Corner accents
            line_len = max(8, int((bx2 - bx1) * 0.25))
            thick = 2
            # Top-left
            cv2.line(canvas, (bx1, by1), (bx1 + line_len, by1), box_color, thick)
            cv2.line(canvas, (bx1, by1), (bx1, by1 + line_len), box_color, thick)
            # Top-right
            cv2.line(canvas, (bx2, by1), (bx2 - line_len, by1), box_color, thick)
            cv2.line(canvas, (bx2, by1), (bx2, by1 + line_len), box_color, thick)
            # Bottom-left
            cv2.line(canvas, (bx1, by2), (bx1 + line_len, by2), box_color, thick)
            cv2.line(canvas, (bx1, by2), (bx1, by2 - line_len), box_color, thick)
            # Bottom-right
            cv2.line(canvas, (bx2, by2), (bx2 - line_len, by2), box_color, thick)
            cv2.line(canvas, (bx2, by2), (bx2, by2 - line_len), box_color, thick)

            # Foot dot (ground contact point)
            foot_x, foot_y = int(det["foot"][0]), int(det["foot"][1])
            cv2.circle(canvas, (foot_x, foot_y), 3, (0, 255, 255), -1)

        # 3. Draw optical flow motion vector hints (optional lightweight subsampling)
        if draw_flow_vectors and flow is not None:
            step = 32
            for y_pt in range(step // 2, h, step):
                for x_pt in range(step // 2, w, step):
                    fx, fy = flow[y_pt, x_pt]
                    mag = np.sqrt(fx * fx + fy * fy)
                    if mag > 1.2:
                        end_x = int(x_pt + fx * 1.5)
                        end_y = int(y_pt + fy * 1.5)
                        cv2.arrowedLine(canvas, (x_pt, y_pt), (end_x, end_y), (180, 240, 255), 1, tipLength=0.3)

        # 4. Draw Zone Boundaries and Data HUD Cards
        for z in zones:
            r = z["row"]
            c = z["col"]
            risk_level = z["risk_level"]
            color = COLOR_MAP.get(risk_level, (180, 180, 180))

            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w) if c < cols - 1 else w
            y2 = int((r + 1) * cell_h) if r < rows - 1 else h

            # Border line
            border_thick = 3 if risk_level in ["Risky", "Dense"] else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, border_thick)

            # Zone Information Card in top-left of each zone
            card_x = x1 + 10
            card_y = y1 + 10
            card_w = min(int(cell_w - 20), 195)
            card_h = 68

            # If predictive bottleneck is active, expand card height
            pred_risk = z.get("predicted_risk_30s", "Normal")
            if pred_risk in ["Risky", "Dense"]:
                card_h += 22

            # Background plate for card
            cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 24, 30), -1)
            cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)

            # Left accent strip
            cv2.rectangle(canvas, (card_x, card_y), (card_x + 5, card_y + card_h), color, -1)

            # Text: Zone ID & Name
            zone_title = f"{z['zone_id']} | {risk_level.upper()}"
            cv2.putText(
                canvas, 
                zone_title, 
                (card_x + 12, card_y + 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.48, 
                color, 
                1, 
                cv2.LINE_AA
            )

            # Stats line 1: Count & Density
            stats_line1 = f"Count: {z['count']}  |  Turb: {z['turbulence_score']}%"
            cv2.putText(
                canvas, 
                stats_line1, 
                (card_x + 12, card_y + 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.38, 
                (220, 220, 220), 
                1, 
                cv2.LINE_AA
            )

            # Stats line 2: Risk Score
            stats_line2 = f"Risk Score: {z['risk_score']}/100"
            cv2.putText(
                canvas, 
                stats_line2, 
                (card_x + 12, card_y + 56), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.38, 
                (180, 200, 210), 
                1, 
                cv2.LINE_AA
            )

            # Stretch feature warning badge
            if pred_risk in ["Risky", "Dense"]:
                warn_color = (0, 140, 255) if pred_risk == "Dense" else (40, 40, 240)
                warn_msg = f"! BOTTLENECK ~30s (Trend +{z.get('trend_slope', 0):.1f}/s)"
                cv2.rectangle(canvas, (card_x + 8, card_y + 64), (card_x + card_w - 8, card_y + card_h - 4), (0, 0, 0), -1)
                cv2.putText(
                    canvas,
                    warn_msg,
                    (card_x + 12, card_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    warn_color,
                    1,
                    cv2.LINE_AA
                )

        # 5. Top Status HUD Bar
        hud_h = 36
        cv2.rectangle(canvas, (0, 0), (w, hud_h), (12, 14, 18), -1)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (50, 60, 75), 1)

        # Title
        cv2.putText(
            canvas,
            "CROWD SENTINEL | REAL-TIME STAMPEDE EARLY WARNING",
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        # Overall Status Badge
        overall_tier = zone_eval.get("overall_tier", "Normal")
        status_color = COLOR_MAP.get(overall_tier, (100, 100, 100))
        cv2.circle(canvas, (w - 360, 18), 6, status_color, -1)
        status_text = f"STATUS: {overall_tier.upper()}"
        cv2.putText(canvas, status_text, (w - 346, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)

        # Headcount & FPS
        total_pax = len(detections)
        stats_top = f"Total: {total_pax} pax | {fps:.1f} FPS"
        cv2.putText(canvas, stats_top, (w - 180, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA)

        # 6. Bottom Privacy Watermark Banner
        banner_h = 24
        cv2.rectangle(canvas, (0, h - banner_h), (w, h), (10, 12, 16), -1)
        privacy_text = "PRIVACY-PRESERVING EDGE AI  *  NO FACIAL RECOGNITION  *  NO IDENTITY TRACKING  *  DENSITY + FLOW METRICS ONLY"
        cv2.putText(
            canvas,
            privacy_text,
            (int(w * 0.04), h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (140, 180, 160),
            1,
            cv2.LINE_AA
        )

        return canvas
