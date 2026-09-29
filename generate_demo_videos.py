import cv2
import numpy as np
import os
import random

def extract_pedestrian_sprites(video_path, max_sprites=30):
    from detector import PersonDetector
    detector = PersonDetector(conf_thresh=0.35)
    cap = cv2.VideoCapture(video_path)
    sprites = []
    
    frame_idx = 0
    while cap.isOpened() and len(sprites) < max_sprites:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 8 != 0:
            continue
            
        res = detector.detect_and_assign_zones(frame)
        for det in res["detections"]:
            x1, y1, x2, y2 = map(int, det["bbox"])
            w = x2 - x1
            h = y2 - y1
            if 30 < w < 130 and 60 < h < 260:
                crop = frame[y1:y2, x1:x2].copy()
                sprites.append(crop)
                if len(sprites) >= max_sprites:
                    break
    cap.release()
    print(f"Extracted {len(sprites)} pedestrian sprites")
    return sprites

def create_calm_crowd_video(source_avi, output_mp4, duration_sec=16, fps=15):
    """
    Creates a calm, orderly walking pedestrian clip where all zones stay Normal/Moderate.
    """
    cap = cv2.VideoCapture(source_avi)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

    total_frames = int(duration_sec * fps)
    count = 0
    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        resized = cv2.resize(frame, (w, h))
        out.write(resized)
        count += 1

    cap.release()
    out.release()
    print(f"Calm crowd video created: {output_mp4} ({count} frames)")

def create_dense_risky_crowd_video(source_avi, output_mp4, sprites, duration_sec=20, fps=15):
    """
    Creates a crowd video with 3 distinct operational phases:
    - 0-4s: Normal flow (low density, smooth movement)
    - 4-9s: Noticeable bottleneck build-up in Zone B2 (Central Plaza), triggering 30s surge forecast
    - 9-20s: Severe crowd crush & chaotic panic motion (>14 people in Zone B2 + high optical flow turbulence) -> RISKY ALERTS!
    """
    cap = cv2.VideoCapture(source_avi)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

    total_frames = int(duration_sec * fps)

    center_x = w * 0.5   # Zone B2 center
    center_y = h * 0.55

    class CrowdAgent:
        def __init__(self, sprite, idx):
            self.sprite = sprite
            self.idx = idx
            # Start scattered across upper half
            self.x = random.uniform(80, w - 80)
            self.y = random.uniform(40, h * 0.45)
            self.vx = random.uniform(-1.2, 1.2)
            self.vy = random.uniform(0.8, 1.6)
            self.scale = random.uniform(0.75, 1.05)
            self.phase3_offset_x = random.uniform(-50, 50)
            self.phase3_offset_y = random.uniform(-40, 40)

        def step(self, phase, t):
            if phase == 1:
                # Gentle walk
                self.x += self.vx
                self.y += self.vy * 0.5
            elif phase == 2:
                # Converging towards Zone B2 (Center Plaza)
                target_x = center_x + self.phase3_offset_x * 1.5
                target_y = center_y + self.phase3_offset_y * 1.2
                dx = target_x - self.x
                dy = target_y - self.y
                dist = np.hypot(dx, dy) + 1e-4
                speed = 2.8
                self.x += (dx / dist) * speed
                self.y += (dy / dist) * speed
            elif phase == 3:
                # Chaotic compression & stampede precursor: rapid jostling, erratic surges
                target_x = center_x + self.phase3_offset_x
                target_y = center_y + self.phase3_offset_y
                # High-frequency turbulence oscillations
                jostle_x = np.sin(t * 8.0 + self.idx) * 4.5 + random.uniform(-2.5, 2.5)
                jostle_y = np.cos(t * 7.0 + self.idx * 1.3) * 3.8 + random.uniform(-2.0, 2.0)
                self.x = target_x + jostle_x
                self.y = target_y + jostle_y

    agents = [CrowdAgent(random.choice(sprites), i) for i in range(28)]

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        canvas = cv2.resize(frame, (w, h))
        t = frame_idx / fps

        if t < 4.0:
            phase = 1
            active_count = 5
        elif t < 9.0:
            phase = 2
            # Interpolate from 5 to 20
            progress = (t - 4.0) / 5.0
            active_count = int(5 + progress * 15)
        else:
            phase = 3
            # Full crush: 26+ people packed directly into Zone B2
            active_count = len(agents)

        active = agents[:active_count]
        # Sort by y for realistic perspective layering
        active.sort(key=lambda a: a.y)

        for ag in active:
            ag.step(phase, t)
            sp = ag.sprite
            sh, sw = sp.shape[:2]
            scaled_w = max(24, int(sw * ag.scale))
            scaled_h = max(50, int(sh * ag.scale))
            sp_resized = cv2.resize(sp, (scaled_w, scaled_h))

            px = int(ag.x - scaled_w // 2)
            py = int(ag.y - scaled_h)

            if px >= 0 and py >= 0 and (px + scaled_w) < w and (py + scaled_h) < h:
                roi = canvas[py:py+scaled_h, px:px+scaled_w]
                blended = cv2.addWeighted(sp_resized, 0.92, roi, 0.08, 0)
                canvas[py:py+scaled_h, px:px+scaled_w] = blended

        out.write(canvas)

    cap.release()
    out.release()
    print(f"Dense risky crowd video created: {output_mp4} ({total_frames} frames)")

if __name__ == "__main__":
    vtest = "/Users/aasimavichavhan/crowd_safety_system/sample_videos/calm_crowd_vtest.avi"
    calm_out = "/Users/aasimavichavhan/crowd_safety_system/sample_videos/calm_crowd.mp4"
    risky_out = "/Users/aasimavichavhan/crowd_safety_system/sample_videos/dense_risky_crowd.mp4"

    sprites = extract_pedestrian_sprites(vtest, max_sprites=25)
    create_calm_crowd_video(vtest, calm_out, duration_sec=16, fps=15)
    create_dense_risky_crowd_video(vtest, risky_out, sprites, duration_sec=20, fps=15)
    print("Videos refreshed!")
