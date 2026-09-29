# 🛡️ CROWD SENTINEL
### Real-Time Crowd Safety & Stampede Early-Warning System
*Autonomous Edge Video Analytics for Large Public Events & Smart Venues*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Nano_CPU-00599C.svg?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Farneback_Optical_Flow-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)
[![Privacy First](https://img.shields.io/badge/Privacy-Zero_Biometrics-10B981.svg)](#-privacy-first-design)

---

## 1. Executive Summary

Traditional crowd control relies on human operators staring at dozens of CCTV monitors. By the time a human operator spots a surge or stampede, people are already trapped or injured.

**Crowd Sentinel** transforms security from **reactive crisis response** to **proactive early warning**. It splits live CCTV streams into a spatial zone grid (default 3×3) and continuously tracks two independent physical signals in real time:
1. **Density (Spatial Headcount)**: Quantifies how tightly packed individuals are in each physical zone using CPU-optimized YOLOv8n.
2. **Turbulence (Kinetic Agitation)**: Quantifies erratic, high-variance motion vectors using OpenCV Farneback Dense Optical Flow.

> **Why Two Signals Matter:** Most fatal crowd crushes happen not at peak static density, but when a dense crowd experiences sudden compression waves, directional reversals, or shockwaves. Density alone cannot distinguish an orderly walking queue from a collapsing stampede. **Density + Motion Turbulence** catches crushes before they happen.

---

## 2. Architecture & Pipeline

```
           Video Source (Preloaded Clips / Local Upload / CCTV Feed)
                                    │
                                    ▼
                         Frame Extraction (OpenCV)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
YOLOv8n Person Detection                            Farneback Optical Flow
(Class 0, Ground Foot BBox)                         (Dense Vector Field u,v)
         │                                                     │
         ▼                                                     ▼
Per-Zone Headcount & Density                        Per-Zone Kinetic Turbulence
Score (0–100)                                       (Magnitude & Directional Variance)
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
                         Rule-Based Risk Engine
             (Combined Risk Tier: Normal / Moderate / Dense / Risky)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        Linear Trend Extrapolation             Early-Warning Alerts
       ("Bottleneck Surge in ~30s")          (Timestamped Alert Log)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                           HUD Zone Visualizer
          (Color-Coded Tint: Green/Yellow/Orange/Red + Telemetry Cards)
                                    │
                                    ▼
                      FastAPI Real-Time WebSocket
                     (Base64 JPEGs + Zone JSON Data)
                                    │
                                    ▼
                  Single-Page Command Center Dashboard
              (Live Stream + Zone Grid + Chart.js + Dispatch Action)
```

---

## 3. Privacy-First Design

> [!IMPORTANT]
> **Strict Non-Biometric Compliance:** This system **intentionally excludes** facial recognition, person re-identification, age/gender estimation, or any biometric tracking. 
> 
> Detections are immediately translated into localized zone headcounts and anonymized velocity vector statistics. It can be legally deployed in public transit hubs, religious gatherings, and stadiums without violating privacy laws (GDPR, CCPA, or local data privacy mandates).

---

## 4. Quick Start (Run with Two Commands)

### Prerequisites
- Python 3.10+ (macOS, Linux, or Windows)
- Standard laptop CPU (No dedicated GPU required)

### Step 1: Install Dependencies
```bash
cd /Users/aasimavichavhan/crowd_safety_system
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start the System
```bash
python run.py
```
Open your browser to: **`http://localhost:8000`**

*(Both backend streaming and frontend command dashboard run together seamlessly on a single port!)*

---

## 5. Preloaded Demo Clips & Acceptance Testing

The repository comes preloaded with generated benchmark video clips in `sample_videos/`:
1. **`calm_crowd.mp4`**: Orderly pedestrian movement. Stays in **Normal / Moderate** across all zones; **0 alerts** fired.
2. **`dense_risky_crowd.mp4`**: Starts calm, followed by an escalating bottleneck surge in Zone B2 (Central Plaza). Exceeds critical thresholds, triggers the **30-second predictive bottleneck banner**, escalates into **Risky (Red)**, and logs timestamped alerts.

### Run Automated Acceptance Verification
You can verify compliance with one command:
```bash
./venv/bin/python test_acceptance.py
```
Expected output:
```
=======================================================
Testing Pipeline on: calm_crowd.mp4
=======================================================
Results for calm_crowd.mp4:
  Observed risk tiers:   {'Normal', 'Moderate'}
  Total alerts fired:    0
✅ PASSED: Calm video stayed Normal/Moderate with 0 alerts!

=======================================================
Testing Pipeline on: dense_risky_crowd.mp4
=======================================================
Results for dense_risky_crowd.mp4:
  Observed risk tiers:   {'Normal', 'Moderate', 'Dense', 'Risky'}
  Total alerts fired:    1
  >>> First Alert: [10:41:30] CRITICAL: Zone B2 (Central Plaza) reached RISKY state!
✅ PASSED: Risky video correctly escalated to Risky with 1 alerts!

🎉 ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!
```

---

## 6. How to Add Your Own Video Clips

Simply drop any standard `.mp4`, `.avi`, or `.mov` clip into the `sample_videos/` folder:
```bash
cp /path/to/my_crowd_video.mp4 /Users/aasimavichavhan/crowd_safety_system/sample_videos/
```
Refresh the browser page — the new video will appear instantly in the **FEED SOURCE** dropdown!

---

## 7. Tunable Thresholds (`config.yaml`)

Thresholds are kept completely separate from pipeline code so security operators can tune sensitivity for different venue sizes:

```yaml
grid:
  rows: 3
  cols: 3

risk_engine:
  density:
    moderate_count: 3     # >= 3 pax = Moderate
    dense_count: 6        # >= 6 pax = Dense
    risky_count: 10       # >= 10 pax = Risky
  
  turbulence:
    moderate_threshold: 25.0
    high_threshold: 45.0
    risky_threshold: 65.0

  thresholds:
    normal_max: 30.0      # Green
    moderate_max: 52.0    # Yellow
    dense_max: 68.0       # Orange
    # > 68: Risky (Red Early Warning)

  rapid_compression_override: true
```

Operators can also switch sensitivity presets (**High**, **Balanced**, **Relaxed**) on the fly directly from the dashboard UI!

---

## 8. Hackathon Judge Pitch & Demo Script (2 Minutes)

1. **The Hook (15s)**:
   *"In large public gatherings like festivals, train stations, and pilgrimages, reactive security arrives after people are already trapped. Density alone doesn't tell the whole story — people can stand peacefully in high density, but a stampede happens when density meets violent kinetic agitation. That's why we built Crowd Sentinel."*
2. **The Calm Baseline (30s)**:
   * Select `Calm Crowd` in the dropdown.
   * Point out the live HUD: all 9 zones remain Soft Green (`Normal`), risk score stays low, and 0 alerts are fired.
   * Point to the **Privacy Badge**: *"Notice zero face recognition or identity tracking is occurring. It runs entirely on edge CPU."*
3. **The Stampede Early-Warning Spike (45s)**:
   * Switch the dropdown to `Dense Risky Crowd`.
   * Watch the crowd surge into Zone B2 (Central Plaza).
   * Show the **Stretch Feature**: The UI flags `⚠️ Surge ~30s` *before* the disaster occurs based on linear slope extrapolation.
   * As turbulence increases, Zone B2 pulses vibrant red (`Risky`).
   * An early-warning alert fires instantly in the incident log with exact zone coordinates and timestamp, while the real-time Chart.js graph spikes.
4. **The Action (15s)**:
   * Click **NOTIFY CONTROL ROOM** to demonstrate automated webhook/SMS dispatch to ground personnel to open secondary exit gates.
5. **The Value (15s)**:
   * Works on existing CCTV cameras without expensive hardware upgrades.
   * Ready for Smart Cities, religious shrine boards, stadium authorities, and event organizers.
