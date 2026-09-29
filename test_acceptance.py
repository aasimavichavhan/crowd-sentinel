import os
import cv2
from config import SystemConfig
from detector import PersonDetector
from optical_flow import OpticalFlowTurbulence
from risk_engine import RiskEngine

def test_video_pipeline(video_path, expected_max_tier, expect_alerts=False):
    print(f"\n=======================================================")
    print(f"Testing Pipeline on: {os.path.basename(video_path)}")
    print(f"=======================================================")
    config = SystemConfig.get_instance("config.yaml")
    detector = PersonDetector(
        model_name=config.get("detection.model_name", "yolov8n.pt"),
        conf_thresh=float(config.get("detection.confidence_threshold", 0.25))
    )
    flow_engine = OpticalFlowTurbulence(
        turbulence_gain=float(config.get("optical_flow.turbulence_gain", 16.0))
    )
    risk_engine = RiskEngine(config)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    observed_tiers = set()
    all_alerts = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    test_limit = min(total_frames, 250)

    while frame_idx < test_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        if w != 640:
            frame = cv2.resize(frame, (640, int(h * (640.0 / w))))

        det_res = detector.detect_and_assign_zones(frame, rows=3, cols=3)
        flow_res = flow_engine.compute_flow(frame, rows=3, cols=3)
        risk_res = risk_engine.evaluate_zones(
            zone_counts=det_res["zone_counts"],
            zone_turbulence=flow_res["zone_turbulence"],
            rows=3,
            cols=3,
            timestamp_sec=(frame_idx / fps)
        )

        observed_tiers.add(risk_res["overall_tier"])
        if risk_res["new_alerts"]:
            all_alerts.extend(risk_res["new_alerts"])

        if frame_idx % 45 == 0:
            max_z = max(risk_res["zones"], key=lambda z: z["risk_score"])
            print(f"  Frame {frame_idx:3d} ({frame_idx/fps:4.1f}s): Overall={risk_res['overall_tier']:8s} | "
                  f"MaxZone={max_z['zone_id']} ({max_z['risk_level']:8s}, Count={max_z['count']:2d}, "
                  f"Turb={max_z['turbulence_score']:4.1f}%, Score={max_z['risk_score']:4.1f})")

    cap.release()
    print(f"\nResults for {os.path.basename(video_path)}:")
    print(f"  Total frames analyzed: {frame_idx}")
    print(f"  Observed risk tiers:   {observed_tiers}")
    print(f"  Total alerts fired:    {len(all_alerts)}")

    if expect_alerts:
        assert "Risky" in observed_tiers, f"Expected 'Risky' tier in {video_path}, got {observed_tiers}"
        assert len(all_alerts) > 0, f"Expected alerts fired in {video_path}, got 0"
        print(f"  >>> First Alert: [{all_alerts[0]['timestamp']}] {all_alerts[0]['message']}")
        print(f"✅ PASSED: Risky video correctly escalated to Risky with {len(all_alerts)} alerts!")
    else:
        assert "Risky" not in observed_tiers, f"Expected no Risky tiers in calm video, got {observed_tiers}"
        assert len(all_alerts) == 0, f"Expected 0 alerts in calm video, got {len(all_alerts)}"
        print(f"✅ PASSED: Calm video stayed Normal/Moderate with 0 alerts!")

if __name__ == "__main__":
    calm_vid = "sample_videos/calm_crowd.mp4"
    risky_vid = "sample_videos/dense_risky_crowd.mp4"

    test_video_pipeline(calm_vid, expected_max_tier="Moderate", expect_alerts=False)
    test_video_pipeline(risky_vid, expected_max_tier="Risky", expect_alerts=True)
    print("\n🎉 ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")
