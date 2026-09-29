import time
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class RiskEngine:
    def __init__(self, config_obj):
        self.config = config_obj
        self.zone_history: Dict[Tuple[int, int], deque] = {}
        self.last_alert_time: Dict[Tuple[int, int], float] = {}
        self.alerts: List[Dict[str, Any]] = []

    def reset(self):
        self.zone_history.clear()
        self.last_alert_time.clear()
        self.alerts.clear()

    def _normalize_density(self, count: int) -> float:
        """
        Maps person count to a 0.0 - 100.0 score based on config thresholds.
        """
        dens_cfg = self.config.get("risk_engine.density", {})
        mod_c = float(dens_cfg.get("moderate_count", 4))
        dense_c = float(dens_cfg.get("dense_count", 8))
        risky_c = float(dens_cfg.get("risky_count", 14))

        if count <= 0:
            return 0.0
        elif count <= mod_c:
            return (count / mod_c) * 40.0
        elif count <= dense_c:
            return 40.0 + ((count - mod_c) / (dense_c - mod_c)) * 30.0
        elif count <= risky_c:
            return 70.0 + ((count - dense_c) / (risky_c - dense_c)) * 25.0
        else:
            overflow = min(count - risky_c, 10.0)
            return min(100.0, 95.0 + (overflow / 10.0) * 5.0)

    def _classify_risk_tier(self, risk_score: float, count: int, turbulence: float) -> str:
        thresh = self.config.get("risk_engine.thresholds", {})
        norm_max = float(thresh.get("normal_max", 30.0))
        mod_max = float(thresh.get("moderate_max", 52.0))
        dense_max = float(thresh.get("dense_max", 68.0))

        # Check rapid compression override: high motion turbulence in an already crowded zone
        rapid_override = self.config.get("risk_engine.rapid_compression_override", True)
        turb_risky_thresh = float(self.config.get("risk_engine.turbulence.risky_threshold", 70.0))
        dense_c = float(self.config.get("risk_engine.density.dense_count", 6))

        if rapid_override and turbulence >= turb_risky_thresh and count >= dense_c:
            return "Risky"

        if risk_score <= norm_max:
            return "Normal"
        elif risk_score <= mod_max:
            return "Moderate"
        elif risk_score <= dense_max:
            return "Dense"
        else:
            return "Risky"

    def _compute_trend_prediction(
        self, 
        zone_key: Tuple[int, int], 
        current_count: int, 
        current_time: float
    ) -> Tuple[str, float]:
        """
        Stretch feature: simple linear trend extrapolation on zone density history
        to forecast risk in ~30 seconds.
        """
        pred_cfg = self.config.get("prediction", {})
        if not pred_cfg.get("enabled", True):
            return "Normal", 0.0

        history_win = float(pred_cfg.get("history_window_seconds", 8.0))
        forecast_sec = float(pred_cfg.get("forecast_seconds", 30.0))
        slope_thresh = float(pred_cfg.get("slope_threshold", 0.8))

        if zone_key not in self.zone_history:
            self.zone_history[zone_key] = deque()

        hist = self.zone_history[zone_key]
        hist.append((current_time, current_count))

        # Evict old entries outside window
        while hist and (current_time - hist[0][0]) > history_win:
            hist.popleft()

        if len(hist) < 3:
            return "Normal", 0.0

        times = np.array([pt[0] - current_time for pt in hist])
        counts = np.array([pt[1] for pt in hist])

        # Simple linear regression (slope)
        cov = np.cov(times, counts)
        if cov[0, 0] > 1e-4:
            slope = float(cov[0, 1] / cov[0, 0])  # people added per second
        else:
            slope = 0.0

        predicted_count_30s = current_count + (slope * forecast_sec)
        risky_c = float(self.config.get("risk_engine.density.risky_count", 14))

        if slope >= slope_thresh and predicted_count_30s >= risky_c:
            predicted_tier = "Risky"
        elif slope >= (slope_thresh * 0.5) and predicted_count_30s >= (risky_c * 0.7):
            predicted_tier = "Dense"
        elif slope > 0.1:
            predicted_tier = "Moderate"
        else:
            predicted_tier = "Normal"

        return predicted_tier, round(slope, 2)

    def evaluate_zones(
        self,
        zone_counts: Dict[Tuple[int, int], int],
        zone_turbulence: Dict[Tuple[int, int], float],
        rows: int = 3,
        cols: int = 3,
        timestamp_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates all zones, computes density scores, combined risk scores,
        risk tiers, 30s trend prediction, and generates alerts when 'Risky'.
        """
        now = time.time() if timestamp_sec is None else timestamp_sec
        now_str = datetime.now().strftime("%H:%M:%S")

        w_dens = float(self.config.get("risk_engine.weights.density", 0.55))
        w_turb = float(self.config.get("risk_engine.weights.turbulence", 0.45))
        zone_names_map = self.config.get("grid.zone_names", {})
        alert_cooldown = float(self.config.get("alert.cooldown_seconds", 4.0))

        zone_results = []
        new_alerts = []
        overall_risk_sum = 0.0

        for r in range(rows):
            for c in range(cols):
                zone_key = (r, c)
                zone_id = f"Z{r * cols + c + 1}"
                name_key = f"{r},{c}"
                zone_name = zone_names_map.get(name_key, f"Zone {zone_id} ({r},{c})")

                count = zone_counts.get(zone_key, 0)
                turbulence = zone_turbulence.get(zone_key, 0.0)

                density_score = self._normalize_density(count)

                # Combined risk formula
                combined_score = round(
                    float(np.clip(
                        (density_score * w_dens) + (turbulence * w_turb),
                        0.0,
                        100.0
                    )),
                    1
                )
                overall_risk_sum += combined_score

                # Risk level classification
                risk_level = self._classify_risk_tier(combined_score, count, turbulence)

                # Trend extrapolation (stretch feature)
                pred_tier, slope = self._compute_trend_prediction(zone_key, count, now)

                zone_info = {
                    "zone_id": zone_id,
                    "row": r,
                    "col": c,
                    "name": zone_name,
                    "count": count,
                    "density_score": round(density_score, 1),
                    "turbulence_score": turbulence,
                    "risk_score": combined_score,
                    "risk_level": risk_level,
                    "predicted_risk_30s": pred_tier,
                    "trend_slope": slope
                }
                zone_results.append(zone_info)

                # Alert generation when zone enters "Risky"
                if risk_level == "Risky":
                    last_time = self.last_alert_time.get(zone_key, 0.0)
                    if (now - last_time) >= alert_cooldown:
                        self.last_alert_time[zone_key] = now
                        alert_event = {
                            "id": f"alert-{int(now * 1000)}-{zone_id}",
                            "zone_id": zone_id,
                            "zone_name": zone_name,
                            "risk_level": "Risky",
                            "risk_score": combined_score,
                            "density_score": round(density_score, 1),
                            "people_count": count,
                            "turbulence_score": turbulence,
                            "timestamp": now_str,
                            "message": f"CRITICAL: {zone_name} reached RISKY state! Density: {count} pax, Turbulence: {turbulence}%"
                        }
                        new_alerts.append(alert_event)
                        self.alerts.insert(0, alert_event)
                        # Keep history limit
                        history_limit = int(self.config.get("alert.history_limit", 100))
                        if len(self.alerts) > history_limit:
                            self.alerts = self.alerts[:history_limit]

        avg_risk = round(overall_risk_sum / (rows * cols), 1) if (rows * cols) > 0 else 0.0

        # Overall severity
        if any(z["risk_level"] == "Risky" for z in zone_results):
            overall_tier = "Risky"
        elif any(z["risk_level"] == "Dense" for z in zone_results):
            overall_tier = "Dense"
        elif any(z["risk_level"] == "Moderate" for z in zone_results):
            overall_tier = "Moderate"
        else:
            overall_tier = "Normal"

        return {
            "zones": zone_results,
            "average_risk": avg_risk,
            "overall_tier": overall_tier,
            "new_alerts": new_alerts,
            "recent_alerts": self.alerts[:15]
        }
