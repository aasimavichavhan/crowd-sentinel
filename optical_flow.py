import cv2
import numpy as np
from typing import Dict, Tuple, Any

class OpticalFlowTurbulence:
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        turbulence_gain: float = 12.0,
        min_motion_threshold: float = 0.3
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.turbulence_gain = turbulence_gain
        self.min_motion_threshold = min_motion_threshold
        self.prev_gray = None
        self.smooth_turbulence: Dict[Tuple[int, int], float] = {}

    def reset(self):
        self.prev_gray = None
        self.smooth_turbulence.clear()

    def compute_flow(
        self, 
        frame: np.ndarray, 
        rows: int = 3, 
        cols: int = 3
    ) -> Dict[str, Any]:
        """
        Computes dense Farneback optical flow and calculates per-zone
        turbulence based on motion magnitude and directional variance.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cell_h = int(h / rows)
        cell_w = int(w / cols)

        zone_turbulence = {(r, c): 0.0 for r in range(rows) for c in range(cols)}
        zone_mean_mag = {(r, c): 0.0 for r in range(rows) for c in range(cols)}
        zone_std_mag = {(r, c): 0.0 for r in range(rows) for c in range(cols)}

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return {
                "flow": None,
                "zone_turbulence": zone_turbulence,
                "zone_mean_mag": zone_mean_mag,
                "zone_std_mag": zone_std_mag,
                "average_turbulence": 0.0
            }

        # Calculate dense optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.prev_gray,
            next=gray,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )
        self.prev_gray = gray

        # Flow components
        u = flow[..., 0]
        v = flow[..., 1]
        magnitude = np.sqrt(u**2 + v**2)

        # Filter out minor sensor noise
        magnitude[magnitude < self.min_motion_threshold] = 0.0

        all_turbulence = []

        for r in range(rows):
            y_start = r * cell_h
            y_end = (r + 1) * cell_h if r < rows - 1 else h
            for c in range(cols):
                x_start = c * cell_w
                x_end = (c + 1) * cell_w if c < cols - 1 else w

                sub_mag = magnitude[y_start:y_end, x_start:x_end]
                sub_u = u[y_start:y_end, x_start:x_end]
                sub_v = v[y_start:y_end, x_start:x_end]

                if sub_mag.size == 0:
                    continue

                mean_val = float(np.mean(sub_mag))
                std_val = float(np.std(sub_mag))

                # Directional variance (angles in radians)
                # Chaotic crowds have high directional entropy / variance
                active_mask = sub_mag > self.min_motion_threshold
                if np.count_nonzero(active_mask) > 10:
                    angles = np.arctan2(sub_v[active_mask], sub_u[active_mask])
                    # Circular variance: 1 - |mean(exp(i * theta))|
                    cos_mean = np.mean(np.cos(angles))
                    sin_mean = np.mean(np.sin(angles))
                    R = np.sqrt(cos_mean**2 + sin_mean**2)
                    dir_variance = float(1.0 - R) # 0 = uniform flow, 1 = complete chaos
                else:
                    dir_variance = 0.0

                # Composite turbulence metric:
                # Combines motion intensity (mean), erratic speed spikes (std),
                # and multi-directional chaos (dir_variance)
                raw_score = (
                    0.35 * mean_val +
                    0.35 * std_val +
                    0.30 * (mean_val * dir_variance * 2.0)
                ) * self.turbulence_gain

                # Scaled to 0.0 - 100.0 with smooth saturation
                raw_clipped = float(np.clip(raw_score, 0.0, 100.0))
                prev_val = self.smooth_turbulence.get((r, c), raw_clipped)
                # Exponential moving average (40% new, 60% historical)
                norm_score = 0.4 * raw_clipped + 0.6 * prev_val
                self.smooth_turbulence[(r, c)] = norm_score

                zone_turbulence[(r, c)] = round(norm_score, 1)
                zone_mean_mag[(r, c)] = round(mean_val, 2)
                zone_std_mag[(r, c)] = round(std_val, 2)
                all_turbulence.append(norm_score)

        avg_turb = round(float(np.mean(all_turbulence)), 1) if all_turbulence else 0.0

        return {
            "flow": flow,
            "zone_turbulence": zone_turbulence,
            "zone_mean_mag": zone_mean_mag,
            "zone_std_mag": zone_std_mag,
            "average_turbulence": avg_turb
        }
