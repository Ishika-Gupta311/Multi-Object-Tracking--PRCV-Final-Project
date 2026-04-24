"""
Billiards Tracking - Phase 1 Implementation
Author: Ridham
Pipeline: Samarth's Detection CSV -> SORT Tracker -> Tracked output for Eugenie's viz

Implements SORT (Simple Online and Realtime Tracking):
  - Kalman Filter for motion prediction
  - Hungarian Algorithm for detection-to-track assignment
  - Persistent ID management across frames

Input:  detections.csv (frame, x, y, w, h, cx, cy) from billiards_detector.py
Output: tracked.csv  (frame, track_id, x, y, w, h, cx, cy)
        Can also run directly on video with a simple built-in detector.
"""

import numpy as np
import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Hungarian Algorithm (fallback when scipy is not installed)
# ---------------------------------------------------------------------------

def hungarian_assignment(cost_matrix):
    """
    Solve the assignment problem using scipy if available,
    otherwise fall back to a greedy approach.
    Returns list of (row, col) matched pairs.
    """
    if cost_matrix.size == 0:
        return []

    if HAS_SCIPY:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return list(zip(row_indices.tolist(), col_indices.tolist()))

    # Greedy fallback: iteratively pick the smallest cost pair
    assignments = []
    cost = cost_matrix.copy()
    used_rows, used_cols = set(), set()

    num_assignments = min(cost.shape[0], cost.shape[1])
    for _ in range(num_assignments):
        # Mask already-used rows/cols
        temp = cost.copy()
        temp[list(used_rows), :] = np.inf
        temp[:, list(used_cols)] = np.inf

        idx = np.unravel_index(np.argmin(temp), temp.shape)
        if temp[idx] == np.inf:
            break
        assignments.append((int(idx[0]), int(idx[1])))
        used_rows.add(idx[0])
        used_cols.add(idx[1])

    return assignments


# ---------------------------------------------------------------------------
# Kalman Filter (constant-velocity model for 2D centroid + bbox size)
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Tracks a single object using a linear constant-velocity Kalman filter.
    """

    _id_counter = 0  # class-level ID generator

    def __init__(self, detection):
        """
        Initialize tracker from a detection dict:
            {'bbox': [x, y, w, h], 'centroid': [cx, cy]}
        """
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter

        cx, cy = detection['centroid']
        x, y, w, h = detection['bbox']

        # State: [cx, cy, w, h, vx, vy, vs]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity)
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # w  += vs

        # Measurement matrix: observe [cx, cy, w, h]
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance matrices
        self.P = np.eye(7, dtype=np.float64) * 10.0   # state covariance
        self.Q = np.eye(7, dtype=np.float64) * 1.0     # process noise
        self.Q[4, 4] = 0.01  # low velocity noise for billiards
        self.Q[5, 5] = 0.01
        self.Q[6, 6] = 0.001
        self.R = np.eye(4, dtype=np.float64) * 1.0     # measurement noise

        # Bookkeeping
        self.hits = 1           # number of successful updates
        self.age = 0            # frames since creation
        self.time_since_update = 0
        self.history = []       # predicted states when not updated

    def predict(self):
        """Advance state by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1

        # Keep w, h positive
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

        self.history.append(self.x.copy())
        return self.get_state()

    def update(self, detection):
        """Correct state with a matched detection."""
        cx, cy = detection['centroid']
        _, _, w, h = detection['bbox']
        z = np.array([cx, cy, w, h], dtype=np.float64)

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        y = z - self.H @ self.x  # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        self.history = []

    def get_state(self):
        """Return current estimated state as a detection-like dict."""
        cx, cy = self.x[0], self.x[1]
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        x = cx - w / 2
        y_pos = cy - h / 2

        return {
            'bbox': [int(round(x)), int(round(y_pos)), int(round(w)), int(round(h))],
            'centroid': [int(round(cx)), int(round(cy))]
        }


# ---------------------------------------------------------------------------
# SORT Tracker
# ---------------------------------------------------------------------------

class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking
    """

    def __init__(self, max_age=10, min_hits=1, iou_thresh=0.2, dist_thresh=60.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.trackers = []
        self.frame_count = 0

    @staticmethod
    def _iou(bb1, bb2):
        """Compute IoU between two [x, y, w, h] boxes."""
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _centroid_distance(det1, det2):
        """Euclidean distance between two centroids."""
        c1 = np.array(det1['centroid'], dtype=np.float64)
        c2 = np.array(det2['centroid'], dtype=np.float64)
        return np.linalg.norm(c1 - c2)

    def update(self, detections):
        self.frame_count += 1

        # --- Predict new locations for all existing trackers ---
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            predicted.append(pred)
            # Remove trackers with NaN state
            if np.any(np.isnan(trk.x)):
                to_remove.append(i)

        for i in reversed(to_remove):
            self.trackers.pop(i)
            predicted.pop(i)

        # --- Build cost matrix (centroid distance) ---
        num_trk = len(self.trackers)
        num_det = len(detections)

        if num_trk == 0 and num_det == 0:
            return []

        matched_trk = set()
        matched_det = set()
        matches = []

        if num_trk > 0 and num_det > 0:
            cost = np.zeros((num_det, num_trk), dtype=np.float64)

            for d in range(num_det):
                for t in range(num_trk):
                    dist = self._centroid_distance(detections[d], predicted[t])
                    cost[d, t] = dist

            # Run Hungarian assignment
            assignment = hungarian_assignment(cost)

            for d, t in assignment:
                if cost[d, t] < self.dist_thresh:
                    matches.append((d, t))
                    matched_det.add(d)
                    matched_trk.add(t)

        # --- Update matched trackers ---
        for d, t in matches:
            self.trackers[t].update(detections[d])

        # --- Create new trackers for unmatched detections ---
        for d in range(num_det):
            if d not in matched_det:
                self.trackers.append(KalmanBoxTracker(detections[d]))

        # --- Remove dead trackers ---
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # --- Build output ---
        results = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                state = trk.get_state()
                results.append({
                    'track_id': trk.id,
                    'bbox': state['bbox'],
                    'centroid': state['centroid']
                })

        return results


# ---------------------------------------------------------------------------
# CSV I/O helpers (interface with Samarth's detector output)
# ---------------------------------------------------------------------------

def load_detections_csv(csv_path):
    detections_by_frame = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            det = {
                'bbox': [int(row['x']), int(row['y']), int(row['w']), int(row['h'])],
                'centroid': [int(row['cx']), int(row['cy'])]
            }
            detections_by_frame[frame].append(det)

    return detections_by_frame


def save_tracked_csv(output_path, tracked_rows):
    """Save tracked output: frame, track_id, x, y, w, h, cx, cy"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track_id', 'x', 'y', 'w', 'h', 'cx', 'cy'])
        writer.writerows(tracked_rows)


# ---------------------------------------------------------------------------
# Simple built-in detector (so you can test without Samarth's code)
# ---------------------------------------------------------------------------

def simple_color_detect(frame, min_area=50, max_area=800):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    balls_mask = cv2.bitwise_not(green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    balls_mask = cv2.morphologyEx(balls_mask, cv2.MORPH_OPEN, kernel)
    balls_mask = cv2.morphologyEx(balls_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(balls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if 0.7 < aspect < 1.3:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'centroid': [x + w // 2, y + h // 2]
                    })

    return detections


# ---------------------------------------------------------------------------
# Main processing modes
# ---------------------------------------------------------------------------

def run_from_csv(csv_path, output_csv, output_json, max_age, min_hits, dist_thresh):
    print(f"Loading detections from: {csv_path}")
    detections_by_frame = load_detections_csv(csv_path)

    if not detections_by_frame:
        print("No detections found in CSV!")
        return

    max_frame = max(detections_by_frame.keys())
    print(f"Frames 0..{max_frame}, total detection entries: "
          f"{sum(len(v) for v in detections_by_frame.values())}")

    tracker = SORTTracker(
        max_age=max_age,
        min_hits=min_hits,
        dist_thresh=dist_thresh
    )

    all_rows = []
    all_json = []

    for frame_idx in range(max_frame + 1):
        dets = detections_by_frame.get(frame_idx, [])
        tracked = tracker.update(dets)

        for t in tracked:
            x, y, w, h = t['bbox']
            cx, cy = t['centroid']
            all_rows.append([frame_idx, t['track_id'], x, y, w, h, cx, cy])
            all_json.append({
                'frame': frame_idx,
                'track_id': t['track_id'],
                'bbox': t['bbox'],
                'centroid': t['centroid']
            })

    save_tracked_csv(output_csv, all_rows)
    print(f"Tracked CSV saved: {output_csv}")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(all_json, f, indent=2)
        print(f"Tracked JSON saved: {output_json}")

    # Summary
    unique_ids = set(r[1] for r in all_rows)
    print(f"Unique tracks: {len(unique_ids)}")
    print(f"Total tracked entries: {len(all_rows)}")


def run_from_video(video_path, output_csv, output_json, viz_path,
                   max_age, min_hits, dist_thresh):
    """
    Mode 2: Run directly on video (built-in detector + SORT).
    Useful for standalone testing without Samarth's CSV.
    """
    import cv2  # import here so CSV mode works without opencv

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = SORTTracker(
        max_age=max_age,
        min_hits=min_hits,
        dist_thresh=dist_thresh
    )

    # Assign consistent colors to track IDs
    id_colors = {}

    def get_color(track_id):
        if track_id not in id_colors:
            rng = np.random.RandomState(track_id * 37)
            id_colors[track_id] = tuple(int(c) for c in rng.randint(80, 255, 3))
        return id_colors[track_id]

    viz_writer = None
    if viz_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (width, height))

    all_rows = []
    frame_idx = 0
    print(f"Processing {total_frames} frames (video mode)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = simple_color_detect(frame)
        tracked = tracker.update(detections)

        for t in tracked:
            x, y, w, h = t['bbox']
            cx, cy = t['centroid']
            all_rows.append([frame_idx, t['track_id'], x, y, w, h, cx, cy])

        # Draw visualization
        if viz_writer:
            viz_frame = frame.copy()
            for t in tracked:
                tid = t['track_id']
                x, y, w, h = t['bbox']
                cx, cy = t['centroid']
                color = get_color(tid)

                cv2.rectangle(viz_frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(viz_frame, (cx, cy), 4, color, -1)
                cv2.putText(viz_frame, f"ID:{tid}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            info = f"Frame: {frame_idx} | Tracks: {len(tracked)}"
            cv2.putText(viz_frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            viz_writer.write(viz_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames}")

    cap.release()
    if viz_writer:
        viz_writer.release()

    save_tracked_csv(output_csv, all_rows)
    print(f"Tracked CSV saved: {output_csv}")

    if output_json:
        all_json = []
        for r in all_rows:
            all_json.append({
                'frame': r[0], 'track_id': r[1],
                'bbox': [r[2], r[3], r[4], r[5]],
                'centroid': [r[6], r[7]]
            })
        with open(output_json, 'w') as f:
            json.dump(all_json, f, indent=2)
        print(f"Tracked JSON saved: {output_json}")

    unique_ids = set(r[1] for r in all_rows)
    print(f"Done! Unique tracks: {len(unique_ids)}, total entries: {len(all_rows)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Billiards SORT Tracker - Phase 1 (Ridham)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From Samarth's detection CSV (primary mode):
  python billiards_tracker.py --input_csv detections.csv

  # Directly from video (standalone testing):
  python billiards_tracker.py --input_video billiards.mp4 --viz tracked_output.mp4

  # Custom tracker params:
  python billiards_tracker.py --input_csv detections.csv --max_age 8 --dist_thresh 40
        """
    )

    # Input source (one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_csv', help="Samarth's detection CSV path")
    input_group.add_argument('--input_video', help='Video path (uses built-in detector)')

    # Output
    parser.add_argument('--output_csv', default='tracked.csv',
                        help='Output tracked CSV (default: tracked.csv)')
    parser.add_argument('--output_json', default=None,
                        help='Output tracked JSON (optional)')
    parser.add_argument('--viz', default=None,
                        help='Output visualization video (video mode only)')

    # Tracker parameters
    parser.add_argument('--max_age', type=int, default=10,
                        help='Max frames to keep a lost track (default: 10)')
    parser.add_argument('--min_hits', type=int, default=1,
                        help='Min consecutive hits before reporting (default: 1)')
    parser.add_argument('--dist_thresh', type=float, default=60.0,
                        help='Max centroid distance for matching (default: 60px)')

    args = parser.parse_args()

    if args.input_csv:
        run_from_csv(
            csv_path=args.input_csv,
            output_csv=args.output_csv,
            output_json=args.output_json,
            max_age=args.max_age,
            min_hits=args.min_hits,
            dist_thresh=args.dist_thresh
        )
    else:
        run_from_video(
            video_path=args.input_video,
            output_csv=args.output_csv,
            output_json=args.output_json,
            viz_path=args.viz,
            max_age=args.max_age,
            min_hits=args.min_hits,
            dist_thresh=args.dist_thresh
        )


if __name__ == '__main__':
    main()