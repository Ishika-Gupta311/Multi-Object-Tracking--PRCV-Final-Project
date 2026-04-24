"""
Pedestrian Analytics - Phase 2
Author: Ishika Gupta
April 2026

Reads tracked CSV from SORT tracker and computes:
  - Entry/exit zone counting (people entering/leaving defined regions)
  - Per-person speed estimation
  - Zone occupancy over time
  - MOT evaluation metrics (MOTA, IDF1, ID switches) against ground truth

Input:  tracked.csv (frame, track_id, x, y, w, h, cx, cy)
Output: pedestrian_events.csv, pedestrian_summary.json

Usage:
  python pedestrian_analytics.py --tracks tracked.csv --output ped_events.csv
  python pedestrian_analytics.py --tracks tracked.csv --gt gt.txt --output ped_events.csv
"""

import numpy as np
import csv
import json
import argparse
from collections import defaultdict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tracked_csv(csv_path):
    """Load tracked output. Returns tracks_by_frame dict and set of all IDs."""
    tracks_by_frame = defaultdict(list)
    all_ids = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            tid = int(row['track_id'])
            tracks_by_frame[frame].append({
                'track_id': tid,
                'bbox': [int(row['x']), int(row['y']), int(row['w']), int(row['h'])],
                'centroid': (int(row['cx']), int(row['cy'])),
            })
            all_ids.add(tid)
    return tracks_by_frame, all_ids


def load_mot_ground_truth(gt_path):
    """
    Load MOT Challenge ground truth format.
    Format: frame, id, x, y, w, h, conf, class, visibility
    Only loads pedestrian entries (class 1) with visibility > 0.

    Returns: dict mapping frame -> list of {'track_id', 'bbox'}
    """
    gt_by_frame = defaultdict(list)
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])

            # Only keep pedestrians (class 1) with positive confidence and visibility
            if class_id == 1 and conf >= 0 and visibility > 0:
                gt_by_frame[frame].append({
                    'track_id': tid,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'centroid': (int(x + w / 2), int(y + h / 2)),
                })
    return gt_by_frame


# ---------------------------------------------------------------------------
# Zone-based Entry/Exit Counter
# ---------------------------------------------------------------------------

class ZoneCounter:
    """
    Counts people entering and exiting rectangular zones.
    Useful for counting people passing through doorways, corridors, etc.

    Args:
        zones: dict of zone_name -> (x1, y1, x2, y2) defining rectangular regions
    """
    def __init__(self, zones):
        self.zones = zones
        # Track which zones each person was in: {track_id: set of zone_names}
        self.prev_zones = defaultdict(set)
        # Counts
        self.entries = defaultdict(int)   # zone_name -> entry count
        self.exits = defaultdict(int)     # zone_name -> exit count
        self.event_log = []

    def _point_in_zone(self, point, zone_rect):
        """Check if a point (cx, cy) is inside a rectangular zone."""
        x1, y1, x2, y2 = zone_rect
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    def update(self, tracked_objects, frame_num):
        """
        Check all tracked people for zone entries and exits.

        Returns:
            list of events this frame
        """
        events = []

        for trk in tracked_objects:
            tid = trk['track_id']
            centroid = trk['centroid']

            current_zones = set()
            for zone_name, zone_rect in self.zones.items():
                if self._point_in_zone(centroid, zone_rect):
                    current_zones.add(zone_name)

            # Check for new entries
            new_entries = current_zones - self.prev_zones[tid]
            for zone in new_entries:
                self.entries[zone] += 1
                event = {
                    'frame': frame_num,
                    'track_id': tid,
                    'event': 'entry',
                    'zone': zone,
                    'position': centroid,
                }
                events.append(event)
                self.event_log.append(event)

            # Check for exits
            new_exits = self.prev_zones[tid] - current_zones
            for zone in new_exits:
                self.exits[zone] += 1
                event = {
                    'frame': frame_num,
                    'track_id': tid,
                    'event': 'exit',
                    'zone': zone,
                    'position': centroid,
                }
                events.append(event)
                self.event_log.append(event)

            self.prev_zones[tid] = current_zones

        return events

    def get_summary(self):
        return {
            'entries': dict(self.entries),
            'exits': dict(self.exits),
            'net_occupancy': {z: self.entries[z] - self.exits[z] for z in self.zones},
        }


# ---------------------------------------------------------------------------
# Speed Estimator
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """Per-person speed estimation from tracked centroids."""
    def __init__(self, fps=30.0, smoothing_window=5):
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.position_history = defaultdict(list)
        self.speed_buffer = defaultdict(list)

    def update(self, track_id, centroid, frame_num):
        self.position_history[track_id].append((centroid[0], centroid[1], frame_num))
        if len(self.position_history[track_id]) < 2:
            return None

        prev = self.position_history[track_id][-2]
        curr = self.position_history[track_id][-1]
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        dist = np.sqrt(dx * dx + dy * dy)
        gap = curr[2] - prev[2]
        if gap <= 0:
            return None

        speed_px_f = dist / gap
        self.speed_buffer[track_id].append(speed_px_f)
        if len(self.speed_buffer[track_id]) > self.smoothing_window:
            self.speed_buffer[track_id] = self.speed_buffer[track_id][-self.smoothing_window:]
        smoothed = float(np.mean(self.speed_buffer[track_id]))

        return {
            'speed_px_per_sec': speed_px_f * self.fps,
            'smoothed_px_per_sec': smoothed * self.fps,
        }


# ---------------------------------------------------------------------------
# MOT Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """Compute IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_mot_metrics(tracks_by_frame, gt_by_frame, iou_threshold=0.5):
    """
    Compute basic MOT evaluation metrics against ground truth.

    Metrics computed:
      - MOTA: Multi-Object Tracking Accuracy
        MOTA = 1 - (FN + FP + ID_switches) / total_GT
      - Precision: TP / (TP + FP)
      - Recall: TP / (TP + FN)
      - ID switches: number of times a GT object changes its matched tracker ID

    Args:
        tracks_by_frame: our tracker output
        gt_by_frame: ground truth from MOT dataset
        iou_threshold: minimum IoU to count as a match

    Returns:
        dict with MOTA, precision, recall, id_switches, etc.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0

    # Track which GT ID was last matched to which tracker ID
    gt_to_tracker = {}

    all_frames = sorted(set(list(tracks_by_frame.keys()) + list(gt_by_frame.keys())))

    for frame in all_frames:
        pred = tracks_by_frame.get(frame, [])
        gt = gt_by_frame.get(frame, [])

        total_gt += len(gt)

        if len(gt) == 0:
            total_fp += len(pred)
            continue
        if len(pred) == 0:
            total_fn += len(gt)
            continue

        # Build IoU cost matrix
        iou_matrix = np.zeros((len(gt), len(pred)))
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                iou_matrix[i, j] = compute_iou(g['bbox'], p['bbox'])

        # Greedy matching (match highest IoU pairs first)
        matched_gt = set()
        matched_pred = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]

            if max_iou < iou_threshold:
                break

            gi, pi = max_idx
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
                total_tp += 1

                # Check for ID switch
                gt_id = gt[gi]['track_id']
                pred_id = pred[pi]['track_id']
                if gt_id in gt_to_tracker:
                    if gt_to_tracker[gt_id] != pred_id:
                        id_switches += 1
                gt_to_tracker[gt_id] = pred_id

            # Zero out matched entries
            iou_matrix[gi, :] = 0
            iou_matrix[:, pi] = 0

        # Unmatched
        total_fn += len(gt) - len(matched_gt)
        total_fp += len(pred) - len(matched_pred)

    # Compute metrics
    mota = 1.0 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        'MOTA': round(mota * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'id_switches': id_switches,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'total_gt': total_gt,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_pedestrian_events_csv(zone_log, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'event_type', 'track_id', 'zone', 'cx', 'cy'])
        for event in zone_log:
            writer.writerow([
                event['frame'], event['event'], event['track_id'],
                event['zone'], event['position'][0], event['position'][1],
            ])
    print(f"Pedestrian events saved: {output_path}")


def save_pedestrian_summary(zone_summary, mot_metrics, total_frames, fps, all_ids, output_path):
    summary = {
        'total_frames': total_frames,
        'duration_sec': round(total_frames / fps, 1),
        'unique_pedestrians': len(all_ids),
        'zone_summary': zone_summary,
    }
    if mot_metrics:
        summary['mot_metrics'] = mot_metrics

    path = output_path.replace('.csv', '_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Pedestrian summary saved: {path}")


# ---------------------------------------------------------------------------
# Visualization overlay
# ---------------------------------------------------------------------------

# Zone display colors: BGR format
ZONE_COLORS = {
    'left':   (255, 100,   0),   # blue
    'right':  (  0, 200, 255),   # orange-yellow
    'top':    (  0, 255, 100),   # green
    'bottom': (180,   0, 255),   # magenta
}


def render_pedestrian_video(video_path, tracks_by_frame, zones, zone_counter,
                            speed_estimator, output_path, fps=30.0):
    """
    Overlay zone boundaries, pedestrian boxes, speed labels, and zone
    occupancy counts on the original video.
    """
    if not HAS_CV2:
        print("OpenCV not available — skipping visualization.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (width, height))

    # Fresh instances for the viz pass
    viz_zone_counter = ZoneCounter(zones)
    viz_speed = SpeedEstimator(fps=fps)

    frame_idx = 0
    print(f"Rendering pedestrian overlay on {total} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        viz = frame.copy()
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        # Draw semi-transparent zone overlays
        overlay = viz.copy()
        for zone_name, (x1, y1, x2, y2) in zones.items():
            color = ZONE_COLORS.get(zone_name, (200, 200, 200))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, viz, 0.85, 0, viz)

        # Draw zone boundary lines
        for zone_name, (x1, y1, x2, y2) in zones.items():
            color = ZONE_COLORS.get(zone_name, (200, 200, 200))
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Update analytics
        zone_events = viz_zone_counter.update(frame_tracks, frame_idx)

        for trk in frame_tracks:
            tid = trk['track_id']
            cx, cy = trk['centroid']
            x, y, w, h = trk['bbox']

            sp = viz_speed.update(tid, trk['centroid'], frame_idx)

            # Draw bounding box
            cv2.rectangle(viz, (x, y), (x + w, y + h), (255, 100, 0), 2)

            # Speed label
            if sp:
                cv2.putText(viz, f"{sp['smoothed_px_per_sec']:.0f} px/s",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Flash on zone entry/exit events
        for event in zone_events:
            pos = event['position']
            color = (0, 255, 0) if event['event'] == 'entry' else (0, 0, 255)
            label = f"{event['event'].upper()} {event['zone']}"
            cv2.circle(viz, (pos[0], pos[1]), 15, color, 2)
            cv2.putText(viz, label, (pos[0] - 30, pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # HUD: zone occupancy
        zone_summary = viz_zone_counter.get_summary()
        hud_parts = []
        for zn in zones:
            net = zone_summary['net_occupancy'].get(zn, 0)
            hud_parts.append(f"{zn}: {net}")
        cv2.putText(viz, f"Occupancy | {' | '.join(hud_parts)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(viz, f"Frame {frame_idx:04d} | Pedestrians: {len(frame_tracks)}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(viz)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"Pedestrian visualization saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(tracks_csv, output_csv, gt_path=None, video_path=None, viz_path=None,
                 fps=30.0, zone_margin_pct=0.15):
    """
    Run pedestrian analytics pipeline.

    Args:
        tracks_csv: path to SORT tracker output
        output_csv: output events CSV path
        gt_path: optional MOT ground truth file for evaluation
        fps: video frame rate
        zone_margin_pct: fraction of frame to use as entry/exit zone width
    """
    print(f"Loading tracks from: {tracks_csv}")
    tracks_by_frame, all_ids = load_tracked_csv(tracks_csv)

    if not tracks_by_frame:
        print("No tracking data found!")
        return

    max_frame = max(tracks_by_frame.keys())
    total_entries = sum(len(v) for v in tracks_by_frame.values())
    print(f"Loaded {total_entries} entries across {max_frame + 1} frames, {len(all_ids)} unique tracks")

    # Estimate frame dimensions
    all_cx = [t['centroid'][0] for tracks in tracks_by_frame.values() for t in tracks]
    all_cy = [t['centroid'][1] for tracks in tracks_by_frame.values() for t in tracks]
    frame_w = max(all_cx) + 50 if all_cx else 1920
    frame_h = max(all_cy) + 50 if all_cy else 1080

    # Define entry/exit zones (borders of the frame)
    margin_x = int(frame_w * zone_margin_pct)
    margin_y = int(frame_h * zone_margin_pct)
    zones = {
        'left':   (0, 0, margin_x, frame_h),
        'right':  (frame_w - margin_x, 0, frame_w, frame_h),
        'top':    (0, 0, frame_w, margin_y),
        'bottom': (0, frame_h - margin_y, frame_w, frame_h),
    }
    print(f"Entry/exit zones: {margin_x}px margins on sides, {margin_y}px on top/bottom")

    # Initialize modules
    zone_counter = ZoneCounter(zones)
    speed_est = SpeedEstimator(fps=fps)

    # Process frame by frame
    for frame_idx in range(max_frame + 1):
        frame_tracks = tracks_by_frame.get(frame_idx, [])
        for trk in frame_tracks:
            speed_est.update(trk['track_id'], trk['centroid'], frame_idx)
        zone_counter.update(frame_tracks, frame_idx)

    # Save events
    save_pedestrian_events_csv(zone_counter.event_log, output_csv)

    # MOT evaluation (if ground truth provided)
    mot_metrics = None
    if gt_path:
        print(f"\nLoading ground truth from: {gt_path}")
        gt_by_frame = load_mot_ground_truth(gt_path)
        gt_total = sum(len(v) for v in gt_by_frame.values())
        print(f"Ground truth: {gt_total} entries across {len(gt_by_frame)} frames")
        mot_metrics = evaluate_mot_metrics(tracks_by_frame, gt_by_frame)

    # Save summary
    zone_summary = zone_counter.get_summary()
    save_pedestrian_summary(zone_summary, mot_metrics, max_frame + 1, fps, all_ids, output_csv)

    # Print summary
    print("\n" + "=" * 60)
    print("PEDESTRIAN ANALYTICS SUMMARY")
    print("=" * 60)
    print(f"Total frames:       {max_frame + 1}")
    print(f"Unique pedestrians: {len(all_ids)}")
    print(f"\nZone activity:")
    for zone_name in zones:
        e = zone_summary['entries'].get(zone_name, 0)
        x = zone_summary['exits'].get(zone_name, 0)
        print(f"  {zone_name:8s}: {e} entries, {x} exits")

    if mot_metrics:
        print(f"\nMOT Evaluation Metrics:")
        print(f"  MOTA:        {mot_metrics['MOTA']}%")
        print(f"  Precision:   {mot_metrics['precision']}%")
        print(f"  Recall:      {mot_metrics['recall']}%")
        print(f"  ID Switches: {mot_metrics['id_switches']}")
        print(f"  TP: {mot_metrics['true_positives']}, FP: {mot_metrics['false_positives']}, FN: {mot_metrics['false_negatives']}")

    print("=" * 60)

    # Optional visualization
    if video_path and viz_path:
        render_pedestrian_video(
            video_path=video_path,
            tracks_by_frame=tracks_by_frame,
            zones=zones,
            zone_counter=zone_counter,
            speed_estimator=speed_est,
            output_path=viz_path,
            fps=fps,
        )


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Analytics - Phase 2 (Ishika)')
    parser.add_argument('--tracks', required=True, help='Tracked CSV from SORT')
    parser.add_argument('--output', default='pedestrian_events.csv')
    parser.add_argument('--gt', default=None, help='MOT ground truth file (for evaluation)')
    parser.add_argument('--video', default=None, help='Original video for viz')
    parser.add_argument('--viz', default=None, help='Output viz video')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--zone_margin', type=float, default=0.15,
                        help='Zone margin as fraction of frame size (default: 0.15)')
    args = parser.parse_args()

    run_analysis(
        tracks_csv=args.tracks,
        output_csv=args.output,
        gt_path=args.gt,
        video_path=args.video,
        viz_path=args.viz,
        fps=args.fps,
        zone_margin_pct=args.zone_margin,
    )


if __name__ == '__main__':
    main()