"""
Traffic Analytics - Phase 2
Author: Ishika Gupta
April 2026

Reads tracked CSV from SORT tracker and computes:
  - Virtual line crossing counter (vehicle counting)
  - Per-vehicle speed estimation
  - Traffic flow density (vehicles per time window)
  - Direction classification (based on crossing direction)

Input:  tracked.csv (frame, track_id, x, y, w, h, cx, cy)
Output: traffic_events.csv, traffic_summary.json

Usage:
  python traffic_analytics.py --tracks tracked.csv --output traffic_events.csv
  python traffic_analytics.py --tracks tracked.csv --video traffic.mp4 --viz traffic_viz.mp4
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
# Data loading — same format as Phase 1
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


# ---------------------------------------------------------------------------
# Line Crossing Counter
# ---------------------------------------------------------------------------

class LineCrossingCounter:
    """
    Counts vehicles crossing a virtual line drawn across the road.
    Uses cross-product to determine which side of the line each object is on.
    A crossing is logged when an object moves from one side to the other.

    Args:
        line_start: (x1, y1) start point of the counting line
        line_end: (x2, y2) end point of the counting line
        direction_labels: tuple of (positive_label, negative_label)
            e.g. ('northbound', 'southbound') for a horizontal line
    """
    def __init__(self, line_start, line_end, direction_labels=('direction_A', 'direction_B')):
        self.line_start = np.array(line_start, dtype=float)
        self.line_end = np.array(line_end, dtype=float)
        self.direction_labels = direction_labels
        self.prev_side = {}
        self.count_positive = 0
        self.count_negative = 0
        self.crossing_log = []

    def _get_side(self, point):
        """Which side of the line is the point on? Returns +1, -1, or 0."""
        line_vec = self.line_end - self.line_start
        point_vec = np.array(point, dtype=float) - self.line_start
        cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0

    def update(self, tracked_objects, frame_num):
        """
        Check all tracked objects for line crossings.

        Args:
            tracked_objects: list of track dicts with 'track_id' and 'centroid'
            frame_num: current frame index

        Returns:
            list of crossing events this frame
        """
        events = []
        for trk in tracked_objects:
            obj_id = trk['track_id']
            centroid = trk['centroid']
            current_side = self._get_side(centroid)

            if current_side == 0:
                continue

            if obj_id in self.prev_side:
                prev = self.prev_side[obj_id]
                if prev != current_side:
                    if current_side == 1:
                        direction = self.direction_labels[0]
                        self.count_positive += 1
                    else:
                        direction = self.direction_labels[1]
                        self.count_negative += 1

                    event = {
                        'frame': frame_num,
                        'track_id': obj_id,
                        'direction': direction,
                        'position': centroid,
                    }
                    events.append(event)
                    self.crossing_log.append(event)

            self.prev_side[obj_id] = current_side

        return events

    def get_counts(self):
        return {
            self.direction_labels[0]: self.count_positive,
            self.direction_labels[1]: self.count_negative,
            'total': self.count_positive + self.count_negative,
        }


# ---------------------------------------------------------------------------
# Speed Estimator (reused from Phase 1 with minor additions)
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """Per-vehicle speed estimation from tracked centroids."""
    def __init__(self, fps=30.0, pixels_per_meter=None, smoothing_window=5):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = smoothing_window
        self.position_history = defaultdict(list)
        self.speed_buffer = defaultdict(list)

    def update(self, track_id, centroid, frame_num):
        self.position_history[track_id].append((centroid[0], centroid[1], frame_num))
        if len(self.position_history[track_id]) < 2:
            return None

        prev = self.position_history[track_id][-2]
        curr = self.position_history[track_id][-1]
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dist = np.sqrt(dx * dx + dy * dy)
        gap = curr[2] - prev[2]
        if gap <= 0:
            return None

        speed_px_f = dist / gap
        speed_px_s = speed_px_f * self.fps

        self.speed_buffer[track_id].append(speed_px_f)
        if len(self.speed_buffer[track_id]) > self.smoothing_window:
            self.speed_buffer[track_id] = self.speed_buffer[track_id][-self.smoothing_window:]
        smoothed = float(np.mean(self.speed_buffer[track_id]))

        result = {
            'speed_px_per_sec': speed_px_s,
            'smoothed_px_per_sec': smoothed * self.fps,
            'velocity': (dx / gap, dy / gap),
        }
        if self.pixels_per_meter and self.pixels_per_meter > 0:
            result['speed_m_per_sec'] = speed_px_s / self.pixels_per_meter
            result['speed_km_per_hr'] = (speed_px_s / self.pixels_per_meter) * 3.6
        return result


# ---------------------------------------------------------------------------
# Traffic Flow Density
# ---------------------------------------------------------------------------

class FlowDensityEstimator:
    """
    Estimates traffic flow density: vehicles counted per time window.

    Args:
        fps: video frame rate
        window_seconds: time window for density calculation
    """
    def __init__(self, fps=30.0, window_seconds=10):
        self.fps = fps
        self.window_frames = int(fps * window_seconds)
        self.window_seconds = window_seconds
        self.vehicle_counts_per_window = []

    def compute(self, crossing_log, max_frame):
        """
        Compute flow density from crossing events over time.

        Returns:
            list of dicts: [{'window_start': s, 'window_end': s, 'count': n, 'flow_per_min': f}, ...]
        """
        results = []
        for window_start in range(0, max_frame + 1, self.window_frames):
            window_end = window_start + self.window_frames
            count = sum(1 for e in crossing_log if window_start <= e['frame'] < window_end)
            flow_per_min = (count / self.window_seconds) * 60
            results.append({
                'window_start_frame': window_start,
                'window_end_frame': window_end,
                'window_start_sec': round(window_start / self.fps, 1),
                'window_end_sec': round(window_end / self.fps, 1),
                'count': count,
                'flow_per_min': round(flow_per_min, 1),
            })
        return results


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_traffic_events_csv(crossing_log, speeds_log, output_path):
    """Save crossing events with speed info."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'event_type', 'track_id', 'direction', 'speed_px_s'])
        for event in crossing_log:
            sp = speeds_log.get((event['frame'], event['track_id']), '')
            writer.writerow([
                event['frame'],
                'line_crossing',
                event['track_id'],
                event['direction'],
                round(sp, 2) if sp != '' else '',
            ])
    print(f"Traffic events saved: {output_path}")


def save_traffic_summary(counts, flow_density, total_frames, fps, output_path):
    """Save summary as JSON."""
    summary = {
        'vehicle_counts': counts,
        'video_duration_sec': round(total_frames / fps, 1),
        'flow_density': flow_density,
    }
    path = output_path.replace('.csv', '_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Traffic summary saved: {path}")


# ---------------------------------------------------------------------------
# Visualization overlay
# ---------------------------------------------------------------------------

def render_traffic_video(video_path, tracks_by_frame, line_counter, speed_estimator,
                         output_path, line_start, line_end):
    """Overlay counting line, vehicle counts, and speed labels on video."""
    if not HAS_CV2:
        print("OpenCV not available — skipping visualization.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Fresh instances for the viz pass
    viz_line_counter = LineCrossingCounter(line_start, line_end,
                                           direction_labels=line_counter.direction_labels)
    viz_speed = SpeedEstimator(fps=fps)

    frame_idx = 0
    print(f"Rendering traffic overlay on {total} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        viz = frame.copy()
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        # Draw counting line
        pt1 = tuple(map(int, line_start))
        pt2 = tuple(map(int, line_end))
        cv2.line(viz, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Update analytics
        crossing_events = viz_line_counter.update(frame_tracks, frame_idx)

        for trk in frame_tracks:
            tid = trk['track_id']
            cx, cy = trk['centroid']
            x, y, w, h = trk['bbox']

            sp = viz_speed.update(tid, trk['centroid'], frame_idx)

            # Draw bounding box
            cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Speed label
            if sp:
                cv2.putText(viz, f"{sp['smoothed_px_per_sec']:.0f} px/s",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Flash on crossings
        for event in crossing_events:
            pos = event['position']
            cv2.circle(viz, (pos[0], pos[1]), 20, (0, 0, 255), 3)
            cv2.putText(viz, "COUNTED", (pos[0] - 30, pos[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD
        counts = viz_line_counter.get_counts()
        labels = viz_line_counter.direction_labels
        cv2.putText(viz, f"Total: {counts['total']} | {labels[0]}: {counts[labels[0]]} | {labels[1]}: {counts[labels[1]]}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(viz, f"Frame {frame_idx:04d} | Vehicles: {len(frame_tracks)}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(viz)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"Traffic visualization saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(tracks_csv, output_csv, video_path=None, viz_path=None,
                 fps=30.0, line_y_pct=0.6, direction_labels=('direction_A', 'direction_B')):
    """
    Run traffic analytics on tracked data.

    Args:
        tracks_csv: path to SORT tracker output
        output_csv: path for events output
        video_path: optional, for visualization
        viz_path: optional, output viz video path
        fps: video frame rate
        line_y_pct: vertical position of counting line (0.0=top, 1.0=bottom)
        direction_labels: names for the two crossing directions
    """
    print(f"Loading tracks from: {tracks_csv}")
    tracks_by_frame, all_ids = load_tracked_csv(tracks_csv)

    if not tracks_by_frame:
        print("No tracking data found!")
        return

    max_frame = max(tracks_by_frame.keys())
    total_entries = sum(len(v) for v in tracks_by_frame.values())
    print(f"Loaded {total_entries} entries across {max_frame + 1} frames, {len(all_ids)} unique tracks")

    # Determine frame dimensions from tracking data for line placement
    all_cx = [t['centroid'][0] for tracks in tracks_by_frame.values() for t in tracks]
    all_cy = [t['centroid'][1] for tracks in tracks_by_frame.values() for t in tracks]
    frame_width = max(all_cx) + 50 if all_cx else 1280
    frame_height = max(all_cy) + 50 if all_cy else 720

    # Place counting line horizontally at specified percentage of frame height
    line_y = int(frame_height * line_y_pct)
    line_start = (0, line_y)
    line_end = (frame_width, line_y)
    print(f"Counting line at y={line_y} ({line_y_pct*100:.0f}% of frame)")

    # Initialize modules
    line_counter = LineCrossingCounter(line_start, line_end, direction_labels)
    speed_est = SpeedEstimator(fps=fps)
    flow_density = FlowDensityEstimator(fps=fps, window_seconds=10)

    speeds_log = {}

    # Process frame by frame
    for frame_idx in range(max_frame + 1):
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        # Speed
        for trk in frame_tracks:
            sp = speed_est.update(trk['track_id'], trk['centroid'], frame_idx)
            if sp:
                speeds_log[(frame_idx, trk['track_id'])] = sp['smoothed_px_per_sec']

        # Line crossing
        line_counter.update(frame_tracks, frame_idx)

    # Compute flow density
    density = flow_density.compute(line_counter.crossing_log, max_frame)

    # Save outputs
    save_traffic_events_csv(line_counter.crossing_log, speeds_log, output_csv)
    counts = line_counter.get_counts()
    save_traffic_summary(counts, density, max_frame + 1, fps, output_csv)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAFFIC ANALYTICS SUMMARY")
    print("=" * 60)
    print(f"Total frames:    {max_frame + 1}")
    print(f"Unique vehicles: {len(all_ids)}")
    print(f"Line crossings:  {counts['total']}")
    print(f"  {direction_labels[0]}: {counts[direction_labels[0]]}")
    print(f"  {direction_labels[1]}: {counts[direction_labels[1]]}")
    if density:
        avg_flow = np.mean([d['flow_per_min'] for d in density])
        print(f"Avg flow rate:   {avg_flow:.1f} vehicles/min")
    print("=" * 60)

    # Optional visualization
    if video_path and viz_path:
        render_traffic_video(video_path, tracks_by_frame, line_counter,
                             speed_est, viz_path, line_start, line_end)


def main():
    parser = argparse.ArgumentParser(description='Traffic Analytics - Phase 2 (Ishika)')
    parser.add_argument('--tracks', required=True, help='Tracked CSV from SORT')
    parser.add_argument('--output', default='traffic_events.csv')
    parser.add_argument('--video', default=None, help='Original video for viz overlay')
    parser.add_argument('--viz', default=None, help='Output visualization video')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--line_y_pct', type=float, default=0.6,
                        help='Counting line position as fraction of frame height (default: 0.6)')
    parser.add_argument('--dir_a', default='direction_A', help='Label for positive crossing direction')
    parser.add_argument('--dir_b', default='direction_B', help='Label for negative crossing direction')
    args = parser.parse_args()

    run_analysis(
        tracks_csv=args.tracks,
        output_csv=args.output,
        video_path=args.video,
        viz_path=args.viz,
        fps=args.fps,
        line_y_pct=args.line_y_pct,
        direction_labels=(args.dir_a, args.dir_b),
    )


if __name__ == '__main__':
    main()
