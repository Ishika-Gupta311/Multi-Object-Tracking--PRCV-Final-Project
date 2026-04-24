"""
Billiards Analytics - Phase 1 Implementation
Author: Ishika Gupta
April 2026

Reads Ridham's tracked CSV and computes:
  - Per-ball speed estimation (px/frame, px/sec, and optionally m/s)
  - Ball-to-ball collision and near-miss detection
  - Summary statistics and event log

Input:  tracked.csv (frame, track_id, x, y, w, h, cx, cy) from billiards_tracker.py
Output: events.csv  (event log with collisions, speeds, near-misses)
        summary printed to stdout

Usage:
  python billiards_analytics.py --tracks ../tracking/tracked.csv --output events.csv
  python billiards_analytics.py --tracks ../tracking/tracked.csv --video ../../data/billiards.mp4 --viz analytics_viz.mp4
"""

import numpy as np
import csv
import argparse
import json
from collections import defaultdict

# Try to import cv2 only when visualization is needed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Data loading — reads Ridham's tracked.csv format
# ---------------------------------------------------------------------------

def load_tracked_csv(csv_path):
    """
    Load tracked output from Ridham's SORT tracker.
    Expected columns: frame, track_id, x, y, w, h, cx, cy

    Returns:
        tracks_by_frame: dict mapping frame_idx -> list of track dicts
        all_track_ids: set of all unique track IDs seen
    """
    tracks_by_frame = defaultdict(list)
    all_track_ids = set()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            tid = int(row['track_id'])
            entry = {
                'track_id': tid,
                'bbox': [int(row['x']), int(row['y']), int(row['w']), int(row['h'])],
                'centroid': (int(row['cx']), int(row['cy'])),
            }
            tracks_by_frame[frame].append(entry)
            all_track_ids.add(tid)

    return tracks_by_frame, all_track_ids


# ---------------------------------------------------------------------------
# Speed Estimation
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Computes per-ball instantaneous and smoothed speed from tracked centroids.

    Speed is computed as the Euclidean distance between consecutive centroid
    positions divided by the frame gap. A rolling average over a configurable
    window provides a smoothed estimate that is less noisy.

    Args:
        fps: video frame rate, used to convert px/frame to px/sec
        pixels_per_meter: optional calibration factor for real-world speed
        smoothing_window: number of frames to average for smoothed speed
    """
    def __init__(self, fps=30.0, pixels_per_meter=None, smoothing_window=5):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = smoothing_window
        # Per-track state: {track_id: [(cx, cy, frame), ...]}
        self.position_history = defaultdict(list)
        # Per-track recent speeds for smoothing: {track_id: [speed, ...]}
        self.speed_buffer = defaultdict(list)

    def update(self, track_id, centroid, frame_num):
        """
        Record a new position for a tracked ball and compute its speed.

        Args:
            track_id: persistent ID from the SORT tracker
            centroid: (cx, cy) pixel coordinates
            frame_num: current frame index

        Returns:
            dict with speed info, or None if this is the first observation
        """
        self.position_history[track_id].append((centroid[0], centroid[1], frame_num))

        if len(self.position_history[track_id]) < 2:
            return None

        prev = self.position_history[track_id][-2]
        curr = self.position_history[track_id][-1]

        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dist_px = np.sqrt(dx * dx + dy * dy)

        frame_gap = curr[2] - prev[2]
        if frame_gap <= 0:
            return None

        speed_px_per_frame = dist_px / frame_gap
        speed_px_per_sec = speed_px_per_frame * self.fps

        # Rolling average for smoothed speed
        self.speed_buffer[track_id].append(speed_px_per_frame)
        if len(self.speed_buffer[track_id]) > self.smoothing_window:
            self.speed_buffer[track_id] = self.speed_buffer[track_id][-self.smoothing_window:]
        smoothed = float(np.mean(self.speed_buffer[track_id]))

        result = {
            'speed_px_per_frame': speed_px_per_frame,
            'speed_px_per_sec': speed_px_per_sec,
            'smoothed_px_per_frame': smoothed,
            'smoothed_px_per_sec': smoothed * self.fps,
            'velocity': (dx / frame_gap, dy / frame_gap),
        }

        if self.pixels_per_meter is not None and self.pixels_per_meter > 0:
            result['speed_m_per_sec'] = speed_px_per_sec / self.pixels_per_meter
            result['smoothed_m_per_sec'] = (smoothed * self.fps) / self.pixels_per_meter

        return result

    def get_velocity(self, track_id):
        """Get most recent velocity vector (vx, vy) in px/frame for a track."""
        hist = self.position_history[track_id]
        if len(hist) < 2:
            return (0.0, 0.0)
        prev, curr = hist[-2], hist[-1]
        gap = curr[2] - prev[2]
        if gap <= 0:
            return (0.0, 0.0)
        return ((curr[0] - prev[0]) / gap, (curr[1] - prev[1]) / gap)

    def get_all_speeds_at_frame(self, tracks_at_frame, frame_num):
        """
        Convenience method: update all tracks in a frame and return speeds.

        Args:
            tracks_at_frame: list of track dicts with 'track_id' and 'centroid'
            frame_num: current frame index

        Returns:
            dict mapping track_id -> speed info dict (or None)
        """
        speeds = {}
        for trk in tracks_at_frame:
            tid = trk['track_id']
            result = self.update(tid, trk['centroid'], frame_num)
            speeds[tid] = result
        return speeds


# ---------------------------------------------------------------------------
# Collision Detection
# ---------------------------------------------------------------------------

class CollisionDetector:
    """
    Detects collisions and near-misses between billiard balls by checking
    pairwise Euclidean distances between tracked centroids each frame.

    A cooldown period prevents the same pair from being flagged repeatedly
    during a single collision event (balls stay close for several frames).

    Args:
        collision_dist: pixel distance threshold for a collision
        near_miss_dist: pixel distance threshold for a near-miss
        cooldown_frames: minimum frames between events for the same pair
    """
    def __init__(self, collision_dist=30, near_miss_dist=60, cooldown_frames=15):
        self.collision_dist = collision_dist
        self.near_miss_dist = near_miss_dist
        self.cooldown_frames = cooldown_frames
        # Cooldown tracker: {(id_a, id_b): last_event_frame}
        self.last_event_frame = {}
        # Full event log
        self.events = []

    def check(self, tracks_at_frame, frame_num, velocities=None):
        """
        Check all pairs of tracked balls for collisions or near-misses.

        Args:
            tracks_at_frame: list of track dicts with 'track_id' and 'centroid'
            frame_num: current frame index
            velocities: optional dict mapping track_id -> (vx, vy) in px/frame.
                        When provided, near-misses are only flagged for pairs
                        that are actively approaching each other.

        Returns:
            list of event dicts detected this frame
        """
        frame_events = []
        n = len(tracks_at_frame)

        for i in range(n):
            for j in range(i + 1, n):
                id_a = tracks_at_frame[i]['track_id']
                id_b = tracks_at_frame[j]['track_id']
                pair = (min(id_a, id_b), max(id_a, id_b))

                ca = np.array(tracks_at_frame[i]['centroid'], dtype=float)
                cb = np.array(tracks_at_frame[j]['centroid'], dtype=float)
                dist = float(np.linalg.norm(ca - cb))

                # Skip if still in cooldown from a recent event for this pair
                if pair in self.last_event_frame:
                    if frame_num - self.last_event_frame[pair] < self.cooldown_frames:
                        continue

                event = None
                if dist < self.collision_dist:
                    event = {
                        'frame': frame_num,
                        'type': 'collision',
                        'id_a': pair[0],
                        'id_b': pair[1],
                        'distance': round(dist, 2),
                        'pos_a': tracks_at_frame[i]['centroid'],
                        'pos_b': tracks_at_frame[j]['centroid'],
                    }
                    self.last_event_frame[pair] = frame_num
                elif dist < self.near_miss_dist:
                    # Require balls to be approaching when velocity data is available
                    approaching = True
                    if velocities is not None:
                        va = velocities.get(id_a, (0.0, 0.0))
                        vb = velocities.get(id_b, (0.0, 0.0))
                        rel_vel = (vb[0] - va[0], vb[1] - va[1])
                        rel_pos = (cb[0] - ca[0], cb[1] - ca[1])
                        # Negative dot product means balls are moving toward each other
                        approaching = (rel_vel[0] * rel_pos[0] + rel_vel[1] * rel_pos[1]) < 0
                    if approaching:
                        event = {
                            'frame': frame_num,
                            'type': 'near_miss',
                            'id_a': pair[0],
                            'id_b': pair[1],
                            'distance': round(dist, 2),
                            'pos_a': tracks_at_frame[i]['centroid'],
                            'pos_b': tracks_at_frame[j]['centroid'],
                        }
                        self.last_event_frame[pair] = frame_num

                if event:
                    frame_events.append(event)
                    self.events.append(event)

        return frame_events

    def get_summary(self):
        """Return a summary of all events detected across the video."""
        collisions = [e for e in self.events if e['type'] == 'collision']
        near_misses = [e for e in self.events if e['type'] == 'near_miss']

        # Which pairs collided most?
        pair_counts = defaultdict(int)
        for e in collisions:
            pair_counts[(e['id_a'], e['id_b'])] += 1

        return {
            'total_collisions': len(collisions),
            'total_near_misses': len(near_misses),
            'collision_pairs': dict(pair_counts),
            'events': self.events,
        }


# ---------------------------------------------------------------------------
# CSV output — event log
# ---------------------------------------------------------------------------

def save_events_csv(events, speeds_log, output_path):
    """
    Save the combined event and speed log to CSV.

    Columns: frame, event_type, id_a, id_b, distance, speed_a, speed_b
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame', 'event_type', 'id_a', 'id_b',
            'distance', 'speed_a_px_s', 'speed_b_px_s'
        ])

        for event in events:
            frame = event['frame']
            # Look up speeds at this frame for both balls
            sp_a = speeds_log.get((frame, event['id_a']), '')
            sp_b = speeds_log.get((frame, event['id_b']), '')
            writer.writerow([
                frame,
                event['type'],
                event['id_a'],
                event['id_b'],
                event['distance'],
                round(sp_a, 2) if sp_a != '' else '',
                round(sp_b, 2) if sp_b != '' else '',
            ])

    print(f"Events CSV saved: {output_path}")


def save_speed_timeseries_csv(speed_timeseries, output_path):
    """
    Save per-ball speed over time for plotting.
    Columns: frame, track_id, speed_px_s, smoothed_px_s
    """
    path = output_path.replace('.csv', '_speeds.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track_id', 'speed_px_s', 'smoothed_px_s'])
        for entry in speed_timeseries:
            writer.writerow([
                entry['frame'],
                entry['track_id'],
                round(entry['speed_px_s'], 2),
                round(entry['smoothed_px_s'], 2),
            ])
    print(f"Speed timeseries saved: {path}")


# ---------------------------------------------------------------------------
# Visualization overlay (optional — requires OpenCV and input video)
# ---------------------------------------------------------------------------

def render_analytics_video(video_path, tracks_by_frame, all_events, speed_estimator,
                           output_path, collision_dist, near_miss_dist):
    """
    Overlay analytics info on the original video:
      - Speed label next to each ball
      - Flash/highlight on collision frames
      - Collision markers at impact points
    """
    if not HAS_CV2:
        print("OpenCV not available — skipping video visualization.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Index events by frame for quick lookup
    events_by_frame = defaultdict(list)
    for e in all_events:
        events_by_frame[e['frame']].append(e)

    # Collision flash duration (frames)
    flash_duration = 8
    # Track recent collision frames for flash effect
    recent_collisions = []  # list of (frame_num, pos_a, pos_b)

    frame_idx = 0
    print(f"Rendering analytics overlay on {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        viz = frame.copy()
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        # Update speed estimator and draw speed labels
        for trk in frame_tracks:
            tid = trk['track_id']
            cx, cy = trk['centroid']
            sp = speed_estimator.update(tid, trk['centroid'], frame_idx)

            if sp is not None:
                speed_text = f"{sp['smoothed_px_per_sec']:.0f} px/s"
                cv2.putText(viz, speed_text, (cx + 12, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Check for events this frame
        frame_events = events_by_frame.get(frame_idx, [])
        for event in frame_events:
            pa = tuple(event['pos_a'])
            pb = tuple(event['pos_b'])
            midpoint = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)

            if event['type'] == 'collision':
                # Draw collision marker: red starburst at midpoint
                cv2.circle(viz, midpoint, 20, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(viz, (midpoint[0]-14, midpoint[1]-14),
                         (midpoint[0]+14, midpoint[1]+14), (0, 0, 255), 2)
                cv2.line(viz, (midpoint[0]+14, midpoint[1]-14),
                         (midpoint[0]-14, midpoint[1]+14), (0, 0, 255), 2)
                cv2.putText(viz, "COLLISION", (midpoint[0]-40, midpoint[1]-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                recent_collisions.append((frame_idx, pa, pb))

            elif event['type'] == 'near_miss':
                # Draw yellow dashed line between the two balls
                cv2.line(viz, pa, pb, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(viz, f"NEAR MISS ({event['distance']:.0f}px)",
                            (midpoint[0]-50, midpoint[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # Flash effect for recent collisions (red border fading out)
        for col_frame, pa, pb in recent_collisions:
            elapsed = frame_idx - col_frame
            if 0 <= elapsed < flash_duration:
                alpha = 1.0 - (elapsed / flash_duration)
                border_color = (0, 0, int(255 * alpha))
                cv2.rectangle(viz, (0, 0), (width - 1, height - 1), border_color,
                              max(1, int(4 * alpha)))

        # Clean up old collision flashes
        recent_collisions = [c for c in recent_collisions
                             if frame_idx - c[0] < flash_duration]

        # HUD: analytics summary
        summary_text = f"Frame {frame_idx:04d} | Balls: {len(frame_tracks)}"
        cv2.putText(viz, summary_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(viz)
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    print(f"Analytics visualization saved: {output_path}")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(tracks_csv, output_csv, video_path=None, viz_path=None,
                 fps=30.0, pixels_per_meter=None,
                 collision_dist=30, near_miss_dist=60, cooldown=15):
    """
    Run the full analytics pipeline on tracked data.

    Steps:
        1. Load tracked CSV from Ridham's SORT output
        2. Process frame-by-frame: compute speeds and detect events
        3. Save event log and speed timeseries to CSV
        4. Print summary statistics
        5. Optionally render analytics overlay video
    """
    # Step 1: Load data
    print(f"Loading tracks from: {tracks_csv}")
    tracks_by_frame, all_ids = load_tracked_csv(tracks_csv)

    if not tracks_by_frame:
        print("No tracking data found!")
        return

    max_frame = max(tracks_by_frame.keys())
    total_entries = sum(len(v) for v in tracks_by_frame.values())
    print(f"Loaded {total_entries} entries across {max_frame + 1} frames, "
          f"{len(all_ids)} unique tracks")

    # Step 2: Initialize modules
    speed_est = SpeedEstimator(
        fps=fps,
        pixels_per_meter=pixels_per_meter,
        smoothing_window=5
    )
    collision_det = CollisionDetector(
        collision_dist=collision_dist,
        near_miss_dist=near_miss_dist,
        cooldown_frames=cooldown
    )

    # Step 3: Process frame by frame
    speeds_log = {}          # (frame, track_id) -> speed_px_per_sec
    speed_timeseries = []    # for the speed CSV

    for frame_idx in range(max_frame + 1):
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        # Speed estimation
        for trk in frame_tracks:
            tid = trk['track_id']
            sp = speed_est.update(tid, trk['centroid'], frame_idx)
            if sp is not None:
                speeds_log[(frame_idx, tid)] = sp['smoothed_px_per_sec']
                speed_timeseries.append({
                    'frame': frame_idx,
                    'track_id': tid,
                    'speed_px_s': sp['speed_px_per_sec'],
                    'smoothed_px_s': sp['smoothed_px_per_sec'],
                })

        # Collision detection — pass velocity vectors so near-misses require
        # balls to be actively approaching each other
        frame_velocities = {trk['track_id']: speed_est.get_velocity(trk['track_id'])
                            for trk in frame_tracks}
        collision_det.check(frame_tracks, frame_idx, frame_velocities)

    # Step 4: Save outputs
    summary = collision_det.get_summary()
    save_events_csv(summary['events'], speeds_log, output_csv)
    save_speed_timeseries_csv(speed_timeseries, output_csv)

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("ANALYTICS SUMMARY")
    print("=" * 60)
    print(f"Total frames analyzed:  {max_frame + 1}")
    print(f"Unique ball tracks:     {len(all_ids)}")
    print(f"Collisions detected:    {summary['total_collisions']}")
    print(f"Near-misses detected:   {summary['total_near_misses']}")

    if summary['collision_pairs']:
        print("\nCollision breakdown by pair:")
        for (a, b), count in sorted(summary['collision_pairs'].items()):
            print(f"  Ball {a} <-> Ball {b}: {count} collision(s)")

    # Per-ball max speed
    max_speeds = defaultdict(float)
    for entry in speed_timeseries:
        tid = entry['track_id']
        max_speeds[tid] = max(max_speeds[tid], entry['smoothed_px_s'])

    if max_speeds:
        print("\nPeak smoothed speed per ball:")
        for tid in sorted(max_speeds.keys()):
            speed_str = f"{max_speeds[tid]:.1f} px/s"
            if pixels_per_meter:
                speed_str += f" ({max_speeds[tid] / pixels_per_meter:.2f} m/s)"
            print(f"  Ball {tid}: {speed_str}")

    print("=" * 60)

    # Step 6: Optional visualization
    if video_path and viz_path:
        # Create a fresh speed estimator for the viz pass
        viz_speed_est = SpeedEstimator(fps=fps, pixels_per_meter=pixels_per_meter)
        render_analytics_video(
            video_path=video_path,
            tracks_by_frame=tracks_by_frame,
            all_events=summary['events'],
            speed_estimator=viz_speed_est,
            output_path=viz_path,
            collision_dist=collision_dist,
            near_miss_dist=near_miss_dist,
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Billiards Analytics - Phase 1 (Ishika)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis from tracked CSV:
  python billiards_analytics.py --tracks ../tracking/tracked.csv --output events.csv

  # With video overlay:
  python billiards_analytics.py --tracks ../tracking/tracked.csv \\
      --video ../../data/billiards.mp4 --viz analytics_viz.mp4 --output events.csv

  # Custom thresholds:
  python billiards_analytics.py --tracks ../tracking/tracked.csv --output events.csv \\
      --collision_dist 25 --near_miss_dist 50 --fps 30
        """
    )

    parser.add_argument('--tracks', required=True,
                        help="Ridham's tracked CSV (frame, track_id, x, y, w, h, cx, cy)")
    parser.add_argument('--output', default='events.csv',
                        help='Output events CSV (default: events.csv)')
    parser.add_argument('--video', default=None,
                        help='Original video path (for visualization overlay)')
    parser.add_argument('--viz', default=None,
                        help='Output analytics visualization video')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Video FPS for speed conversion (default: 30)')
    parser.add_argument('--pixels_per_meter', type=float, default=None,
                        help='Pixel-to-meter calibration (e.g. 500 for a standard pool table)')
    parser.add_argument('--collision_dist', type=float, default=30.0,
                        help='Collision distance threshold in pixels (default: 30)')
    parser.add_argument('--near_miss_dist', type=float, default=60.0,
                        help='Near-miss distance threshold in pixels (default: 60)')
    parser.add_argument('--cooldown', type=int, default=15,
                        help='Cooldown frames between events for same pair (default: 15)')

    args = parser.parse_args()

    run_analysis(
        tracks_csv=args.tracks,
        output_csv=args.output,
        video_path=args.video,
        viz_path=args.viz,
        fps=args.fps,
        pixels_per_meter=args.pixels_per_meter,
        collision_dist=args.collision_dist,
        near_miss_dist=args.near_miss_dist,
        cooldown=args.cooldown,
    )


if __name__ == '__main__':
    main()