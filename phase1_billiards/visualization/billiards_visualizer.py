"""
Billiards Visualization - Phase 1 Implementation
Author: Eugenie

Input: Ridham's SORT tracker output (frame, track_id, x, y, w, h, cx, cy)
       OR synthetic/dummy data for testing while waiting for tracker

Features:
  - Color-coded fading trajectory trails per ball ID
  - Bounding boxes + ID labels
  - Accumulated heatmap over full video
  - Supports dummy data mode for dev/testing

Usage:
  python billiards_visualizer.py --input video.mp4 --tracks tracks.csv --output viz_output.mp4
  python billiards_visualizer.py --input video.mp4 --dummy  # run with synthetic data
"""

import cv2
import numpy as np
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────
# Color palette: one distinct color per track ID
# ──────────────────────────────────────────────
TRACK_COLORS = [
    (255,  80,  80),   # red
    ( 80, 200, 255),   # sky blue
    ( 80, 255, 120),   # green
    (255, 200,  50),   # gold
    (200,  80, 255),   # purple
    (255, 140,  50),   # orange
    ( 50, 255, 230),   # cyan
    (255,  80, 200),   # pink
    (160, 255,  80),   # lime
    (255, 255,  80),   # yellow
    ( 80,  80, 255),   # blue
    (255, 160, 160),   # salmon
    (160, 255, 200),   # mint
    (200, 160, 255),   # lavender
    (255, 220, 160),   # peach
    ( 80, 200, 160),   # teal
]

def get_color(track_id: int):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# ──────────────────────────────────────────────
# Trail drawing
# ──────────────────────────────────────────────
def draw_trail(frame, trail_points, color, trail_len=40):
    """
    Draw a fading trail for a single track.
    trail_points: list of (cx, cy) in chronological order
    The trail fades from 0% opacity at the oldest point to 100% at the newest.
    """
    pts = trail_points[-trail_len:]  # keep only recent history
    n = len(pts)
    for i in range(1, n):
        alpha = i / n                       # 0 → old, 1 → new
        thickness = max(1, int(3 * alpha))
        opacity = alpha
        color_faded = tuple(int(c * opacity) for c in color)
        cv2.line(frame, pts[i-1], pts[i], color_faded, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Heatmap accumulator
# ──────────────────────────────────────────────
class HeatmapAccumulator:
    def __init__(self, width, height, sigma=15):
        self.width = width
        self.height = height
        self.sigma = sigma
        self.accumulator = np.zeros((height, width), dtype=np.float32)

    def add_point(self, cx, cy):
        """Splat a Gaussian blob at (cx, cy)."""
        x = int(np.clip(cx, 0, self.width - 1))
        y = int(np.clip(cy, 0, self.height - 1))
        self.accumulator[y, x] += 1.0

    def render(self, base_frame=None, alpha=0.6):
        """
        Render heatmap as a colored overlay.
        Returns a BGR image.
        """
        if self.accumulator.max() == 0:
            return base_frame.copy() if base_frame is not None else \
                   np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Gaussian blur to spread the blobs
        blurred = cv2.GaussianBlur(self.accumulator, (0, 0), self.sigma)

        # Normalize to [0, 255]
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        heat_uint8 = norm.astype(np.uint8)

        # Apply colormap (INFERNO: black → purple → red → yellow)
        heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)

        if base_frame is not None:
            # Blend with the base frame
            mask = heat_uint8 > 10  # only blend where there's actual heat
            result = base_frame.copy()
            result[mask] = cv2.addWeighted(
                base_frame, 1 - alpha,
                heatmap_color, alpha, 0
            )[mask]
            return result
        return heatmap_color


# ──────────────────────────────────────────────
# Tracker output loader
# ──────────────────────────────────────────────
def load_tracks_csv(csv_path):
    """
    Expected columns: frame, track_id, x, y, w, h, cx, cy
    Returns: dict[frame_idx] -> list of track dicts
    """
    tracks_by_frame = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracks_by_frame[int(row['frame'])].append({
                'track_id': int(row['track_id']),
                'bbox': [int(row['x']), int(row['y']), int(row['w']), int(row['h'])],
                'centroid': [int(row['cx']), int(row['cy'])],
            })
    return tracks_by_frame


def load_tracks_json(json_path):
    """Alternate loader for JSON format."""
    tracks_by_frame = defaultdict(list)
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        tracks_by_frame[entry['frame']].append({
            'track_id': entry['track_id'],
            'bbox': [entry['x'], entry['y'], entry['w'], entry['h']],
            'centroid': [entry['cx'], entry['cy']],
        })
    return tracks_by_frame


# ──────────────────────────────────────────────
# Dummy / synthetic data generator
# (for testing visualization before tracker is ready)
# ──────────────────────────────────────────────
def generate_dummy_tracks(num_frames, width, height, num_balls=5, seed=42):
    """
    Simulate 'num_balls' balls moving in straight lines with bouncing.
    Returns: dict[frame_idx] -> list of track dicts
    """
    rng = np.random.default_rng(seed)
    tracks_by_frame = defaultdict(list)

    # Init positions and velocities
    positions = rng.uniform(50, min(width, height) - 50, (num_balls, 2))
    velocities = rng.uniform(-4, 4, (num_balls, 2))
    # Give each ball a minimum speed
    for i in range(num_balls):
        if np.linalg.norm(velocities[i]) < 1.5:
            velocities[i] = velocities[i] / np.linalg.norm(velocities[i]) * 2.0

    ball_size = 18  # approximate ball width/height in pixels

    for frame in range(num_frames):
        for ball_id in range(num_balls):
            cx, cy = positions[ball_id]
            vx, vy = velocities[ball_id]

            # Bounce off walls
            if cx - ball_size // 2 <= 0 or cx + ball_size // 2 >= width:
                velocities[ball_id][0] *= -1
            if cy - ball_size // 2 <= 0 or cy + ball_size // 2 >= height:
                velocities[ball_id][1] *= -1

            positions[ball_id] += velocities[ball_id]
            cx, cy = int(positions[ball_id][0]), int(positions[ball_id][1])

            tracks_by_frame[frame].append({
                'track_id': ball_id,
                'bbox': [cx - ball_size//2, cy - ball_size//2, ball_size, ball_size],
                'centroid': [cx, cy],
            })

    return tracks_by_frame


# ──────────────────────────────────────────────
# Main visualization pipeline
# ──────────────────────────────────────────────
def process_video(input_path, tracks_by_frame, output_path, heatmap_path,
                  trail_len=40, show_boxes=True, show_ids=True,
                  heatmap_title="Ball Position Heatmap"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    heatmap = HeatmapAccumulator(width, height)
    trail_history = defaultdict(list)   # track_id -> [(cx, cy), ...]

    last_frame = None  # used for heatmap background

    frame_idx = 0
    print(f"Rendering {total} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()
        viz = frame.copy()

        frame_tracks = tracks_by_frame.get(frame_idx, [])

        for trk in frame_tracks:
            tid = trk['track_id']
            cx, cy = trk['centroid']
            x, y, w, h = trk['bbox']
            color = get_color(tid)

            # Accumulate heatmap
            heatmap.add_point(cx, cy)

            # Update trail
            trail_history[tid].append((cx, cy))

            # Draw trail
            draw_trail(viz, trail_history[tid], color, trail_len=trail_len)

            # Draw bounding box
            if show_boxes:
                cv2.rectangle(viz, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)

            # Draw centroid dot
            cv2.circle(viz, (cx, cy), 4, color, -1, cv2.LINE_AA)

            # Draw ID label
            if show_ids:
                label = f"ID {tid}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(viz, (x, y - lh - 6), (x + lw + 4, y), color, -1)
                cv2.putText(viz, label, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # HUD
        cv2.putText(viz, f"Frame {frame_idx:04d}  |  Tracked: {len(frame_tracks)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(viz)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"Visualization saved: {output_path}")

    # ── Save final heatmap ──
    if heatmap_path and last_frame is not None:
        heatmap_img = heatmap.render(base_frame=last_frame, alpha=0.65)
        cv2.putText(heatmap_img, heatmap_title,
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(heatmap_path, heatmap_img)
        print(f"Heatmap saved:        {heatmap_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Billiards Visualization - Phase 1 (Eugenie)')
    parser.add_argument('--input',    required=True, help='Input video path')
    parser.add_argument('--tracks',   default=None,  help='Tracker CSV (from Ridham). Columns: frame,track_id,x,y,w,h,cx,cy')
    parser.add_argument('--tracks_json', default=None, help='Tracker JSON (alternate format)')
    parser.add_argument('--dummy',    action='store_true', help='Use synthetic data instead of real tracks')
    parser.add_argument('--output',   default='viz_output.mp4',  help='Output visualization video')
    parser.add_argument('--heatmap',  default='heatmap.png',     help='Output heatmap image')
    parser.add_argument('--trail_len',type=int, default=40,      help='Trail length in frames')
    parser.add_argument('--no_boxes', action='store_true',       help='Hide bounding boxes')
    parser.add_argument('--no_ids',   action='store_true',       help='Hide ID labels')
    parser.add_argument('--heatmap_title', default='Ball Position Heatmap',
                        help='Title text overlaid on heatmap (default: Ball Position Heatmap)')
    args = parser.parse_args()

    # ── Load or generate tracks ──
    if args.dummy:
        print("Running in DUMMY MODE with synthetic ball tracks.")
        cap = cv2.VideoCapture(args.input)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        tracks_by_frame = generate_dummy_tracks(total, width, height, num_balls=8)

    elif args.tracks:
        print(f"Loading tracks from CSV: {args.tracks}")
        tracks_by_frame = load_tracks_csv(args.tracks)

    elif args.tracks_json:
        print(f"Loading tracks from JSON: {args.tracks_json}")
        tracks_by_frame = load_tracks_json(args.tracks_json)

    else:
        parser.error("Provide --tracks, --tracks_json, or --dummy")

    process_video(
        input_path    = args.input,
        tracks_by_frame = tracks_by_frame,
        output_path   = args.output,
        heatmap_path  = args.heatmap,
        trail_len     = args.trail_len,
        show_boxes    = not args.no_boxes,
        show_ids      = not args.no_ids,
        heatmap_title = args.heatmap_title,
    )


if __name__ == '__main__':
    main()