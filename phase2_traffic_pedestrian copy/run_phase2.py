"""
Phase 2 Pipeline - Traffic & Pedestrian
April 2026

Unified pipeline that runs: YOLOv8 detection -> SORT tracking -> analytics -> visualization
Reuses the SORT tracker and visualizer from Phase 1.

Usage:
  # Traffic scenario
  python run_phase2.py --input traffic.mp4 --mode traffic --output_dir traffic_output/

  # Pedestrian scenario
  python run_phase2.py --input pedestrian.mp4 --mode pedestrian --output_dir ped_output/

  # Pedestrian with MOT17 ground truth evaluation
  python run_phase2.py --input MOT17-04.mp4 --mode pedestrian --gt gt/gt.txt --output_dir mot17_output/
"""

import argparse
import subprocess
import sys
import os


def run_cmd(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Pipeline')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--mode', required=True, choices=['traffic', 'pedestrian'])
    parser.add_argument('--output_dir', default='output/', help='Output directory')
    parser.add_argument('--gt', default=None, help='MOT ground truth file (pedestrian only)')
    parser.add_argument('--model', default='yolov8n', help='YOLOv8 model size')
    parser.add_argument('--confidence', type=float, default=0.4)
    parser.add_argument('--tracker_max_age', type=int, default=30,
                        help='Max frames to keep lost track (higher for crowded scenes)')
    parser.add_argument('--tracker_dist', type=float, default=100.0,
                        help='Max distance for tracker matching (higher for larger objects)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    phase1_tracking = os.path.join(script_dir, '..', 'phase1_billiards', 'tracking')
    phase1_viz = os.path.join(script_dir, '..', 'phase1_billiards', 'visualization')

    det_csv = os.path.join(args.output_dir, 'detections.csv')
    det_viz = os.path.join(args.output_dir, 'detection_viz.mp4')
    tracked_csv = os.path.join(args.output_dir, 'tracked.csv')
    tracked_viz = os.path.join(args.output_dir, 'tracked_viz.mp4')
    viz_output = os.path.join(args.output_dir, 'viz_output.mp4')
    heatmap = os.path.join(args.output_dir, 'heatmap.png')
    events_csv = os.path.join(args.output_dir, f'{args.mode}_events.csv')
    analytics_viz = os.path.join(args.output_dir, f'{args.mode}_analytics_viz.mp4')

    # Step 1: YOLOv8 Detection
    run_cmd([
        sys.executable, os.path.join(script_dir, 'yolo_detector.py'),
        '--input', args.input,
        '--mode', args.mode,
        '--output_csv', det_csv,
        '--viz', det_viz,
        '--model', args.model,
        '--confidence', str(args.confidence),
    ], f'YOLOv8 Detection ({args.mode})')

    # Step 2: SORT Tracking (reuse Phase 1 tracker)
    tracker_script = os.path.join(phase1_tracking, 'billards_tracking.py')
    if os.path.exists(tracker_script):
        run_cmd([
            sys.executable, tracker_script,
            '--input_csv', det_csv,
            '--output_csv', tracked_csv,
            '--max_age', str(args.tracker_max_age),
            '--dist_thresh', str(args.tracker_dist),
        ], 'SORT Tracking')
    else:
        print(f"WARNING: Tracker not found at {tracker_script}")
        print("Copy billards_tracking.py from phase1 or run tracker separately.")
        return

    # Step 3: Visualization (reuse Phase 1 visualizer)
    heatmap_title = 'Vehicle Position Heatmap' if args.mode == 'traffic' else 'Pedestrian Position Heatmap'
    viz_script = os.path.join(phase1_viz, 'billiards_visualizer.py')
    if os.path.exists(viz_script):
        run_cmd([
            sys.executable, viz_script,
            '--input', args.input,
            '--tracks', tracked_csv,
            '--output', viz_output,
            '--heatmap', heatmap,
            '--trail_len', '30',
            '--heatmap_title', heatmap_title,
        ], 'Trajectory Visualization')

    # Step 4: Analytics (scenario-specific)
    if args.mode == 'traffic':
        run_cmd([
            sys.executable, os.path.join(script_dir, 'traffic_analytics.py'),
            '--tracks', tracked_csv,
            '--output', events_csv,
            '--video', args.input,
            '--viz', analytics_viz,
            '--dir_a', 'northbound',
            '--dir_b', 'southbound',
        ], 'Traffic Analytics')
    else:
        cmd = [
            sys.executable, os.path.join(script_dir, 'pedestrian_analytics.py'),
            '--tracks', tracked_csv,
            '--output', events_csv,
            '--video', args.input,
            '--viz', analytics_viz,
        ]
        if args.gt:
            cmd.extend(['--gt', args.gt])
        run_cmd(cmd, 'Pedestrian Analytics')

    print(f"\n{'='*60}")
    print(f"DONE! All outputs in: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()