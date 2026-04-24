# Phase 2: Traffic & Pedestrian Tracking

## Setup
```bash
pip3 install ultralytics  # for YOLOv8
```

## Quick Start (full pipeline in one command)

### Traffic
```bash
python run_phase2.py --input ../data/traffic.mp4 --mode traffic --output_dir traffic_output/
```

### Pedestrian
```bash
python run_phase2.py --input ../data/pedestrian.mp4 --mode pedestrian --output_dir ped_output/
```

### Pedestrian with MOT17 evaluation
```bash
python run_phase2.py --input ../data/MOT17-04.mp4 --mode pedestrian --gt ../data/MOT17-04/gt/gt.txt --output_dir mot17_output/
```

## Step-by-Step (manual)

### 1. Detection (YOLOv8)
```bash
python yolo_detector.py --input ../data/traffic.mp4 --mode traffic --output_csv detections.csv --viz detection_viz.mp4
python yolo_detector.py --input ../data/pedestrian.mp4 --mode pedestrian --output_csv detections.csv --viz detection_viz.mp4
```

### 2. Tracking (reuse Phase 1 SORT tracker)
```bash
python ../phase1_billiards/tracking/billards_tracking.py --input_csv detections.csv --output_csv tracked.csv --max_age 30 --dist_thresh 100
```

### 3. Analytics
```bash
# Traffic: line crossing + flow density
python traffic_analytics.py --tracks tracked.csv --output traffic_events.csv --video ../data/traffic.mp4 --viz traffic_viz.mp4

# Pedestrian: zone counting + optional MOT eval
python pedestrian_analytics.py --tracks tracked.csv --output ped_events.csv --gt ../data/MOT17-04/gt/gt.txt
```

### 4. Visualization (reuse Phase 1 visualizer)
```bash
python ../phase1_billiards/visualization/billiards_visualizer.py --input ../data/traffic.mp4 --tracks tracked.csv --output viz_output.mp4 --heatmap heatmap.png
```

## Datasets

### Traffic
- UA-DETRAC: https://detrac-db.rit.albany.edu
- Or any dashcam / traffic camera YouTube video

### Pedestrian
- MOT17: https://motchallenge.net/data/MOT17/
  - Download a sequence like MOT17-04 (fixed camera, crowded)
  - Has ground truth for evaluation
