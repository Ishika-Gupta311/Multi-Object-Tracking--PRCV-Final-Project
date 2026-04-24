"""
YOLOv8 Detector - Phase 2 (Traffic & Pedestrian)
Author: Samarth (detection) / Ishika (integration)
April 2026

Wraps YOLOv8 for detecting vehicles or pedestrians in video.
Outputs the same CSV format as Phase 1 so the SORT tracker works unchanged.

Usage:
  python yolo_detector.py --input video.mp4 --mode traffic --output_csv detections.csv
  python yolo_detector.py --input video.mp4 --mode pedestrian --output_csv detections.csv
"""

import cv2
import csv
import argparse
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("WARNING: ultralytics not installed. Run: pip3 install ultralytics")


# COCO class IDs for filtering
# Vehicles: car=2, motorcycle=3, bus=5, truck=7
# Pedestrians: person=0
VEHICLE_CLASSES = [2, 3, 5, 7]
PEDESTRIAN_CLASSES = [0]


class YOLODetector:
    """
    YOLOv8 detector that filters by object class depending on scenario.

    Args:
        mode: 'traffic' to detect vehicles, 'pedestrian' to detect people
        model_size: YOLOv8 model variant ('yolov8n', 'yolov8s', 'yolov8m')
        confidence: minimum detection confidence threshold
    """
    def __init__(self, mode='traffic', model_size='yolov8n', confidence=0.4):
        if not HAS_YOLO:
            raise RuntimeError("ultralytics package required. Install with: pip3 install ultralytics")

        self.mode = mode
        self.confidence = confidence
        self.model = YOLO(f'{model_size}.pt')

        if mode == 'traffic':
            self.target_classes = VEHICLE_CLASSES
        else:
            self.target_classes = PEDESTRIAN_CLASSES

    def detect(self, frame):
        """
        Run YOLOv8 on a frame and return filtered detections.

        Args:
            frame: BGR image (numpy array)

        Returns:
            list of detection dicts matching Phase 1 format:
            [{'bbox': [x, y, w, h], 'centroid': [cx, cy], 'confidence': float, 'class_id': int}, ...]
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.target_classes:
                    continue

                # YOLOv8 gives x1, y1, x2, y2 format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                cx, cy = x + w // 2, y + h // 2
                conf = float(box.conf[0])

                detections.append({
                    'bbox': [x, y, w, h],
                    'centroid': [cx, cy],
                    'confidence': conf,
                    'class_id': class_id,
                })

        return detections


def process_video(input_path, output_csv, viz_path, mode, model_size, confidence):
    """Run detection on full video and save CSV in the same format as Phase 1."""
    detector = YOLODetector(mode=mode, model_size=model_size, confidence=confidence)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    viz_writer = None
    if viz_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (width, height))

    csv_rows = []
    frame_idx = 0
    print(f"Processing {total_frames} frames ({mode} mode, {model_size})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det['centroid']
            csv_rows.append([frame_idx, x, y, w, h, cx, cy])

        # Visualization
        if viz_writer:
            viz_frame = frame.copy()
            for det in detections:
                x, y, w, h = det['bbox']
                cx, cy = det['centroid']
                color = (0, 255, 0) if mode == 'traffic' else (255, 100, 0)
                cv2.rectangle(viz_frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(viz_frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(viz_frame, f"Frame: {frame_idx} | {mode}: {len(detections)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            viz_writer.write(viz_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames}")

    cap.release()
    if viz_writer:
        viz_writer.release()

    # Save CSV (same format as Phase 1)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x', 'y', 'w', 'h', 'cx', 'cy'])
        writer.writerows(csv_rows)

    print(f"CSV saved: {output_csv} ({len(csv_rows)} detections)")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Detector - Phase 2')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--mode', choices=['traffic', 'pedestrian'], required=True)
    parser.add_argument('--output_csv', default='detections.csv')
    parser.add_argument('--viz', default=None, help='Output visualization video')
    parser.add_argument('--model', default='yolov8n', help='YOLOv8 model size (yolov8n/s/m)')
    parser.add_argument('--confidence', type=float, default=0.4)
    args = parser.parse_args()

    process_video(args.input, args.output_csv, args.viz, args.mode, args.model, args.confidence)


if __name__ == '__main__':
    main()
