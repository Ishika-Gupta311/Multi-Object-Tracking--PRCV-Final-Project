"""
Billiards Detection - Phase 1 Implementation
Author: Samarth
Pipeline: Detection -> (outputs for Ridham's SORT tracker)

Outputs per frame: bounding boxes and centroids for each ball
Methods: 'mog2', 'color', 'hybrid' (recommended)
"""

import cv2
import numpy as np
import argparse
import json
import csv
from pathlib import Path

class BilliardsDetector:
    def __init__(self, method='hybrid', min_area=50, max_area=800):
        self.method = method
        self.min_area = min_area
        self.max_area = max_area
        
        # Initialize MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=25, 
            detectShadows=True
        )
        
        # Morphology kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # HSV range for green felt (tune for your table)
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
    
    def detect_mog2(self, frame):
        """Background subtraction detection"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (127) and keep foreground (255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Clean noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        return self._find_balls(fg_mask)
    
    def detect_color(self, frame):
        """Color-based segmentation - invert green felt"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mask green table
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Find the table playing surface: the largest green contour
        # This isolates just the felt area and excludes the border/surroundings
        green_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        table_contours, _ = cv2.findContours(green_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not table_contours:
            return []
        
        # The table is the largest green region
        table_contour = max(table_contours, key=cv2.contourArea)
        
        # Create a mask of just the table interior (shrink slightly to exclude rails)
        table_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(table_mask, [table_contour], -1, 255, -1)
        table_mask = cv2.erode(table_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        
        # Invert green to get non-green objects, but ONLY inside the table area
        balls_mask = cv2.bitwise_not(green_mask)
        balls_mask = cv2.bitwise_and(balls_mask, table_mask)
        
        # Clean
        balls_mask = cv2.morphologyEx(balls_mask, cv2.MORPH_OPEN, self.kernel_open)
        balls_mask = cv2.morphologyEx(balls_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        return self._find_balls(balls_mask)
    
    def detect_hybrid(self, frame):
        """Combine both methods for robustness"""
        mog2_dets = self.detect_mog2(frame)
        color_dets = self.detect_color(frame)
        
        # Simple fusion: take color detections (more stable for static camera)
        # and add MOG2 detections that don't overlap
        detections = color_dets.copy()
        
        for m_det in mog2_dets:
            mx, my = m_det['centroid']
            overlap = False
            for c_det in color_dets:
                cx, cy = c_det['centroid']
                if np.sqrt((mx-cx)**2 + (my-cy)**2) < 20:  # 20px threshold
                    overlap = True
                    break
            if not overlap:
                detections.append(m_det)
        
        return detections
    
    def _find_balls(self, mask):
        """Extract ball contours and filter by size/circularity"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if self.min_area < area < self.max_area:
                # Filter by circularity to remove cues/table edges
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Balls should be fairly circular (0.7-1.2, allow some distortion)
                if circularity > 0.6:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    
                    # Additional aspect ratio check (relaxed for low-res overhead video)
                    aspect = w / float(h)
                    if 0.5 < aspect < 2.0:
                        detections.append({
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'centroid': [int(cx), int(cy)],
                            'area': float(area),
                            'circularity': float(circularity)
                        })
        
        return detections
    
    def detect(self, frame):
        if self.method == 'mog2':
            return self.detect_mog2(frame)
        elif self.method == 'color':
            return self.detect_color(frame)
        else:
            return self.detect_hybrid(frame)


def process_video(input_path, output_csv, output_json, viz_path, method, min_area, max_area):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detector = BilliardsDetector(method=method, min_area=min_area, max_area=max_area)
    
    # Setup video writer for visualization
    viz_writer = None
    if viz_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (width, height))
    
    all_detections = []
    csv_rows = []
    
    frame_idx = 0
    print(f"Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        
        # Store for Ridham (SORT input format)
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det['centroid']
            
            all_detections.append({
                'frame': frame_idx,
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': cx, 'cy': cy
            })
            
            csv_rows.append([frame_idx, x, y, w, h, cx, cy])
        
        # Visualization
        if viz_writer:
            viz_frame = frame.copy()
            for det in detections:
                x, y, w, h = det['bbox']
                cx, cy = det['centroid']
                cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(viz_frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(viz_frame, f"({cx},{cy})", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(viz_frame, f"Frame: {frame_idx} | Detections: {len(detections)} | Method: {method}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            viz_writer.write(viz_frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames}")
    
    cap.release()
    if viz_writer:
        viz_writer.release()
    
    # Save outputs
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'x', 'y', 'w', 'h', 'cx', 'cy'])
            writer.writerows(csv_rows)
        print(f"CSV saved: {output_csv}")
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(all_detections, f, indent=2)
        print(f"JSON saved: {output_json}")
    
    print(f"Done! Total detections: {len(all_detections)}")
    return all_detections


def main():
    parser = argparse.ArgumentParser(description='Billiards Ball Detection - Phase 1')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--method', choices=['mog2', 'color', 'hybrid'], default='color',
                       help='Detection method (color works best for overhead)')
    parser.add_argument('--output_csv', default='detections.csv', help='Output CSV for Ridham')
    parser.add_argument('--output_json', default='detections.json', help='Output JSON')
    parser.add_argument('--viz', default='output_viz.mp4', help='Visualization video path')
    parser.add_argument('--min_area', type=int, default=50, help='Minimum contour area')
    parser.add_argument('--max_area', type=int, default=800, help='Maximum contour area')
    parser.add_argument('--tune_hsv', action='store_true', help='Interactive HSV tuning for green felt')
    
    args = parser.parse_args()
    
    if args.tune_hsv:
        tune_hsv_range(args.input)
        return
    
    process_video(
        args.input, args.output_csv, args.output_json, 
        args.viz, args.method, args.min_area, args.max_area
    )


def tune_hsv_range(video_path):
    """Helper to find correct HSV values for your green table"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    cv2.namedWindow('Tune HSV')
    
    def nothing(x): pass
    
    cv2.createTrackbar('H_low', 'Tune HSV', 35, 179, nothing)
    cv2.createTrackbar('H_high', 'Tune HSV', 85, 179, nothing)
    cv2.createTrackbar('S_low', 'Tune HSV', 40, 255, nothing)
    cv2.createTrackbar('S_high', 'Tune HSV', 255, 255, nothing)
    cv2.createTrackbar('V_low', 'Tune HSV', 40, 255, nothing)
    cv2.createTrackbar('V_high', 'Tune HSV', 255, 255, nothing)
    
    print("Adjust sliders until green table is white in mask. Press 'q' to quit.")
    print("Copy the values to update green_lower/green_upper in code.")
    
    while True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_low = cv2.getTrackbarPos('H_low', 'Tune HSV')
        h_high = cv2.getTrackbarPos('H_high', 'Tune HSV')
        s_low = cv2.getTrackbarPos('S_low', 'Tune HSV')
        s_high = cv2.getTrackbarPos('S_high', 'Tune HSV')
        v_low = cv2.getTrackbarPos('V_low', 'Tune HSV')
        v_high = cv2.getTrackbarPos('V_high', 'Tune HSV')
        
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        cv2.imshow('Tune HSV', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"green_lower = np.array([{h_low}, {s_low}, {v_low}])")
            print(f"green_upper = np.array([{h_high}, {s_high}, {v_high}])")
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
