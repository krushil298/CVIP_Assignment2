"""
CVIP Assignment-2: Traffic Monitoring System using YOLO
Main Traffic Detection Module

This module implements vehicle detection using YOLOv8 for traffic monitoring.
Detects: Cars, Trucks, Buses, Motorcycles, Bicycles, and Traffic Lights

Author: Student Name
Date: 2025
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import time
from datetime import datetime
import json


class TrafficDetector:
    """
    YOLO-based Traffic Detection System
    Detects and counts vehicles in traffic images
    """

    # Vehicle-related class IDs from COCO dataset
    VEHICLE_CLASSES = {
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7,
        'bicycle': 1,
        'traffic light': 9,
        'stop sign': 11
    }

    # Color mapping for different vehicle types (BGR format)
    COLORS = {
        'car': (0, 255, 0),          # Green
        'motorcycle': (255, 0, 0),    # Blue
        'bus': (0, 165, 255),        # Orange
        'truck': (0, 0, 255),        # Red
        'bicycle': (255, 255, 0),    # Cyan
        'traffic light': (0, 255, 255), # Yellow
        'stop sign': (128, 0, 128)   # Purple
    }

    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.25):
        """
        Initialize Traffic Detector

        Args:
            model_name (str): YOLO model variant
            confidence_threshold (float): Minimum confidence for detection
        """
        print(f"\n{'='*70}")
        print("TRAFFIC MONITORING SYSTEM - YOLO")
        print(f"{'='*70}")
        print(f"[INFO] Loading YOLO model: {model_name}")

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names

        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Confidence threshold: {confidence_threshold}")
        print(f"[INFO] Target vehicle classes: {', '.join(self.VEHICLE_CLASSES.keys())}")
        print(f"{'='*70}\n")

    def detect_traffic(self, image_path, save_path=None, show_result=True):
        """
        Detect vehicles in a traffic image

        Args:
            image_path (str): Path to input image
            save_path (str): Path to save annotated image
            show_result (bool): Display result window

        Returns:
            dict: Detection results with vehicle counts and statistics
        """
        print(f"\n[INFO] Processing: {Path(image_path).name}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return None

        original_image = image.copy()
        height, width = image.shape[:2]

        # Perform detection
        start_time = time.time()
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time

        # Process detections
        vehicle_detections = []
        vehicle_counts = {vehicle: 0 for vehicle in self.VEHICLE_CLASSES.keys()}
        all_detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Extract detection information
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]

                # Check if it's a vehicle class
                if class_name in self.VEHICLE_CLASSES.keys():
                    detection_info = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                    }
                    vehicle_detections.append(detection_info)
                    vehicle_counts[class_name] += 1

                    # Draw bounding box
                    color = self.COLORS.get(class_name, (255, 255, 255))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # Create label
                    label = f"{class_name}: {confidence:.2f}"

                    # Draw label background
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    y1_label = max(y1, label_height + 10)
                    cv2.rectangle(
                        image,
                        (x1, y1_label - label_height - 10),
                        (x1 + label_width, y1_label),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        image,
                        label,
                        (x1, y1_label - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        # Calculate statistics
        total_vehicles = sum(vehicle_counts.values())
        traffic_density = self._calculate_traffic_density(total_vehicles, width, height)
        traffic_level = self._classify_traffic_level(traffic_density)

        # Add summary overlay
        self._add_summary_overlay(image, vehicle_counts, total_vehicles,
                                  inference_time, traffic_level)

        # Print results
        self._print_results(Path(image_path).name, width, height,
                           vehicle_counts, total_vehicles,
                           inference_time, traffic_level, traffic_density)

        # Save annotated image
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            print(f"[INFO] Annotated image saved: {save_path}")

        # Display result
        if show_result:
            self._display_result(image, width, height)

        return {
            'filename': Path(image_path).name,
            'image_size': {'width': width, 'height': height},
            'detections': vehicle_detections,
            'vehicle_counts': vehicle_counts,
            'total_vehicles': total_vehicles,
            'traffic_density': traffic_density,
            'traffic_level': traffic_level,
            'inference_time': inference_time,
            'fps': 1.0 / inference_time if inference_time > 0 else 0,
            'annotated_image': image,
            'original_image': original_image
        }

    def _calculate_traffic_density(self, vehicle_count, width, height):
        """
        Calculate traffic density (vehicles per 10000 square pixels)

        Args:
            vehicle_count (int): Number of vehicles detected
            width (int): Image width
            height (int): Image height

        Returns:
            float: Traffic density value
        """
        area = width * height
        density = (vehicle_count / area) * 10000  # Vehicles per 10k pixels
        return round(density, 4)

    def _classify_traffic_level(self, density):
        """
        Classify traffic level based on density

        Args:
            density (float): Traffic density value

        Returns:
            str: Traffic level classification
        """
        if density < 2:
            return "LIGHT"
        elif density < 5:
            return "MODERATE"
        elif density < 10:
            return "HEAVY"
        else:
            return "CONGESTED"

    def _add_summary_overlay(self, image, vehicle_counts, total_vehicles,
                            inference_time, traffic_level):
        """Add summary information overlay to image"""
        overlay = image.copy()
        height, width = image.shape[:2]

        # Semi-transparent background for summary
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Add summary text
        y_offset = 35
        cv2.putText(image, f"Total Vehicles: {total_vehicles}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        y_offset += 30
        for vehicle_type, count in vehicle_counts.items():
            if count > 0:
                color = self.COLORS.get(vehicle_type, (255, 255, 255))
                cv2.putText(image, f"{vehicle_type.title()}: {count}",
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, color, 2)
                y_offset += 25

        # Traffic level
        level_color = {
            "LIGHT": (0, 255, 0),
            "MODERATE": (0, 255, 255),
            "HEAVY": (0, 165, 255),
            "CONGESTED": (0, 0, 255)
        }
        cv2.putText(image, f"Traffic: {traffic_level}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, level_color.get(traffic_level, (255, 255, 255)), 2)

        # Inference time
        y_offset += 30
        cv2.putText(image, f"Time: {inference_time:.3f}s",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)

    def _print_results(self, filename, width, height, vehicle_counts,
                       total_vehicles, inference_time, traffic_level, density):
        """Print detection results to console"""
        print(f"\n{'='*70}")
        print(f"DETECTION RESULTS: {filename}")
        print(f"{'='*70}")
        print(f"Image Size: {width} x {height} pixels")
        print(f"Inference Time: {inference_time:.3f} seconds")
        print(f"FPS: {1.0/inference_time:.2f}")
        print(f"\n{'Vehicle Type':<20} {'Count':<10}")
        print(f"{'-'*30}")

        for vehicle_type, count in vehicle_counts.items():
            if count > 0:
                print(f"{vehicle_type.title():<20} {count:<10}")

        print(f"{'-'*30}")
        print(f"{'TOTAL':<20} {total_vehicles:<10}")
        print(f"\nTraffic Density: {density:.4f} vehicles/10k pixels")
        print(f"Traffic Level: {traffic_level}")
        print(f"{'='*70}\n")

    def _display_result(self, image, width, height):
        """Display result in a window"""
        # Resize if image is too large
        max_width = 1200
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(image, (new_width, new_height))
        else:
            display_image = image

        cv2.imshow('Traffic Detection Results - Press any key to close', display_image)
        print("[INFO] Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_detection_report(self, results, report_path):
        """
        Save detection results to a JSON file

        Args:
            results (dict): Detection results
            report_path (str): Path to save report
        """
        # Remove non-serializable items
        report_data = {k: v for k, v in results.items()
                      if k not in ['annotated_image', 'original_image']}

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)

        print(f"[INFO] Detection report saved: {report_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Traffic Monitoring System using YOLO - CVIP Assignment-2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python traffic_detector.py --image input_images/traffic.jpg
  python traffic_detector.py --image traffic.jpg --output results/ --conf 0.3
  python traffic_detector.py --image traffic.jpg --model yolov8m.pt --no-show
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input traffic image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_images/annotated',
        help='Output directory for annotated images (default: output_images/annotated)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model variant (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1, default: 0.25)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display result window'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detection report as JSON'
    )

    args = parser.parse_args()

    # Initialize detector
    detector = TrafficDetector(
        model_name=args.model,
        confidence_threshold=args.conf
    )

    # Prepare output paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(args.image).stem
    save_path = output_dir / f"{image_name}_detected.jpg"

    # Perform detection
    results = detector.detect_traffic(
        args.image,
        save_path=save_path,
        show_result=not args.no_show
    )

    # Save report if requested
    if args.save_report and results:
        report_path = Path('reports') / f"{image_name}_report.json"
        detector.save_detection_report(results, report_path)

    print("\n[INFO] Processing complete!")


if __name__ == "__main__":
    main()
