"""
Report Generator for Traffic Monitoring System
Generates various report formats
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """
    Generate various types of reports for traffic detection results
    """

    @staticmethod
    def generate_json_report(results, output_path):
        """
        Generate JSON report

        Args:
            results: Detection results dictionary
            output_path: Path to save JSON file
        """
        # Remove non-serializable items
        report_data = {
            k: v for k, v in results.items()
            if k not in ['annotated_image', 'original_image']
        }

        report_data['generated_at'] = datetime.now().isoformat()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=4)

        return output_path

    @staticmethod
    def generate_csv_report(results, output_path):
        """
        Generate CSV report of detections

        Args:
            results: Detection results dictionary
            output_path: Path to save CSV file
        """
        detections = results.get('detections', [])

        if not detections:
            print("[WARNING] No detections to save in CSV")
            return None

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            fieldnames = ['Detection_ID', 'Vehicle_Type', 'Confidence',
                         'BBox_X1', 'BBox_Y1', 'BBox_X2', 'BBox_Y2',
                         'Center_X', 'Center_Y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            for idx, detection in enumerate(detections, 1):
                writer.writerow({
                    'Detection_ID': idx,
                    'Vehicle_Type': detection['class'],
                    'Confidence': detection['confidence'],
                    'BBox_X1': detection['bbox'][0],
                    'BBox_Y1': detection['bbox'][1],
                    'BBox_X2': detection['bbox'][2],
                    'BBox_Y2': detection['bbox'][3],
                    'Center_X': detection['center'][0],
                    'Center_Y': detection['center'][1]
                })

        return output_path

    @staticmethod
    def generate_text_report(results, output_path):
        """
        Generate human-readable text report

        Args:
            results: Detection results dictionary
            output_path: Path to save text file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAFFIC DETECTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            f.write(f"Image: {results['filename']}\n")
            f.write(f"Image Size: {results['image_size']['width']} x {results['image_size']['height']}\n")
            f.write(f"Processing Time: {results['inference_time']:.3f} seconds\n")
            f.write(f"FPS: {results['fps']:.2f}\n\n")

            f.write("-"*70 + "\n")
            f.write("DETECTION SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Vehicles: {results['total_vehicles']}\n")
            f.write(f"Traffic Level: {results['traffic_level']}\n")
            f.write(f"Traffic Density: {results['traffic_density']:.4f}\n\n")

            f.write("Vehicle Breakdown:\n")
            for vehicle_type, count in results['vehicle_counts'].items():
                if count > 0:
                    f.write(f"  {vehicle_type.title()}: {count}\n")

            f.write("\n")
            f.write("-"*70 + "\n")
            f.write("INDIVIDUAL DETECTIONS\n")
            f.write("-"*70 + "\n\n")

            for idx, detection in enumerate(results['detections'], 1):
                f.write(f"Detection #{idx}:\n")
                f.write(f"  Type: {detection['class'].title()}\n")
                f.write(f"  Confidence: {detection['confidence']:.3f}\n")
                f.write(f"  Bounding Box: {detection['bbox']}\n")
                f.write(f"  Center: {detection['center']}\n")
                f.write("\n")

            f.write("="*70 + "\n")

        return output_path

    @staticmethod
    def generate_summary_table(results_list, output_path):
        """
        Generate summary table for multiple images

        Args:
            results_list: List of detection results
            output_path: Path to save CSV file
        """
        summary_data = []

        for result in results_list:
            row = {
                'Image': result['filename'],
                'Total_Vehicles': result['total_vehicles'],
                'Cars': result['vehicle_counts'].get('car', 0),
                'Trucks': result['vehicle_counts'].get('truck', 0),
                'Buses': result['vehicle_counts'].get('bus', 0),
                'Motorcycles': result['vehicle_counts'].get('motorcycle', 0),
                'Bicycles': result['vehicle_counts'].get('bicycle', 0),
                'Traffic_Lights': result['vehicle_counts'].get('traffic light', 0),
                'Traffic_Level': result['traffic_level'],
                'Density': result['traffic_density'],
                'Inference_Time': result['inference_time'],
                'FPS': result['fps']
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        return output_path
