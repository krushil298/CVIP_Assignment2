"""
CVIP Assignment-2: Batch Traffic Processing
Process multiple traffic images and generate comprehensive reports

This module processes multiple traffic images in batch mode and creates:
- Individual detection results for each image
- Aggregate statistics across all images
- Comparison visualizations
- Summary reports

Author: Student Name
Date: 2025
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from traffic_detector import TrafficDetector
from traffic_analyzer import TrafficAnalyzer


class BatchProcessor:
    """
    Batch Processing for Multiple Traffic Images
    """

    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.25):
        """
        Initialize Batch Processor

        Args:
            model_name (str): YOLO model variant
            confidence_threshold (float): Detection confidence threshold
        """
        print(f"\n{'='*70}")
        print("BATCH TRAFFIC PROCESSING SYSTEM")
        print(f"{'='*70}\n")

        self.detector = TrafficDetector(model_name, confidence_threshold)
        self.analyzer = TrafficAnalyzer()

    def process_directory(self, input_dir, output_dir='output_images', generate_reports=True):
        """
        Process all images in a directory

        Args:
            input_dir (str): Directory containing traffic images
            output_dir (str): Output directory for results
            generate_reports (bool): Generate analysis reports

        Returns:
            dict: Batch processing summary
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directories
        annotated_dir = output_path / 'annotated'
        analysis_dir = output_path / 'analysis'
        reports_dir = Path('reports')

        annotated_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"[ERROR] No images found in {input_dir}")
            return None

        print(f"[INFO] Found {len(image_files)} images to process")
        print(f"[INFO] Output directory: {output_path}")
        print(f"\n{'='*70}\n")

        # Process each image
        all_results = []
        failed_images = []

        for image_path in tqdm(image_files, desc="Processing images", unit="img"):
            try:
                # Detection
                save_path = annotated_dir / f"{image_path.stem}_detected{image_path.suffix}"
                results = self.detector.detect_traffic(
                    str(image_path),
                    save_path=str(save_path),
                    show_result=False
                )

                if results:
                    all_results.append(results)

                    # Individual analysis
                    if generate_reports:
                        self.analyzer.analyze_single_image(results, analysis_dir)

            except Exception as e:
                print(f"\n[ERROR] Failed to process {image_path.name}: {str(e)}")
                failed_images.append(image_path.name)

        # Generate batch summary
        if all_results:
            summary = self._generate_batch_summary(all_results, reports_dir)

            # Comparison visualization
            if len(all_results) > 1:
                self.analyzer.compare_multiple_images(
                    all_results,
                    analysis_dir / 'batch_comparison.png'
                )

            self._print_batch_summary(summary, failed_images)

            return summary
        else:
            print("\n[ERROR] No images were successfully processed")
            return None

    def _generate_batch_summary(self, results_list, reports_dir):
        """Generate comprehensive batch summary"""
        summary = {
            'total_images': len(results_list),
            'total_vehicles_detected': sum(r['total_vehicles'] for r in results_list),
            'average_vehicles_per_image': np.mean([r['total_vehicles'] for r in results_list]),
            'average_inference_time': np.mean([r['inference_time'] for r in results_list]),
            'average_fps': np.mean([r['fps'] for r in results_list]),
            'traffic_levels': {},
            'vehicle_type_totals': {}
        }

        # Aggregate vehicle types
        for result in results_list:
            # Traffic levels count
            level = result['traffic_level']
            summary['traffic_levels'][level] = summary['traffic_levels'].get(level, 0) + 1

            # Vehicle types aggregation
            for vehicle_type, count in result['vehicle_counts'].items():
                if count > 0:
                    summary['vehicle_type_totals'][vehicle_type] = \
                        summary['vehicle_type_totals'].get(vehicle_type, 0) + count

        # Create detailed CSV report
        self._create_detailed_csv(results_list, reports_dir / 'batch_detailed_report.csv')

        # Create summary CSV
        self._create_summary_csv(summary, results_list, reports_dir / 'batch_summary.csv')

        # Create text summary
        self._create_text_summary(summary, results_list, reports_dir / 'batch_summary.txt')

        return summary

    def _create_detailed_csv(self, results_list, save_path):
        """Create detailed CSV with all detections"""
        all_data = []

        for result in results_list:
            for detection in result['detections']:
                all_data.append({
                    'Image': result['filename'],
                    'Vehicle_Type': detection['class'],
                    'Confidence': detection['confidence'],
                    'BBox_X1': detection['bbox'][0],
                    'BBox_Y1': detection['bbox'][1],
                    'BBox_X2': detection['bbox'][2],
                    'BBox_Y2': detection['bbox'][3],
                    'Center_X': detection['center'][0],
                    'Center_Y': detection['center'][1],
                    'Traffic_Level': result['traffic_level'],
                    'Image_Width': result['image_size']['width'],
                    'Image_Height': result['image_size']['height']
                })

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(save_path, index=False)
            print(f"\n[INFO] Detailed CSV report saved: {save_path}")

    def _create_summary_csv(self, summary, results_list, save_path):
        """Create summary CSV per image"""
        summary_data = []

        for result in results_list:
            row = {
                'Image': result['filename'],
                'Total_Vehicles': result['total_vehicles'],
                'Traffic_Level': result['traffic_level'],
                'Traffic_Density': result['traffic_density'],
                'Inference_Time': result['inference_time'],
                'FPS': result['fps']
            }

            # Add vehicle counts
            for vehicle_type in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light']:
                row[f'{vehicle_type.title()}_Count'] = result['vehicle_counts'].get(vehicle_type, 0)

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Summary CSV report saved: {save_path}")

    def _create_text_summary(self, summary, results_list, save_path):
        """Create human-readable text summary"""
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BATCH TRAFFIC PROCESSING SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            f.write("OVERALL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Images Processed: {summary['total_images']}\n")
            f.write(f"Total Vehicles Detected: {summary['total_vehicles_detected']}\n")
            f.write(f"Average Vehicles per Image: {summary['average_vehicles_per_image']:.2f}\n")
            f.write(f"Average Inference Time: {summary['average_inference_time']:.3f}s\n")
            f.write(f"Average FPS: {summary['average_fps']:.2f}\n\n")

            f.write("VEHICLE TYPE DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            for vehicle_type, count in sorted(summary['vehicle_type_totals'].items(),
                                             key=lambda x: x[1], reverse=True):
                percentage = (count / summary['total_vehicles_detected']) * 100
                f.write(f"{vehicle_type.title():<20}: {count:>5} ({percentage:>5.1f}%)\n")

            f.write("\n")
            f.write("TRAFFIC LEVEL DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            for level, count in sorted(summary['traffic_levels'].items()):
                percentage = (count / summary['total_images']) * 100
                f.write(f"{level:<20}: {count:>5} images ({percentage:>5.1f}%)\n")

            f.write("\n")
            f.write("PER-IMAGE DETAILS\n")
            f.write("-"*70 + "\n")
            for idx, result in enumerate(results_list, 1):
                f.write(f"\n{idx}. {result['filename']}\n")
                f.write(f"   Total Vehicles: {result['total_vehicles']}\n")
                f.write(f"   Traffic Level: {result['traffic_level']}\n")
                f.write(f"   Vehicles: ")
                vehicle_list = [f"{v_type}({count})"
                               for v_type, count in result['vehicle_counts'].items()
                               if count > 0]
                f.write(", ".join(vehicle_list) + "\n")

            f.write("\n" + "="*70 + "\n")

        print(f"[INFO] Text summary saved: {save_path}")

    def _print_batch_summary(self, summary, failed_images):
        """Print batch summary to console"""
        print(f"\n\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"\nProcessed Images: {summary['total_images']}")

        if failed_images:
            print(f"Failed Images: {len(failed_images)}")
            for img in failed_images:
                print(f"  - {img}")

        print(f"\nTotal Vehicles Detected: {summary['total_vehicles_detected']}")
        print(f"Average Vehicles/Image: {summary['average_vehicles_per_image']:.2f}")
        print(f"Average Processing Time: {summary['average_inference_time']:.3f}s")
        print(f"Average FPS: {summary['average_fps']:.2f}")

        print(f"\nVehicle Distribution:")
        for vehicle_type, count in sorted(summary['vehicle_type_totals'].items(),
                                         key=lambda x: x[1], reverse=True):
            print(f"  {vehicle_type.title():<15}: {count}")

        print(f"\nTraffic Levels:")
        for level, count in sorted(summary['traffic_levels'].items()):
            print(f"  {level:<15}: {count} images")

        print(f"\n{'='*70}\n")


def main():
    """Main function for batch processing"""
    parser = argparse.ArgumentParser(
        description='Batch Traffic Processing - CVIP Assignment-2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_processor.py --input input_images/
  python batch_processor.py --input traffic_data/ --output results/ --model yolov8m.pt
  python batch_processor.py --input images/ --conf 0.3 --no-reports
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing traffic images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_images',
        help='Output directory for results (default: output_images)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model variant'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip individual analysis reports (faster processing)'
    )

    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(
        model_name=args.model,
        confidence_threshold=args.conf
    )

    # Process directory
    summary = processor.process_directory(
        args.input,
        args.output,
        generate_reports=not args.no_reports
    )

    if summary:
        print("\n[SUCCESS] Batch processing completed successfully!")
        print(f"[INFO] Check '{args.output}' for annotated images")
        print(f"[INFO] Check 'reports' for detailed reports")
    else:
        print("\n[ERROR] Batch processing failed!")


if __name__ == "__main__":
    main()
