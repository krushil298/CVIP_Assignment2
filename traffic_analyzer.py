"""
CVIP Assignment-2: Traffic Analysis Module
Provides detailed traffic analysis and visualization

This module analyzes detection results and generates:
- Statistical reports
- Visual charts (bar charts, pie charts)
- Traffic density heatmaps
- Comparative analysis

Author: Student Name
Date: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse
from traffic_detector import TrafficDetector


class TrafficAnalyzer:
    """
    Traffic Analysis and Visualization Class
    Analyzes detection results and creates visual reports
    """

    def __init__(self):
        """Initialize Traffic Analyzer"""
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def analyze_single_image(self, detection_results, output_dir='output_images/analysis'):
        """
        Analyze single image detection results and create visualizations

        Args:
            detection_results (dict): Results from TrafficDetector
            output_dir (str): Directory to save analysis outputs
        """
        if not detection_results:
            print("[ERROR] No detection results provided")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = detection_results['filename']
        basename = Path(filename).stem

        print(f"\n[INFO] Analyzing: {filename}")

        # Create visualizations
        self._create_bar_chart(detection_results, output_path / f"{basename}_bar_chart.png")
        self._create_pie_chart(detection_results, output_path / f"{basename}_pie_chart.png")
        self._create_statistics_summary(detection_results, output_path / f"{basename}_summary.png")

        # Generate CSV report
        self._generate_csv_report(detection_results, output_path / f"{basename}_report.csv")

        print(f"[INFO] Analysis complete! Results saved to: {output_path}")

    def _create_bar_chart(self, results, save_path):
        """Create bar chart of vehicle counts"""
        vehicle_counts = results['vehicle_counts']

        # Filter out zero counts
        counts_filtered = {k: v for k, v in vehicle_counts.items() if v > 0}

        if not counts_filtered:
            print("[WARNING] No vehicles detected for bar chart")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        vehicles = list(counts_filtered.keys())
        counts = list(counts_filtered.values())

        # Define colors
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

        bars = ax.bar(vehicles, counts, color=colors[:len(vehicles)], alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('Vehicle Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.set_title(f'Vehicle Distribution - {results["filename"]}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(counts) * 1.2)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        # Add grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Bar chart saved: {save_path}")

    def _create_pie_chart(self, results, save_path):
        """Create pie chart of vehicle distribution"""
        vehicle_counts = results['vehicle_counts']

        # Filter out zero counts
        counts_filtered = {k: v for k, v in vehicle_counts.items() if v > 0}

        if not counts_filtered:
            print("[WARNING] No vehicles detected for pie chart")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        vehicles = list(counts_filtered.keys())
        counts = list(counts_filtered.values())

        # Define colors
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=vehicles,
            colors=colors[:len(vehicles)],
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(vehicles),
            shadow=True
        )

        # Enhance text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')

        ax.set_title(f'Vehicle Distribution (%) - {results["filename"]}',
                    fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Pie chart saved: {save_path}")

    def _create_statistics_summary(self, results, save_path):
        """Create visual statistics summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Traffic Analysis Summary - {results["filename"]}',
                    fontsize=18, fontweight='bold', y=0.98)

        # 1. Key Metrics
        ax1.axis('off')
        metrics_text = f"""
        KEY METRICS

        Total Vehicles: {results['total_vehicles']}

        Image Size: {results['image_size']['width']} x {results['image_size']['height']}

        Traffic Level: {results['traffic_level']}

        Traffic Density: {results['traffic_density']:.4f}

        Inference Time: {results['inference_time']:.3f}s

        FPS: {results['fps']:.2f}
        """
        ax1.text(0.1, 0.9, metrics_text, fontsize=13, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Vehicle Count Table
        ax2.axis('off')
        vehicle_counts = results['vehicle_counts']
        counts_filtered = {k.title(): v for k, v in vehicle_counts.items() if v > 0}

        if counts_filtered:
            df = pd.DataFrame(list(counts_filtered.items()),
                            columns=['Vehicle Type', 'Count'])
            table = ax2.table(cellText=df.values, colLabels=df.columns,
                            cellLoc='center', loc='center',
                            colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)

            # Style header
            for i in range(len(df.columns)):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')

            # Alternate row colors
            for i in range(1, len(df) + 1):
                if i % 2 == 0:
                    for j in range(len(df.columns)):
                        table[(i, j)].set_facecolor('#ecf0f1')

            ax2.set_title('Vehicle Counts', fontsize=14, fontweight='bold', pad=10)

        # 3. Bar Chart (small version)
        counts_filtered = {k: v for k, v in vehicle_counts.items() if v > 0}
        if counts_filtered:
            vehicles = list(counts_filtered.keys())
            counts = list(counts_filtered.values())
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

            bars = ax3.bar(vehicles, counts, color=colors[:len(vehicles)], alpha=0.8)
            ax3.set_xlabel('Vehicle Type', fontweight='bold')
            ax3.set_ylabel('Count', fontweight='bold')
            ax3.set_title('Distribution', fontsize=14, fontweight='bold')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 4. Traffic Level Indicator
        ax4.axis('off')
        traffic_level = results['traffic_level']
        level_colors = {
            "LIGHT": '#2ecc71',
            "MODERATE": '#f39c12',
            "HEAVY": '#e67e22',
            "CONGESTED": '#e74c3c'
        }

        circle = plt.Circle((0.5, 0.5), 0.3, color=level_colors.get(traffic_level, 'gray'))
        ax4.add_patch(circle)
        ax4.text(0.5, 0.5, traffic_level, ha='center', va='center',
                fontsize=20, fontweight='bold', color='white')
        ax4.text(0.5, 0.1, 'TRAFFIC LEVEL', ha='center', va='center',
                fontsize=14, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Statistics summary saved: {save_path}")

    def _generate_csv_report(self, results, save_path):
        """Generate CSV report of detections"""
        detections = results['detections']

        if not detections:
            print("[WARNING] No detections to save in CSV")
            return

        # Create DataFrame
        data = []
        for idx, detection in enumerate(detections, 1):
            data.append({
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

        df = pd.DataFrame(data)

        # Add summary row
        summary_data = {
            'Detection_ID': 'SUMMARY',
            'Vehicle_Type': f"Total: {results['total_vehicles']}",
            'Confidence': f"Traffic: {results['traffic_level']}",
            'BBox_X1': f"Density: {results['traffic_density']:.4f}",
            'BBox_Y1': f"Time: {results['inference_time']:.3f}s",
            'BBox_X2': f"FPS: {results['fps']:.2f}",
            'BBox_Y2': '',
            'Center_X': '',
            'Center_Y': ''
        }
        df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)

        # Save to CSV
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"[INFO] CSV report saved: {save_path}")
        print(f"       Total detections: {len(detections)}")

    def compare_multiple_images(self, results_list, output_path='output_images/analysis/comparison.png'):
        """
        Create comparison chart for multiple images

        Args:
            results_list (list): List of detection results
            output_path (str): Path to save comparison chart
        """
        if not results_list:
            print("[ERROR] No results provided for comparison")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Multi-Image Traffic Comparison', fontsize=18, fontweight='bold')

        # Extract data
        filenames = [r['filename'] for r in results_list]
        total_vehicles = [r['total_vehicles'] for r in results_list]
        densities = [r['traffic_density'] for r in results_list]

        # Chart 1: Total Vehicles Comparison
        bars1 = ax1.bar(range(len(filenames)), total_vehicles,
                       color='#3498db', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Image', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Total Vehicles', fontweight='bold', fontsize=12)
        ax1.set_title('Total Vehicles per Image', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(filenames)))
        ax1.set_xticklabels([f"Img {i+1}" for i in range(len(filenames))], rotation=0)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # Chart 2: Traffic Density Comparison
        bars2 = ax2.bar(range(len(filenames)), densities,
                       color='#e74c3c', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Image', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Traffic Density', fontweight='bold', fontsize=12)
        ax2.set_title('Traffic Density per Image', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(filenames)))
        ax2.set_xticklabels([f"Img {i+1}" for i in range(len(filenames))], rotation=0)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Comparison chart saved: {output_path}")


def main():
    """Main function for standalone analysis"""
    parser = argparse.ArgumentParser(
        description='Traffic Analysis Tool - CVIP Assignment-2'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to traffic image to analyze'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model to use'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_images/analysis',
        help='Output directory for analysis'
    )

    args = parser.parse_args()

    # Initialize detector and analyzer
    print("\n[INFO] Initializing Traffic Analysis System...")
    detector = TrafficDetector(model_name=args.model, confidence_threshold=args.conf)
    analyzer = TrafficAnalyzer()

    # Perform detection
    results = detector.detect_traffic(args.image, show_result=False)

    if results:
        # Perform analysis
        analyzer.analyze_single_image(results, output_dir=args.output)
        print("\n[SUCCESS] Analysis complete!")
    else:
        print("\n[ERROR] Detection failed!")


if __name__ == "__main__":
    main()
