"""
CVIP Assignment-2: Traffic Monitoring System - Demo Script
Easy-to-use demonstration interface

This script provides a simple way to test the traffic monitoring system
with various examples and options.

Author: Student Name
Date: 2025
"""

import sys
from pathlib import Path
from traffic_detector import TrafficDetector
from traffic_analyzer import TrafficAnalyzer
from batch_processor import BatchProcessor


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║           TRAFFIC MONITORING SYSTEM - YOLO                        ║
    ║           CVIP Assignment-2                                       ║
    ║                                                                   ║
    ║           Vehicle Detection & Analysis System                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print main menu"""
    menu = """
    ┌───────────────────────────────────────────────────────────────┐
    │                        MAIN MENU                              │
    ├───────────────────────────────────────────────────────────────┤
    │                                                               │
    │  1. Detect vehicles in a single image                        │
    │  2. Analyze image with detailed reports                      │
    │  3. Process multiple images (batch mode)                     │
    │  4. View system information                                  │
    │  5. Quick demo with sample image                             │
    │  6. Exit                                                     │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘
    """
    print(menu)


def single_image_detection():
    """Detect vehicles in a single image"""
    print("\n" + "="*70)
    print("SINGLE IMAGE DETECTION")
    print("="*70)

    image_path = input("\nEnter image path: ").strip()

    if not Path(image_path).exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        return

    # Model selection
    print("\nSelect YOLO model:")
    print("  1. YOLOv8n (Nano - Fastest)")
    print("  2. YOLOv8s (Small)")
    print("  3. YOLOv8m (Medium - Balanced)")
    print("  4. YOLOv8l (Large)")
    print("  5. YOLOv8x (XLarge - Most Accurate)")

    model_choice = input("\nEnter choice (1-5, default=1): ").strip() or "1"

    models = {
        '1': 'yolov8n.pt',
        '2': 'yolov8s.pt',
        '3': 'yolov8m.pt',
        '4': 'yolov8l.pt',
        '5': 'yolov8x.pt'
    }

    model_name = models.get(model_choice, 'yolov8n.pt')

    # Confidence threshold
    conf_input = input("\nConfidence threshold (0.1-0.9, default=0.25): ").strip()
    try:
        confidence = float(conf_input) if conf_input else 0.25
        confidence = max(0.1, min(0.9, confidence))
    except:
        confidence = 0.25

    # Initialize detector
    print("\n" + "-"*70)
    detector = TrafficDetector(model_name=model_name, confidence_threshold=confidence)

    # Perform detection
    output_dir = Path('output_images/annotated')
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem
    save_path = output_dir / f"{image_name}_detected.jpg"

    results = detector.detect_traffic(
        image_path,
        save_path=save_path,
        show_result=True
    )

    if results:
        print(f"\n[SUCCESS] Detection complete!")
        print(f"[INFO] Annotated image saved to: {save_path}")

        # Ask if user wants detailed analysis
        analyze = input("\nGenerate detailed analysis? (y/n, default=y): ").strip().lower()
        if analyze != 'n':
            analyzer = TrafficAnalyzer()
            analyzer.analyze_single_image(results, 'output_images/analysis')
            print(f"\n[INFO] Analysis saved to: output_images/analysis/")


def analyze_with_reports():
    """Analyze image with detailed reports"""
    print("\n" + "="*70)
    print("IMAGE ANALYSIS WITH REPORTS")
    print("="*70)

    image_path = input("\nEnter image path: ").strip()

    if not Path(image_path).exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        return

    print("\nInitializing system...")
    detector = TrafficDetector()
    analyzer = TrafficAnalyzer()

    # Detect
    output_dir = Path('output_images/annotated')
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem
    save_path = output_dir / f"{image_name}_detected.jpg"

    print("\nPerforming detection...")
    results = detector.detect_traffic(
        image_path,
        save_path=save_path,
        show_result=False
    )

    if results:
        print("\nGenerating detailed analysis...")
        analyzer.analyze_single_image(results, 'output_images/analysis')

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nGenerated Files:")
        print(f"  • Annotated Image: {save_path}")
        print(f"  • Bar Chart: output_images/analysis/{image_name}_bar_chart.png")
        print(f"  • Pie Chart: output_images/analysis/{image_name}_pie_chart.png")
        print(f"  • Summary: output_images/analysis/{image_name}_summary.png")
        print(f"  • CSV Report: output_images/analysis/{image_name}_report.csv")

        # Display result
        show = input("\nDisplay annotated image? (y/n, default=y): ").strip().lower()
        if show != 'n':
            import cv2
            img = cv2.imread(str(save_path))
            cv2.imshow('Detection Results - Press any key to close', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def batch_processing():
    """Process multiple images"""
    print("\n" + "="*70)
    print("BATCH PROCESSING")
    print("="*70)

    input_dir = input("\nEnter directory containing images: ").strip()

    if not Path(input_dir).exists():
        print(f"\n[ERROR] Directory not found: {input_dir}")
        return

    # Check for images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"\n[ERROR] No images found in {input_dir}")
        return

    print(f"\n[INFO] Found {len(image_files)} images")

    # Options
    output_dir = input("\nOutput directory (default='output_images'): ").strip() or 'output_images'

    reports = input("Generate individual analysis reports? (y/n, default=y): ").strip().lower()
    generate_reports = reports != 'n'

    # Process
    print("\nInitializing batch processor...")
    processor = BatchProcessor()

    print("\nProcessing images...")
    summary = processor.process_directory(
        input_dir,
        output_dir,
        generate_reports=generate_reports
    )

    if summary:
        print("\n[SUCCESS] Batch processing complete!")
        print(f"\nCheck the following locations for results:")
        print(f"  • Annotated Images: {output_dir}/annotated/")
        print(f"  • Analysis: {output_dir}/analysis/")
        print(f"  • Reports: reports/")


def system_info():
    """Display system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)

    print("\nTraffic Monitoring System - YOLO")
    print("CVIP Assignment-2")
    print("\nCapabilities:")
    print("  • Real-time vehicle detection using YOLOv8")
    print("  • Detects: Cars, Trucks, Buses, Motorcycles, Bicycles, Traffic Lights")
    print("  • Traffic density analysis")
    print("  • Batch processing support")
    print("  • Comprehensive reporting (CSV, JSON, visualizations)")
    print("\nAvailable Models:")
    print("  • YOLOv8n - Nano (Fastest, ~3-5ms)")
    print("  • YOLOv8s - Small (~5-8ms)")
    print("  • YOLOv8m - Medium (Balanced, ~10-15ms)")
    print("  • YOLOv8l - Large (~15-25ms)")
    print("  • YOLOv8x - XLarge (Most Accurate, ~25-40ms)")
    print("\nOutput Formats:")
    print("  • Annotated images with bounding boxes")
    print("  • Statistical charts (bar, pie)")
    print("  • CSV reports")
    print("  • JSON reports")
    print("  • Text summaries")

    print("\n" + "="*70)


def quick_demo():
    """Quick demonstration"""
    print("\n" + "="*70)
    print("QUICK DEMO")
    print("="*70)

    # Check for sample images
    sample_dir = Path('input_images')

    if sample_dir.exists():
        sample_images = list(sample_dir.glob('*.jpg')) + list(sample_dir.glob('*.png'))

        if sample_images:
            print(f"\n[INFO] Found {len(sample_images)} sample images")
            print("\nAvailable samples:")
            for idx, img in enumerate(sample_images[:5], 1):
                print(f"  {idx}. {img.name}")

            choice = input("\nSelect image (1-5) or press Enter for first: ").strip()
            try:
                idx = int(choice) - 1 if choice else 0
                sample_image = sample_images[idx]
            except:
                sample_image = sample_images[0]

            print(f"\n[INFO] Using: {sample_image.name}")
        else:
            print("\n[WARNING] No sample images found in input_images/")
            print("Please add some sample images or use option 1 with your own image.")
            return
    else:
        print("\n[WARNING] input_images/ directory not found")
        print("Please create it and add sample images, or use option 1.")
        return

    # Run demo
    print("\nRunning detection...")
    detector = TrafficDetector()
    analyzer = TrafficAnalyzer()

    output_dir = Path('output_images/annotated')
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"demo_{sample_image.name}"

    results = detector.detect_traffic(
        str(sample_image),
        save_path=save_path,
        show_result=True
    )

    if results:
        print("\nGenerating analysis...")
        analyzer.analyze_single_image(results, 'output_images/analysis')
        print("\n[SUCCESS] Demo complete!")
        print(f"[INFO] Results saved to: output_images/")


def main():
    """Main demo function"""
    print_banner()

    while True:
        print_menu()

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == '1':
            single_image_detection()
        elif choice == '2':
            analyze_with_reports()
        elif choice == '3':
            batch_processing()
        elif choice == '4':
            system_info()
        elif choice == '5':
            quick_demo()
        elif choice == '6':
            print("\n" + "="*70)
            print("Thank you for using Traffic Monitoring System!")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("\n[ERROR] Invalid choice. Please enter 1-6.")

        input("\n\nPress Enter to continue...")
        print("\n" * 2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Program interrupted by user.")
        print("Goodbye!\n")
        sys.exit(0)
