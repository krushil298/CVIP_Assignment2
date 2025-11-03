# Traffic Monitoring System using YOLO

**CVIP Assignment-2: Object Detection Project**

A comprehensive traffic monitoring and vehicle detection system built using YOLOv8 (You Only Look Once) for real-time object detection in traffic images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Detection Capabilities](#detection-capabilities)
- [Output Examples](#output-examples)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This project implements an intelligent traffic monitoring system that uses state-of-the-art YOLO (You Only Look Once) object detection to identify and count vehicles in traffic images. The system provides:

- **Real-time vehicle detection** with multiple YOLO model variants
- **Comprehensive analysis** including traffic density and congestion levels
- **Batch processing** for multiple images
- **Visual reports** with charts, graphs, and annotated images
- **Statistical reports** in multiple formats (CSV, JSON, text)

### Why YOLO?

YOLO (You Only Look Once) is chosen for this project because:
- ⚡ **Fast**: Can process images in milliseconds, suitable for real-time applications
- 🎯 **Accurate**: High detection accuracy with minimal false positives
- 🔄 **Single Pass**: Processes entire image in one forward pass through the network
- 📦 **Easy to Use**: Pre-trained models available, simple API

---

## ✨ Features

### Core Capabilities

- ✅ **Multi-Vehicle Detection**
  - Cars, Trucks, Buses, Motorcycles, Bicycles
  - Traffic lights and Stop signs

- ✅ **Traffic Analysis**
  - Vehicle counting by category
  - Traffic density calculation
  - Traffic level classification (Light/Moderate/Heavy/Congested)

- ✅ **Batch Processing**
  - Process multiple images at once
  - Aggregate statistics
  - Comparative analysis

- ✅ **Visualization**
  - Annotated images with bounding boxes
  - Bar charts showing vehicle distribution
  - Pie charts for percentage breakdown
  - Statistical summary dashboards

- ✅ **Reporting**
  - CSV reports with detailed detection data
  - JSON exports for programmatic access
  - Human-readable text summaries
  - Performance metrics (inference time, FPS)

### Advanced Features

- 🔧 Configurable confidence thresholds
- 🎨 Color-coded bounding boxes by vehicle type
- 📊 Traffic density heatmaps
- 🚀 Multiple YOLO model support (nano to xlarge)
- 💾 Export results in multiple formats

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-compatible GPU (optional, for faster processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/CVIP_Assignment2_Traffic_Monitoring.git
cd CVIP_Assignment2_Traffic_Monitoring
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch & TorchVision
- Ultralytics YOLOv8
- OpenCV
- Matplotlib & Seaborn
- Pandas, NumPy
- And other required libraries

### Step 3: Verify Installation

```bash
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

---

## 📁 Project Structure

```
CVIP_Assignment2_Traffic_Monitoring/
│
├── traffic_detector.py          # Main detection module
├── traffic_analyzer.py          # Analysis and visualization
├── batch_processor.py           # Batch processing script
├── demo.py                      # Interactive demo
│
├── utils/
│   ├── __init__.py
│   ├── drawing_utils.py        # Drawing utilities
│   └── report_generator.py     # Report generation
│
├── input_images/               # Place your test images here
├── output_images/
│   ├── annotated/             # Detected images with boxes
│   └── analysis/              # Charts and visualizations
│
├── reports/                   # Generated reports (CSV, JSON, txt)
├── models/                    # Downloaded YOLO models (auto-created)
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 💻 Usage

### Method 1: Interactive Demo (Recommended for Beginners)

The easiest way to use the system:

```bash
python demo.py
```

This launches an interactive menu with options for:
1. Single image detection
2. Detailed analysis with reports
3. Batch processing
4. System information
5. Quick demo with samples

### Method 2: Command Line - Single Image

Detect vehicles in a single image:

```bash
python traffic_detector.py --image input_images/traffic.jpg
```

With custom parameters:

```bash
python traffic_detector.py \
    --image input_images/traffic.jpg \
    --model yolov8m.pt \
    --conf 0.3 \
    --output results/ \
    --save-report
```

**Parameters:**
- `--image`: Path to input image (required)
- `--model`: YOLO model variant (default: yolov8n.pt)
  - `yolov8n.pt` - Nano (fastest)
  - `yolov8s.pt` - Small
  - `yolov8m.pt` - Medium (balanced)
  - `yolov8l.pt` - Large
  - `yolov8x.pt` - XLarge (most accurate)
- `--conf`: Confidence threshold 0-1 (default: 0.25)
- `--output`: Output directory (default: output_images/annotated)
- `--no-show`: Don't display result window
- `--save-report`: Save JSON report

### Method 3: Analysis with Visualizations

Generate detailed analysis with charts:

```bash
python traffic_analyzer.py --image input_images/traffic.jpg
```

This creates:
- Annotated image with detections
- Bar chart of vehicle counts
- Pie chart of distribution
- Statistical summary dashboard
- CSV report with all detections

### Method 4: Batch Processing

Process multiple images at once:

```bash
python batch_processor.py --input input_images/
```

With options:

```bash
python batch_processor.py \
    --input input_images/ \
    --output results/ \
    --model yolov8m.pt \
    --conf 0.3 \
    --no-reports  # Skip individual analysis (faster)
```

This generates:
- Individual detections for each image
- Aggregate statistics across all images
- Comparison charts
- Summary reports (CSV, text)

---

## 🚗 Detection Capabilities

### Detected Vehicle Types

| Vehicle Type | COCO Class ID | Detection Accuracy |
|-------------|---------------|-------------------|
| Car | 2 | ⭐⭐⭐⭐⭐ Excellent |
| Motorcycle | 3 | ⭐⭐⭐⭐ Very Good |
| Bicycle | 1 | ⭐⭐⭐⭐ Very Good |
| Bus | 5 | ⭐⭐⭐⭐⭐ Excellent |
| Truck | 7 | ⭐⭐⭐⭐ Very Good |
| Traffic Light | 9 | ⭐⭐⭐ Good |
| Stop Sign | 11 | ⭐⭐⭐ Good |

### Traffic Level Classification

The system automatically classifies traffic density:

- **LIGHT** - Density < 2 vehicles per 10k pixels
- **MODERATE** - Density 2-5 vehicles per 10k pixels
- **HEAVY** - Density 5-10 vehicles per 10k pixels
- **CONGESTED** - Density > 10 vehicles per 10k pixels

---

## 📊 Output Examples

### 1. Annotated Images

Images with color-coded bounding boxes and labels showing:
- Vehicle type
- Confidence score
- Summary overlay with counts

### 2. Statistical Charts

**Bar Chart** - Vehicle counts by category
```
Cars: ████████████ 12
Trucks: ████ 4
Buses: ██ 2
```

**Pie Chart** - Percentage distribution
```
Cars: 60%
Trucks: 25%
Buses: 15%
```

### 3. CSV Reports

```csv
Detection_ID,Vehicle_Type,Confidence,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2
1,car,0.92,145,230,342,456
2,truck,0.88,450,180,680,420
...
```

### 4. Performance Metrics

```
Inference Time: 0.023s
FPS: 43.48
Total Vehicles: 18
Traffic Level: MODERATE
```

---

## 🔧 Technical Details

### YOLO Architecture

This project uses **YOLOv8** from Ultralytics, the latest version of YOLO with:
- Improved accuracy over previous versions
- Faster inference times
- Better small object detection
- Pre-trained on COCO dataset (80 classes)

### Model Comparison

| Model | Size | Speed | mAP | Best For |
|-------|------|-------|-----|----------|
| YOLOv8n | 3.2M | ⚡⚡⚡ 1.47ms | 37.3 | Real-time, resource-constrained |
| YOLOv8s | 11.2M | ⚡⚡ 2.66ms | 44.9 | Balanced |
| YOLOv8m | 25.9M | ⚡ 5.86ms | 50.2 | Production |
| YOLOv8l | 43.7M | 8.05ms | 52.9 | High accuracy |
| YOLOv8x | 68.2M | 12.41ms | 53.9 | Maximum accuracy |

*Speed benchmarked on NVIDIA T4 GPU*

### Algorithm Flow

```
Input Image
    ↓
Preprocessing (resize, normalize)
    ↓
YOLO Forward Pass
    ↓
Non-Max Suppression
    ↓
Filter by Confidence Threshold
    ↓
Filter Vehicle Classes
    ↓
Calculate Statistics
    ↓
Generate Visualizations
    ↓
Output Results
```

### Traffic Density Calculation

```python
density = (vehicle_count / image_area) * 10000
```

Where:
- `vehicle_count` = Total detected vehicles
- `image_area` = width × height in pixels
- Result normalized per 10,000 pixels

---

## 📖 Examples

### Example 1: Basic Detection

```python
from traffic_detector import TrafficDetector

# Initialize detector
detector = TrafficDetector(model_name='yolov8n.pt', confidence_threshold=0.25)

# Detect vehicles
results = detector.detect_traffic(
    'input_images/traffic.jpg',
    save_path='output.jpg',
    show_result=True
)

# Access results
print(f"Total vehicles: {results['total_vehicles']}")
print(f"Traffic level: {results['traffic_level']}")
print(f"Vehicle counts: {results['vehicle_counts']}")
```

### Example 2: Batch Analysis

```python
from batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor(model_name='yolov8m.pt')

# Process directory
summary = processor.process_directory(
    'input_images/',
    'output_images/',
    generate_reports=True
)

# View summary
print(f"Processed {summary['total_images']} images")
print(f"Detected {summary['total_vehicles_detected']} vehicles")
```

---

## 🎓 Assignment Deliverables

This project fulfills CVIP Assignment-2 requirements:

✅ **Implementation**
- YOLO-based object detection system
- Working code with proper documentation
- Multiple detection modes

✅ **Results**
- Annotated images with bounding boxes
- Detection statistics and analysis
- Performance metrics

✅ **Documentation**
- Comprehensive README
- Code comments and docstrings
- Usage examples

✅ **Visualization**
- Charts and graphs
- Statistical summaries
- Comparison analysis

---

## 🛠️ Troubleshooting

### Common Issues

**1. "Model not found" error**
```bash
# First run will auto-download model
python traffic_detector.py --image test.jpg
```

**2. CUDA out of memory**
```bash
# Use smaller model
python traffic_detector.py --image test.jpg --model yolov8n.pt
```

**3. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**4. Slow performance**
- Use GPU if available
- Try smaller model (yolov8n.pt)
- Reduce image resolution

---

## 📝 Future Enhancements

Potential improvements for this project:

- [ ] Real-time video processing
- [ ] Webcam live detection
- [ ] Vehicle tracking across frames
- [ ] Speed estimation
- [ ] License plate detection
- [ ] Web-based dashboard (Streamlit/Gradio)
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] Database storage for results
- [ ] API endpoint for integration

---

## 👥 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **COCO Dataset** for pre-trained weights
- **OpenCV** community for computer vision tools
- **PyTorch** team for deep learning framework

---

## 📧 Contact

For questions or support:

- **Student Name**: Your Name
- **Email**: your.email@example.com
- **Course**: Computer Vision & Image Processing (CVIP)
- **Assignment**: Assignment-2

---

## 🌟 Star This Repository

If you found this project helpful, please give it a ⭐!

---

**Made with ❤️ for CVIP Assignment-2**
