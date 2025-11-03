# Quick Start Guide

Get started with the Traffic Monitoring System in 5 minutes!

## 🚀 Installation (2 minutes)

```bash
# 1. Navigate to project directory
cd CVIP_Assignment2_Traffic_Monitoring

# 2. Install dependencies
pip install -r requirements.txt

# That's it! The YOLO model will auto-download on first use.
```

## 📸 Get Sample Images (1 minute)

Download free traffic images:

1. Visit [Unsplash](https://unsplash.com/s/photos/traffic)
2. Download 2-3 traffic images
3. Save them to `input_images/` folder

Or use these direct links:
- [Highway Traffic](https://unsplash.com/photos/traffic-jam)
- [City Traffic](https://unsplash.com/photos/city-traffic)
- [Busy Intersection](https://unsplash.com/photos/intersection)

## 🎮 Run Your First Detection (2 minutes)

### Option 1: Interactive Demo (Easiest)

```bash
python demo.py
```

Then select option 5 for quick demo!

### Option 2: Command Line

```bash
# Single image detection
python traffic_detector.py --image input_images/your_image.jpg
```

### Option 3: With Analysis

```bash
# Full analysis with charts
python traffic_analyzer.py --image input_images/your_image.jpg
```

## 📊 View Results

Results are saved in:
- `output_images/annotated/` - Images with bounding boxes
- `output_images/analysis/` - Charts and graphs
- `reports/` - CSV and text reports

## 🎯 Common Use Cases

### Detect in Single Image
```bash
python traffic_detector.py --image input_images/traffic.jpg
```

### Process Multiple Images
```bash
python batch_processor.py --input input_images/
```

### Custom Confidence Threshold
```bash
python traffic_detector.py --image traffic.jpg --conf 0.3
```

### Use Different Model (More Accurate)
```bash
python traffic_detector.py --image traffic.jpg --model yolov8m.pt
```

## 💡 Tips

1. **First run is slow** - YOLO model downloads automatically (~6MB)
2. **Better accuracy** - Use `--model yolov8m.pt` (slower but more accurate)
3. **Faster processing** - Use `--model yolov8n.pt` (default, fastest)
4. **Adjust sensitivity** - Use `--conf 0.3` for more detections, `--conf 0.5` for fewer

## 🐛 Troubleshooting

**Problem: "No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**Problem: "CUDA out of memory"**
```bash
# Use smaller model
python traffic_detector.py --image traffic.jpg --model yolov8n.pt
```

**Problem: "Image not found"**
```bash
# Check file path is correct
ls input_images/
```

## 📚 What's Next?

1. ✅ Try different traffic images
2. ✅ Experiment with confidence thresholds
3. ✅ Compare different YOLO models
4. ✅ Process batch of images
5. ✅ Generate comprehensive reports
6. ✅ Check the full [README.md](README.md) for advanced features

## 🎓 For Your Assignment

Make sure to:
- [x] Run detection on at least 5 traffic images
- [x] Generate analysis reports with charts
- [x] Include annotated output images
- [x] Document your results
- [x] Compare different model performance

## 📞 Need Help?

- Read the detailed [README.md](README.md)
- Check [GITHUB_SETUP.md](GITHUB_SETUP.md) for GitHub instructions
- Review the code comments in Python files

---

**Happy Detecting! 🚗🚙🚚**
