# Input Images Directory

Place your traffic images here for processing.

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Getting Sample Images

You can find free traffic images from:

1. **Unsplash** - https://unsplash.com/s/photos/traffic
2. **Pexels** - https://www.pexels.com/search/traffic/
3. **Pixabay** - https://pixabay.com/images/search/traffic/
4. **Google Images** (with usage rights filter)

## Recommended Image Properties

- **Resolution**: 640x640 or higher
- **Format**: JPEG or PNG
- **Content**: Clear view of vehicles
- **Lighting**: Good lighting conditions for best results

## Example File Names

```
traffic_highway_1.jpg
city_traffic_busy.png
intersection_morning.jpg
parking_lot.jpg
```

## Quick Test

After adding images, test with:

```bash
# Single image
python traffic_detector.py --image input_images/your_image.jpg

# All images in directory
python batch_processor.py --input input_images/
```

## Tips for Best Results

✅ **DO:**
- Use clear, high-resolution images
- Ensure good lighting
- Include multiple vehicles for interesting analysis
- Try different traffic scenarios (light/heavy)

❌ **DON'T:**
- Use blurry or low-quality images
- Use images with extreme angles
- Use images with heavy occlusion
- Use copyrighted images without permission
