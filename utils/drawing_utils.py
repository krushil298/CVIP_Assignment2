"""
Drawing Utilities for Traffic Monitoring System
Provides helper functions for visualization
"""

import cv2
import numpy as np


class DrawingUtils:
    """
    Utility class for drawing visualizations on images
    """

    @staticmethod
    def draw_bounding_box(image, bbox, label, color=(0, 255, 0), thickness=2):
        """
        Draw bounding box with label

        Args:
            image: Image array
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Label text
            color: Box color (BGR)
            thickness: Line thickness
        """
        x1, y1, x2, y2 = bbox

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

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

        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1_label - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return image

    @staticmethod
    def draw_info_panel(image, info_dict, position='top-left'):
        """
        Draw information panel on image

        Args:
            image: Image array
            info_dict: Dictionary of info to display
            position: Panel position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        """
        height, width = image.shape[:2]
        panel_width = 350
        panel_height = 50 + (len(info_dict) * 30)

        # Determine position
        if position == 'top-left':
            x, y = 10, 10
        elif position == 'top-right':
            x, y = width - panel_width - 10, 10
        elif position == 'bottom-left':
            x, y = 10, height - panel_height - 10
        else:  # bottom-right
            x, y = width - panel_width - 10, height - panel_height - 10

        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Draw info text
        y_offset = y + 30
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(
                image,
                text,
                (x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 30

        return image

    @staticmethod
    def create_color_map(num_classes):
        """
        Generate distinct colors for different classes

        Args:
            num_classes: Number of classes

        Returns:
            dict: Class index to color mapping
        """
        np.random.seed(42)  # For consistency
        colors = {}
        for i in range(num_classes):
            colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))
        return colors

    @staticmethod
    def add_watermark(image, text="Traffic Monitoring System", position='bottom-right'):
        """
        Add watermark to image

        Args:
            image: Image array
            text: Watermark text
            position: Position ('bottom-right', 'bottom-left')
        """
        height, width = image.shape[:2]

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        if position == 'bottom-right':
            x = width - text_width - 20
        else:
            x = 20

        y = height - 20

        # Add semi-transparent background
        cv2.rectangle(
            image,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + 5),
            (0, 0, 0),
            -1
        )

        # Add text
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return image

    @staticmethod
    def draw_density_heatmap(image, detection_centers, radius=50, alpha=0.4):
        """
        Create density heatmap overlay

        Args:
            image: Image array
            detection_centers: List of (x, y) center coordinates
            radius: Heatmap radius around each point
            alpha: Overlay transparency

        Returns:
            Image with heatmap overlay
        """
        if not detection_centers:
            return image

        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Create heatmap
        for center_x, center_y in detection_centers:
            cv2.circle(heatmap, (center_x, center_y), radius, 1.0, -1)

        # Normalize and apply colormap
        heatmap = cv2.GaussianBlur(heatmap, (radius*2+1, radius*2+1), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay on original image
        result = cv2.addWeighted(image, 1-alpha, heatmap_color, alpha, 0)

        return result
