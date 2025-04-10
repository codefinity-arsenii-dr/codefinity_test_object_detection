import os
import torch
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

import warnings as w
w.simplefilter('ignore')

from data_split import split

def load_data(image_dir, labels_dir, test_size=0.2, random_state=42):
    images = []
    labels = []

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".png", ".txt"))

        if os.path.exists(label_path):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0  # Normalize

            with open(label_path, "r") as f:
                label_data = f.readline().strip().split(" ")[1:]
                labels.append(list(map(float, label_data)))

            images.append(img)

    # Convert to NumPy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def draw_bbox(img, bbox, color=(255, 255, 0), thickness=2, normalized=False):
    """
    Draw a bounding box on an image.

    :param img: Input image (NumPy array).
    :param bbox: Bounding box [x, y, width, height].
    :param color: Box color (default: yellow).
    :param thickness: Line thickness.
    :param normalized: Whether the bbox values are normalized (0-1 range).
    :return: Image with drawn bounding box.
    """
    height, width = img.shape[:2]

    # Ensure bbox is a 1D NumPy array
    bbox = np.array(bbox).flatten()

    # Extract bounding box values
    x, y, w, h = bbox[:4]

    # Convert normalized coordinates to pixel values if needed
    if normalized:
        x, y, w, h = x * width, y * height, w * width, h * height

    # Compute top-left and bottom-right coordinates
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # Ensure coordinates are within image bounds
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)

    # Draw rectangle and label
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.putText(img, "Person", (x1, max(10, y1 - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return img


def main():
    # Load YOLO model
    # Load trained YOLO model
    yolo = YOLO('models/yolo_trained.pt')

    # Load image
    img = cv2.imread('data/images/train/5.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run prediction
    results = yolo.predict(img_rgb, conf=0.25)

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates (in pixels)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Get class name (optional)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo.model.names[cls_id]

            # Draw rectangle and label
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_rgb, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display image with boxes
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()