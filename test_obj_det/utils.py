import os

import cv2
import numpy as np
from ultralytics.engine.results import Results


def load_data(base_dir, split="train", img_size=(128, 128)):
    """
    Load images and YOLO-format labels from a split dataset (train or val).

    :param base_dir: Root dataset directory (contains 'images/train', 'labels/train', etc.).
    :param split: Dataset split to load ("train" or "val").
    :param img_size: Tuple (width, height) to resize images to.
    :return: Tuple of (images, labels, original_shapes)
             - images: resized images as NumPy array (N, H, W, C)
             - labels: bounding boxes in YOLO format (N, 4), one per image
             - original_shapes: list of (h, w) tuples for each image
    """
    image_dir = os.path.join(base_dir, "images", split)
    labels_dir = os.path.join(base_dir, "labels", split)

    images = []
    labels = []
    original_shapes = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        original_shapes.append((h, w))

        # Resize
        img_resized = cv2.resize(img, img_size)
        images.append(img_resized)

        # Load label (only first object, YOLO format)
        with open(label_path, "r") as f:
            line = f.readline()
            if not line:
                continue
            label_data = list(map(float, line.strip().split()))
            labels.append(label_data[1:5])  # Exclude class_id

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.float32)

    return images, labels, original_shapes


def load_dataset():
    pass


def draw_bbox(img, bbox, color=(255, 255, 0), thickness=1, class_name="Object", orig_shape=None):
    """
    Unified function to draw bounding boxes.
    Supports:
      - Custom model: bbox as [x, y, w, h]
      - YOLOv8: bbox as list of `Results` objects

    :param img: Image (NumPy array).
    :param bbox: Bounding box array or list of YOLOv8 Results.
    :param color: Color of the box.
    :param thickness: Thickness of rectangle.
    :param class_name: Label for custom model.
    :param orig_shape: Resize back to original shape if provided.
    :return: Image with bounding box(es) drawn.
    """
    # Resize image back to original if shape provided
    if orig_shape:
        img = cv2.resize(img, (orig_shape[1], orig_shape[0]))
        height, width = orig_shape
    else:
        height, width = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check if YOLOv8 Results list
    if isinstance(bbox, list) and all(isinstance(b, Results) for b in bbox):
        for b in bbox:
            for i in range(len(b.boxes)):
                box = b.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                conf = float(b.boxes.conf[i])
                cls_id = int(b.boxes.cls[i])
                label = f"{b.names[cls_id]} {conf:.2f}"

                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                img = cv2.putText(img, label, (x1, max(10, y1 - 10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        return img

    # Otherwise assume [x, y, w, h] format (custom model)
    bbox = np.array(bbox).flatten()
    x, y, w, h = bbox
    x = x * width
    y = y * height
    w = w * width
    h = h * height

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.putText(img, class_name, (x1, max(10, y1 - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return img

