import os

import cv2
import numpy as np
import requests
from ultralytics.engine.results import Results
from sklearn.metrics import mean_squared_error


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


def evaluate_custom_model(y_test, y_pred, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Evaluate custom object detector with metrics: MSE, mean IoU, precision, recall, and mAP.

    Assumes one bounding box per image.

    :param y_test: Ground truth boxes in YOLO format (N, 4)
    :param y_pred: Predicted values of boxes
    :param iou_thresholds: IoU thresholds for mAP computation (default: 0.5 to 0.95)
    """
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        box1_x1, box1_y1 = x1 - w1 / 2, y1 - h1 / 2
        box1_x2, box1_y2 = x1 + w1 / 2, y1 + h1 / 2
        box2_x1, box2_y1 = x2 - w2 / 2, y2 - h2 / 2
        box2_x2, box2_y2 = x2 + w2 / 2, y2 + h2 / 2

        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0

    # Compute IoUs
    ious = [compute_iou(pred, true) for pred, true in zip(y_pred, y_test)]
    mean_iou = np.mean(ious)
    print(f"Mean IoU: {mean_iou:.4f}")

    # Precision/mAP at multiple thresholds
    precisions = []

    for thresh in iou_thresholds:
        TP = sum(iou >= thresh for iou in ious)
        FP = sum(iou < thresh for iou in ious)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        precisions.append(precision)

    print(f"Precision @IoU=0.50: {precisions[0]:.4f}")
    print(f"mAP@[0.50:0.95]: {np.mean(precisions):.4f}")


