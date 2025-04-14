from test_obj_det.utils import display_hint, display_solution, display_check


def hint8():
    hint = """
Step 8:
- Validate the YOLOv8 model by using the correct data.yaml file.
- Use the `yolo_model.val()` function to evaluate the model on the validation dataset.
- Extract metrics like precision, recall, mAP50, and mAP from the results.
- These metrics help in evaluating how well the YOLOv8 model performs on your dataset.
"""
    display_hint(hint)


def solution8():
    code = """
# === 8. Evaluate YOLOv8 model ===
# Validate the YOLO model using the correct data.yaml file
metrics = yolo_model.val(data="https://codefinity-content-media-v2.s3.eu-west-1.amazonaws.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/object_detection_project/data/data.yaml")

# Extract and print performance metrics
precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map = metrics.box.map

print("\\nYOLO results:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP50: {map50:.4f}")
print(f"mAP50-95: {map:.4f}")
"""
    display_solution(code)


def check8(metrics):
    try:
        # Extract required metrics
        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map = metrics.box.map

        # Check if all metrics are present and are floats
        if all(isinstance(m, float) for m in [precision, recall, map50, map]):
            display_check(True, "YOLO metrics extracted successfully.")
        else:
            display_check(False, "One or more metrics are missing or invalid.")

    except Exception as e:
        display_check(False, f"Something went wrong: {str(e)}")

