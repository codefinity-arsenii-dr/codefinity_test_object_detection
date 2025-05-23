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
metrics = yolo_model.val(data="/path/to/dataset/directory/data.yaml")

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
            display_check(True, "YOLO metrics extracted successfully. Here's your next key part: A777CS")
        else:
            display_check(False, "One or more metrics are missing or invalid.")

    except Exception as e:
        display_check(False, f"Something went wrong: {str(e)}")

