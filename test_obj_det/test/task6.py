import numpy as np
from ultralytics import YOLO

from test_obj_det.utils import display_hint, display_solution, display_check

def hint6():
    hint = """
Step 6:
- Load a pretrained YOLOv8 model using the `YOLO` class.
- Set a confidence threshold for predictions (for example, 0.5) to filter low-confidence results.
"""
    display_hint(hint)


def solution6():
    code = """
yolo_model = YOLO('https://codefinity-content-media-v2.s3.eu-west-1.amazonaws.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/object_detection_project/models/yolo_trained.pt')
yolo_results = yolo_model.predict([img for img in val_images[:9]], conf=0.3)
"""
    display_solution(code)


def check6(yolo_model):
    try:
        # Check if the YOLO model is loaded properly
        if not isinstance(yolo_model, YOLO):
            display_check(False, "YOLO model is not loaded correctly.")
            return

        display_check(True, "YOLOv8 predictions and visualization are correct. Next key: LDB67X")

    except Exception as e:
        display_check(False, f"Something went wrong: {str(e)}")
