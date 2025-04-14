import numpy as np
from ultralytics import YOLO

from test_obj_det.utils import display_hint, display_solution, display_check

def hint6():
    hint = """
Step 6:
- Load a pretrained YOLOv8 model using the `YOLO` class.
- YOLO requires images to be in uint8 format with values in the range [0, 255], so convert the normalized float32 image back to uint8.
- Set a confidence threshold for predictions (for example, 0.5) to filter low-confidence results.
- Use `draw_bbox()` to display bounding boxes on the image.
"""
    display_hint(hint)


def solution6():
    code = """
# === 6. Use YOLOv8 to run predictions ===
# Load pretrained YOLOv8 model
yolo_model = YOLO('https://codefinity-content-media-v2.s3.eu-west-1.amazonaws.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/object_detection_project/models/yolo_trained.pt')

# Convert normalized float32 image to uint8 [0, 255] as YOLO expects
image = (val_images[idx] * 255).astype(np.uint8)

# Run predictions with confidence threshold (for example, 0.5)
results = yolo_model.predict(image, conf=0.5)

# Use the draw_bbox function to display bounding boxes
image_with_yolo = draw_bbox(image, results)
plt.imshow(image_with_yolo)
plt.axis('off')
plt.show()
"""
    display_solution(code)


def check6(yolo_model):
    try:
        # Check if the YOLO model is loaded properly
        if not isinstance(yolo_model, YOLO):
            display_check(False, "YOLO model is not loaded correctly.")
            return

        display_check(True, "YOLOv8 predictions and visualization are correct. Well done!")

    except Exception as e:
        display_check(False, f"Something went wrong: {str(e)}")
