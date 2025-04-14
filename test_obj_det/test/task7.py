from test_obj_det.utils import display_hint, display_solution, display_check


def hint7():
    hint = """
Step 7:
- Use your custom model to generate predictions on the validation set.
- Use the `evaluate_custom_model` function to calculate metrics like Mean Squared Error (MSE), mean IoU, precision, and mAP.
- Ensure that the predicted bounding boxes and the ground truth bounding boxes are in the same format.
- Consider IoU thresholds from 0.5 to 0.95 for mAP computation.
"""
    display_hint(hint)

def solution7():
    code = """
# === 7. Predict on validation set and evaluate ===
# Generate predictions from the custom model
pred_labels = model.predict(val_images)

# Evaluate the model
evaluate_custom_model(val_labels, pred_labels)
"""
    display_solution(code)

def check7(model, val_images, val_labels):
    try:
        # Check if the model is compiled and trained
        if not model:
            display_check(False, "The model is not properly compiled or trained.")
            return

        # Generate predictions
        pred_labels = model.predict(val_images)
        if pred_labels is None:
            display_check(False, "No predictions generated. Ensure the model is working.")
            return

        display_check(True, "Predictions generated and evaluation completed correctly.")

    except Exception as e:
        display_check(False, f"Something went wrong: {str(e)}")
