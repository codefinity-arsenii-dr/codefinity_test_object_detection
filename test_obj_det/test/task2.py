import numpy as np

from test_obj_det.utils import display_hint, display_solution, display_check


def hint2():
    hint = """
Images should be normalized so that pixel values lie between 0 and 1.
You can achieve this by dividing the pixel values by 255.0.
Create two variables: one for training (`train_images_norm`) and one for validation (`val_images_norm`).
"""
    display_hint(hint)

def solution2():
    code = """
train_images_norm = train_images / 255.0
val_images_norm = val_images / 255.0
"""
    display_solution(code)

def check2(train_images_norm, val_images_norm):
    if not isinstance(train_images_norm, np.ndarray) or not isinstance(val_images_norm, np.ndarray):
        display_check(False, "Output variables should be NumPy arrays.")
    elif not (0.0 <= np.min(train_images_norm) <= np.max(train_images_norm) <= 1.0):
        display_check(False, "train_images_norm is not correctly normalized.")
    elif not (0.0 <= np.min(val_images_norm) <= np.max(val_images_norm) <= 1.0):
        display_check(False, "val_images_norm is not correctly normalized.")
    else:
        display_check(True, "Great job! Here's your next key part: MXR84N")
