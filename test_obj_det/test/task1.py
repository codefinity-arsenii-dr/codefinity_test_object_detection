from test_obj_det.utils import display_hint, display_solution, display_check

def hint1():
    hint = """
Use the `load_data()` function twice — once with `split="train"` and once with `split="val"`.
Make sure to assign the output to three separate variables each time, like:
images, labels, shapes = load_data(...)
"""
    display_hint(hint)

def solution1():
    code = """
train_images, train_labels, train_shapes = load_data("/path/to/dataset/data", split="train")
val_images, val_labels, val_shapes = load_data("/path/to/dataset/data", split="val")
"""
    display_solution(code)

def check1(train_images, train_labels, train_shapes, val_images, val_labels, val_shapes):
    if train_images is None or val_images is None:
        display_check(False, "Make sure you're calling `load_data()` correctly.")
    elif train_images.shape[0] == 0 or val_images.shape[0] == 0:
        display_check(False, "No images were loaded. Double check the data paths and split argument.")
    elif train_images.shape[1:] != (128, 128, 3):
        display_check(False, "Images should be resized to 128x128 with 3 channels.")
    elif train_labels.shape[1] != 4 or val_labels.shape[1] != 4:
        display_check(False, "Labels must be in YOLO format (x, y, w, h).")
    elif len(train_shapes) != train_images.shape[0] or len(val_shapes) != val_images.shape[0]:
        display_check(False, "Original shape count doesn't match number of images.")
    else:
        display_check(True, "Well done! Here's your next key part: LDA92P")
