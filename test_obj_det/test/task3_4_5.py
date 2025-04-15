import os

from test_obj_det.utils import display_hint, display_solution, display_check


def hint3():
    hint = """
You're building a CNN for predicting 4 normalized bounding box coordinates (x, y, w, h).

- Start with 4 convolutional blocks (Conv2D + MaxPooling2D)
- The input shape is (128, 128, 3)
- Use a Dense layer with 1024 units before the output
- Final layer should be `Dense(4, activation='sigmoid')` to output normalized values
"""
    display_hint(hint)

def solution3():
    code = """
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='sigmoid')
    ])
    return model
"""
    display_solution(code)

def hint4():
    hint = """
- Use `.compile()` before training or loading weights.
- If weights already exist, use `.load_weights()`.
- Otherwise, train the model using `.fit()` and save weights with `.save_weights()`.
"""
    display_hint(hint)

def solution4():
    code = """
def build_load_model(X_train, y_train, weights_path, optimizer, loss, metrics):
    model = build_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        model.save_weights(weights_path)

    return model
"""
    display_solution(code)

def hint5():
    hint = """
- Call `build_load_model()` with real training data and valid parameters for training.
"""
    display_hint(hint)

def solution5():
    code = """
model = build_load_model(
    X_train=train_images_norm,
    y_train=train_labels,
    weights_path="model_weights.h5",
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    metrics=["mae"]
)
"""
    display_solution(code)

def check345(model, weights_path):
    try:
        # Check model architecture (types only)
        expected_layer_types = [
            "Conv2D",
            "MaxPooling2D",
            "Conv2D",
            "MaxPooling2D",
            "Conv2D",
            "MaxPooling2D",
            "Conv2D",
            "MaxPooling2D",
            "Flatten",
            "Dense",
            "Dropout",
            "Dense"
        ]

        layers_list = model.layers
        if len(layers_list) != len(expected_layer_types):
            display_check(False, f"Expected {len(expected_layer_types)} layers, found {len(layers_list)}.")
            return

        for i, expected_type in enumerate(expected_layer_types):
            if expected_type not in str(type(layers_list[i])):
                display_check(False, f"Layer {i+1} should be {expected_type}, found {type(layers_list[i]).__name__}.")
                return

        # Check if model is compiled
        if model.optimizer is None:
            display_check(False, "Model is not compiled. Did you call `compile()`?")
            return

        # Check if model is either trained or has weights loaded
        has_weights_loaded = os.path.exists(weights_path)
        has_training_history = hasattr(model, 'history') and hasattr(model.history, 'history') and len(model.history.history) > 0

        if not has_weights_loaded and not has_training_history:
            display_check(False, "Model is neither trained nor has weights loaded.")
            return

        display_check(True, "Model architecture and setup look good! Next key: 7XPB9R")

    except Exception as e:
        display_check(False, f"Something went wrong: {str(e)}")

