from moe import MoEModel
import numpy as np
from PIL import Image
import tensorflow as tf


# path
DECISION_MODEL_PATH = "../weights/decision_model.h5"
ANIMAL_MODEL_PATH = "../weights/animals_baseline_model.h5"
ART_MODEL_PATH = "../weights/art_model.h5"
FACE_MODEL_PATH = "../weights/art_baseline_model.h5"
BASELINE_MODEL_PATH = "../weights/baseline_model.h5"

# load
decision_model = tf.keras.models.load_model(DECISION_MODEL_PATH)
animal_expert = tf.keras.models.load_model(ANIMAL_MODEL_PATH)
art_expert = tf.keras.models.load_model(ART_MODEL_PATH)
face_expert = tf.keras.models.load_model(FACE_MODEL_PATH)
baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
experts = [animal_expert, art_expert, face_expert]
experts_names = ["animals", "art", "faces"]


moe = MoEModel(decision_model, experts)


def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"), "RGB")
    image = image.convert("RGB").resize((200, 200))
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image):
    x = preprocess(image)
    print("input shape:", x.shape)
    score = moe(x, training=False).numpy()[0][0]
    label = "real" if score >= 0.5 else "fake"
    expert = experts_names[moe.last_expert_idx]
    return label, expert


def run_expert(x, model_name):
    if model_name == "Mixture of Experts":
        score = moe(x, training=False).numpy()[0][0]
        expert_used = experts_names[moe.last_expert_idx]
    elif model_name == "Art":
        score = art_expert(x, training=False).numpy()[0][0]
        expert_used = "art"
    elif model_name == "Animal":
        score = animal_expert(x, training=False).numpy()[0][0]
        expert_used = "animals"
    elif model_name == "Faces":
        score = face_expert(x, training=False).numpy()[0][0]
        expert_used = "faces"
    elif model_name == "Baseline":
        score = baseline_model(x, training=False).numpy()[0][0]
        expert_used = "baseline"
    else:
        raise ValueError(f"Unknown model choice {model_name}")

    label = "real" if score >= 0.5 else "fake"
    return label, expert_used
