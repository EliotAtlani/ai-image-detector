import gradio as gr
import random
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================
# 1. Load Model & Gather Images
# =========================

model_path = "/Users/eliotatlani/Documents/Harvard/AC209b/ai-image-detector/weights/baseline_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
model = tf.keras.models.load_model(model_path)


classes = ["Fake", "Human"]


test_images_dir = "/Users/eliotatlani/Documents/Harvard/AC209b/ai-image-detector/dataset_test/faces"
if not os.path.isdir(test_images_dir):
    raise FileNotFoundError(
        f"Test images directory not found at: {test_images_dir}")

image_list = []
for root, dirs, files in os.walk(test_images_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_list.append(os.path.join(root, file))
if not image_list:
    raise ValueError(
        "No image files found in the specified directory. Please verify your path and file extensions.")

# =========================
# 2. Helper Functions
# =========================


def preprocess(image):

    if not isinstance(image, Image.Image):

        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    image = image.resize((200, 200))
    image = np.array(image) / 255.0

    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)

    elif image.shape[-1] != 3:

        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)
    return image


def get_ground_truth(image_path):

    parent = os.path.basename(os.path.dirname(image_path))
    if parent.lower() in ["real", "human"]:
        return "Human"
    elif parent.lower() in ["fake", "artificial"]:
        return "Fake"
    else:
        print(
            f"Warning: Could not infer ground truth for {image_path} from its directory name.")
        return "Unknown"


# =========================
# 3. Initialize Quiz State
# =========================


initial_image_path = random.choice(image_list)
initial_state = {
    "question_num": 0,
    "user_score": 0,
    "ai_score": 0,
    "current_image": initial_image_path
}

# load the initial image
try:
    initial_image = Image.open(initial_image_path).convert("RGB")
except FileNotFoundError:
    print(
        f"Error: Initial image not found at {initial_image_path}. Choosing another image.")

    image_list.remove(initial_image_path)  # Remove the invalid path
    if image_list:
        initial_image_path = random.choice(image_list)
        initial_state["current_image"] = initial_image_path
        initial_image = Image.open(initial_image_path).convert("RGB")
    else:
        raise ValueError(
            "No valid image files found after attempting to load the first.")


# =========================
# 4. Update Function for the Quiz
# =========================


def update_quiz(user_guess, state):

    image_path = state["current_image"]
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:

        return gr.update(value=None), "Error loading image. Skipping question.", f"Question: {state['question_num']}/10\nYour score: {state['user_score']}\nAI score: {state['ai_score']}", gr.update(value=None), state

    ground_truth = get_ground_truth(image_path)
    if ground_truth == "Unknown":
        result_message = "Could not determine ground truth for this image. Skipping score update."
        score_message = (
            f"Question: {state['question_num']} / 10\n"
            f"Your score: {state['user_score']}\n"
            f"AI score: {state['ai_score']}"
        )

        if state["question_num"] < 10 and image_list:
            new_image_path = random.choice(image_list)
            state["current_image"] = new_image_path
            new_image = Image.open(new_image_path).convert("RGB")
            radio_update = gr.update(value=None)
            return new_image, result_message, score_message, radio_update, state
        elif state["question_num"] >= 10:
            final_message = (
                f"Quiz Finished!\nFinal Score: {state['user_score']} (You) vs "
                f"{state['ai_score']} (AI) out of 10 questions."
            )
            radio_update = gr.update(value=None, interactive=False)
            return image, result_message + "\n" + final_message, score_message, radio_update, state
        else:
            return image, "Quiz ended. No more images.", score_message, gr.update(interactive=False), state

    input_image = preprocess(image)
    predictions = model.predict(input_image)
    pred_index = np.argmax(predictions[0])
    model_guess = classes[pred_index]

    user_correct = (user_guess == ground_truth)
    model_correct = (model_guess == ground_truth)

    state["question_num"] += 1
    if user_correct:
        state["user_score"] += 1
    if model_correct:
        state["ai_score"] += 1

    result_message = (
        f"Your answer: {user_guess} \n Ground truth: {ground_truth} \n"
        f"{'Correct' if user_correct else 'Incorrect'}.\n"
        f"AI predicted: {model_guess}."
    )

    score_message = (
        f"Question: {state['question_num']} / 10\n"
        f"Your score: {state['user_score']}\n"
        f"AI score: {state['ai_score']}"
    )

    if state["question_num"] < 10:

        new_image_path = random.choice(image_list)
        state["current_image"] = new_image_path
        new_image = Image.open(new_image_path).convert("RGB")

        radio_update = gr.update(value=None)
        return new_image, result_message, score_message, radio_update, state
    else:

        final_message = (
            f"Quiz Finished!\nFinal Score: {state['user_score']} (You) vs "
            f"{state['ai_score']} (AI) out of 10 questions."
        )
        radio_update = gr.update(value=None, interactive=False)

        return image, result_message + "\n" + final_message, score_message, radio_update, state

# =========================
# 5. Build the Gradio Interface Using Blocks (for flexible state management)
# =========================


css = """
#image-output {
    
    width: 100%;
    height: 100%; 
    display: flex; 
    justify-content: center; 
    align-items: center; 
    overflow: hidden;
}

#image-output img {
    width: 100%;
    height: 100%;
    object-fit: fill;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Row():
        gr.Markdown("# Image Quiz: You vs AI")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            result_text = gr.Textbox(
                label="Result", interactive=False, elem_id="result-textbox")
            score_text = gr.Textbox(
                label="Score", interactive=False, elem_id="score-textbox")

            user_guess = gr.Radio(
                choices=classes,
                label="Is this image Fake or Real?",
                value=None,
                elem_id="center-radio"
            )

            state_component = gr.State(value=initial_state)

            submit_btn = gr.Button("Submit Answer")

        with gr.Column(scale=2, min_width=600):

            image_output = gr.Image(
                label="Image",
                value=initial_image,
                elem_id="image-output",

            )

    submit_btn.click(
        fn=update_quiz,
        inputs=[user_guess, state_component],
        outputs=[image_output, result_text,
                 score_text, user_guess, state_component]
    )


demo.launch()
