import os
import random
from PIL import Image
import gradio as gr
from utils import run_expert, preprocess


def load_quiz_images(folder="../dataset_test"):
    pairs = []
    # for each category (animals, art, faces)
    for category in os.listdir(folder):
        cat_path = os.path.join(folder, category)
        if not os.path.isdir(cat_path):
            continue
        for truth in ["real", "fake"]:
            truth_path = os.path.join(cat_path, truth)
            if not os.path.isdir(truth_path):
                continue
            for fname in os.listdir(truth_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(truth_path, fname)
                    pairs.append((img_path, truth))
    return pairs


all_quiz_pairs = load_quiz_images()


def start_quiz():
    quiz = random.sample(all_quiz_pairs, k=10)
    return quiz, 0, 0, 0, Image.open(quiz[0][0]).convert("RGB")


def answer_round(user_guess, quiz, idx, user_score, model_score):
    path, true_label = quiz[idx]
    img = Image.open(path).convert("RGB")
    model_pred, _ = run_expert(preprocess(img), "Mixture of Experts")

    if user_guess == true_label:
        user_score += 1
    if model_pred == true_label:
        model_score += 1
    idx += 1

    if idx < len(quiz):
        next_img = Image.open(quiz[idx][0]).convert("RGB")
        feedback = (
            f"Round {idx}: you guessed {user_guess},\n "
            f"model guessed {model_pred}, \n answer is {true_label}."
        )
    else:
        next_img = None
        feedback = (
            f"Quiz over. final scores — you: {user_score}, model: {model_score}.")

    return (next_img, feedback, quiz, idx, user_score, model_score, str(user_score), str(model_score))


with gr.Blocks() as demo:
    with gr.Tabs():
        for model_name in ["Mixture of Experts", "Art", "Animal", "Faces"]:
            with gr.TabItem(model_name):
                inp = gr.Image(type="pil", label="Upload Image",
                               width=500, height=500)
                out_label = gr.Label(num_top_classes=1, label="Prediction")
                out_expert = gr.Textbox(label="Expert Routed")
                btn = gr.Button("Classify with " + model_name)
                btn.click(
                    fn=lambda img, m=model_name: run_expert(
                        preprocess(img), m),
                    inputs=inp,
                    outputs=[out_label, out_expert]
                )
        with gr.TabItem("Quiz"):
            img_display = gr.Image(label="Quiz Image", type="pil")
            choice = gr.Radio(
                ["real", "fake"],
                label="Your guess",
            )
            start_btn = gr.Button("Start Quiz")
            answer_btn = gr.Button("Submit Guess")
            feedback = gr.Textbox(label="Feedback")
            user_score_txt = gr.Textbox(label="Your Score")
            model_score_txt = gr.Textbox(label="Model Score")
            quiz_state = gr.State()
            idx_state = gr.State()
            user_score_state = gr.State()
            model_score_state = gr.State()

            start_btn.click(
                fn=start_quiz,
                outputs=[quiz_state, idx_state, user_score_state,
                         model_score_state, img_display,],
            )
            answer_btn.click(
                fn=answer_round,
                inputs=[choice, quiz_state, idx_state,
                        user_score_state, model_score_state],
                outputs=[img_display, feedback, quiz_state, idx_state, user_score_state,
                         model_score_state, user_score_txt, model_score_txt,],
            )

    demo.launch()
