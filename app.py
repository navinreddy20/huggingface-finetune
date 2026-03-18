"""
Gradio GUI for the Emotion Fine-Tuning Masterclass.
Launch with:  uv run python app.py
"""

import os
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

import gradio as gr
from transformers import pipeline
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
def load_env():
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


env = load_env()
openai_key = env.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key) if openai_key else None

EMOTION_MAP = {
    "LABEL_0": "Sadness 😢",
    "LABEL_1": "Joy 😄",
    "LABEL_2": "Love ❤️",
    "LABEL_3": "Anger 😡",
    "LABEL_4": "Fear 😨",
    "LABEL_5": "Surprise 😲",
}

MODEL_DIR = os.path.join("steps", "emotion-distilbert-model")
BASE_CHECKPOINT = "distilbert-base-uncased"

# ---------------------------------------------------------------------------
# Lazy-load classifiers (loaded once on first prediction)
# ---------------------------------------------------------------------------
_base_clf = None
_expert_clf = None


def get_base_classifier():
    global _base_clf
    if _base_clf is None:
        _base_clf = pipeline("text-classification", model=BASE_CHECKPOINT, tokenizer=BASE_CHECKPOINT)
    return _base_clf


def get_expert_classifier():
    global _expert_clf
    if _expert_clf is None:
        if not os.path.exists(MODEL_DIR):
            raise gr.Error("Fine-tuned model not found! Run Step 8 (training) first.")
        _expert_clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
    return _expert_clf


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_emotion(text):
    if not text.strip():
        return "Please enter a sentence.", "", ""

    # Base model
    base_clf = get_base_classifier()
    base_result = base_clf(text)[0]
    base_out = f"{base_result['label']}  ({base_result['score']*100:.1f}%)"

    # Fine-tuned model
    expert_clf = get_expert_classifier()
    expert_result = expert_clf(text)[0]
    expert_label = EMOTION_MAP.get(expert_result["label"], expert_result["label"])
    expert_out = f"{expert_label}  ({expert_result['score']*100:.1f}%)"

    # GPT-4o verification
    if client:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emotion classifier. Classify the emotion of the given text "
                        "into exactly one of: Sadness, Joy, Love, Anger, Fear, Surprise. "
                        "Reply with ONLY the emotion word, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0,
        )
        gpt_out = response.choices[0].message.content.strip()
    else:
        gpt_out = "No OPENAI_API_KEY in .env"

    return base_out, expert_out, gpt_out


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
STEPS = [
    ("Step 1: Authentication", "step01_auth.py"),
    ("Step 2: Load Dataset", "step02_load_data.py"),
    ("Step 3: Filter & Engineer", "step03_filter_data.py"),
    ("Step 4: Streaming Demo", "step04_streaming.py"),
    ("Step 5: Custom Dataset", "step05_custom_dataset.py"),
    ("Step 6: Tokenizer Demo", "step06_tokenizer.py"),
    ("Step 7: Tokenize Dataset", "step07_tokenize_dataset.py"),
    ("Step 8: Fine-Tune Model", "step08_train.py"),
    ("Step 9: Test Model", "step09_test_model.py"),
]


def run_step(step_name):
    if not step_name:
        return "Select a step to run."

    # Find the matching file
    filename = None
    for name, fname in STEPS:
        if name == step_name:
            filename = fname
            break

    if not filename:
        return "Unknown step."

    script_path = os.path.join("steps", filename)
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", script_path],
            capture_output=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=1800,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        output = result.stdout.decode("utf-8", errors="replace")
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr.decode("utf-8", errors="replace")
        if result.returncode != 0:
            output += f"\n\n❌ Step exited with code {result.returncode}"
        return output if output.strip() else "✅ Step completed (no output)."
    except subprocess.TimeoutExpired:
        return "⏰ Step timed out (30 min limit)."
    except Exception as e:
        return f"❌ Error: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Emotion Fine-Tuning Masterclass",
    theme=gr.themes.Soft(),
) as app:
    gr.Markdown(
        """
        # 🧠 Hugging Face Fine-Tuning Masterclass
        ### By Navin Reddy — Telusko

        Fine-tune DistilBERT on the Emotion dataset and compare it with GPT-4o.
        """
    )

    with gr.Tab("🎯 Emotion Predictor"):
        gr.Markdown("Enter a sentence and see how 3 models classify its emotion.")

        text_input = gr.Textbox(
            label="Enter a sentence",
            placeholder="e.g. I laughed so hard I started crying and couldn't stop shaking.",
            lines=2,
        )
        predict_btn = gr.Button("Predict Emotion", variant="primary")

        with gr.Row():
            base_output = gr.Textbox(label="❌ Base Model (Untrained)", interactive=False)
            expert_output = gr.Textbox(label="✅ Fine-Tuned Model", interactive=False)
            gpt_output = gr.Textbox(label="🤖 GPT-4o", interactive=False)

        predict_btn.click(
            fn=predict_emotion,
            inputs=text_input,
            outputs=[base_output, expert_output, gpt_output],
        )

        gr.Markdown("### Try these examples:")
        gr.Examples(
            examples=[
                ["I am so happy today, everything is going great!"],
                ["I love you so much, you mean the world to me."],
                ["This makes me really angry, I can't believe it."],
                ["I feel so lonely and empty inside."],
                ["I can't believe I won, I was sure I would lose everything."],
                ["My heart was racing when I opened the letter, I never expected this."],
                ["She smiled at me and walked away without saying a word."],
                ["I laughed so hard I started crying and couldn't stop shaking."],
            ],
            inputs=text_input,
        )

    with gr.Tab("🚀 Run Steps"):
        gr.Markdown(
            """
            Run each training step one by one. **Run them in order** — each step
            depends on the previous one.

            ⚠️ **Step 8 (training) will take several minutes** depending on your hardware.
            """
        )

        step_dropdown = gr.Dropdown(
            label="Select a step",
            choices=[name for name, _ in STEPS],
        )
        run_btn = gr.Button("▶ Run Step", variant="primary")
        step_output = gr.Textbox(
            label="Output",
            lines=25,
            max_lines=50,
            interactive=False,
        )

        run_btn.click(fn=run_step, inputs=step_dropdown, outputs=step_output)

    gr.Markdown(
        """
        ---
        **Labels:** 0=Sadness 😢 · 1=Joy 😄 · 2=Love ❤️ · 3=Anger 😡 · 4=Fear 😨 · 5=Surprise 😲
        """
    )

if __name__ == "__main__":
    app.launch()
