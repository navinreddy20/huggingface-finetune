# Hugging Face Fine-Tuning Masterclass

Fine-tune DistilBERT on the Emotion dataset using Hugging Face — step by step, no Jupyter needed.

## Setup

```bash
# Clone the repo
git clone https://github.com/navinreddy20/huggingface-finetune.git
cd huggingface-finetune

# Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/

# Create .env file with your Hugging Face tokens
# HF_TOKEN=your_read_token
# HF_TOKEN_W=your_write_token

# Install dependencies (Python 3.13 + CUDA PyTorch)
uv sync
```

## Run Step by Step

Navigate to the `steps/` folder and run each module one at a time:

```bash
cd steps

# Step 1: Authenticate with Hugging Face Hub
uv run step01_auth.py

# Step 2: Load the Emotion dataset
uv run step02_load_data.py

# Step 3: Filter data & engineer features
uv run step03_filter_data.py

# Step 4: Stream large datasets without loading into memory
uv run step04_streaming.py

# Step 5: Build a custom dataset & push to Hub
uv run step05_custom_dataset.py

# Step 6: Tokenization — convert text to numbers
uv run step06_tokenizer.py

# Step 7: Tokenize the entire dataset
uv run step07_tokenize_dataset.py

# Step 8: Fine-tune DistilBERT (uses GPU if available)
uv run step08_train.py

# Step 9: Compare base model vs fine-tuned model
uv run step09_test_model.py
```

## Run All at Once

```bash
uv run python huggingface_finetune.py
```

## Gradio Web App

Launch the Gradio GUI to predict emotions and run steps from the browser:

```bash
uv run python app.py
```

Open http://127.0.0.1:7860 in your browser. The app has two tabs:

- **Emotion Predictor** — Compare predictions from the base model, fine-tuned model, and GPT-4o side by side.
- **Run Steps** — Select and run any training step directly from the UI.