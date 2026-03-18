"""
Step 8: Load the Model & Set Up Training 🧠
Load DistilBERT, define metrics, and configure hyperparameters.
Then fine-tune! 🔥
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from shared import load_state, save_state
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

print("=" * 60)
print("Step 8: Fine-Tuning 🔥")
print("=" * 60)

state = load_state()
checkpoint = state["checkpoint"]

# --- Load model ---
print("🚀 Downloading the DistilBERT Model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Model loaded! Training device: {device.upper()}")
if device == "cpu":
    print("⚠️  Running on CPU — training will be slower. Use a CUDA GPU for best performance.")

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="emotion-distilbert-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
)

# --- Load tokenized data ---
print("📂 Loading tokenized dataset...")
tokenized_datasets = load_from_disk(state["tokenized_path"])

split_data = tokenized_datasets.train_test_split(test_size=0.1)
train_set = split_data["train"]
test_set = split_data["test"]

# --- Train! ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics,
)

print("🚀 IGNITION! Starting the Fine-Tuning Process...\n")
trainer.train()
print("\n🎉 Training Complete!")

# --- Save ---
print("💾 Saving the final model...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
trainer.save_model("emotion-distilbert-model")
tokenizer.save_pretrained("emotion-distilbert-model")
save_state({"model_path": "emotion-distilbert-model"})
print("✅ Model saved to emotion-distilbert-model/")

print("\n👉 Run next:  uv run python steps/step09_test_model.py")
