"""
Hugging Face Masterclass: Fine-Tuning DistilBERT on Emotion Dataset

Run with uv:
    uv run finetune

Or directly:
    uv run huggingface_finetune.py

Reads HF_TOKEN and HF_TOKEN_W from a .env file in the same directory.
Format (one per line):  KEY=value
"""

import os
import warnings
import numpy as np
import pandas as pd
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch


def main():
    warnings.filterwarnings("ignore")

    # ============================================================
    # Module 1: Authentication
    # ============================================================
    print("="*60)
    print("Module 1: Authentication")
    print("="*60)

    # Read tokens from .env file
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()

    READ_TOKEN = env.get("HF_TOKEN")
    WRITE_TOKEN = env.get("HF_TOKEN_W")

    if WRITE_TOKEN:
        login(token=WRITE_TOKEN)
        print("✅ Authenticated successfully! Welcome to the Hub.")
    elif READ_TOKEN:
        login(token=READ_TOKEN)
        print("✅ Authenticated with read token.")
    else:
        print("⚠️  No HF tokens found. Set HF_TOKEN / HF_TOKEN_W environment variables.")
        print("   Continuing without authentication (public datasets will still work).\n")

    # ============================================================
    # Module 2: Loading the Emotion Dataset
    # ============================================================
    print("\n" + "="*60)
    print("Module 2: Summoning the Data")
    print("="*60)

    print("⏳ Fetching the Emotion dataset...")
    emotion_data = load_dataset("dair-ai/emotion", "split", split="train")

    print(f"Total records fetched: {len(emotion_data)}\n")

    first_record = emotion_data[0]
    print("--- 📝 Sample Record ---")
    print(f"Text: '{first_record['text']}'")
    print(f"Emotion Label ID: {first_record['label']}")

    # ============================================================
    # Module 3: Data Purification & Engineering
    # ============================================================
    print("\n" + "="*60)
    print("Module 3: Data Purification & Engineering")
    print("="*60)

    def extract_joy(row):
        return row["label"] == 1

    joy_dataset = emotion_data.filter(extract_joy)

    def count_words(row):
        return {"word_count": len(row["text"].split())}

    processed_joy_data = joy_dataset.map(count_words)

    print(f"🎉 We now have {len(processed_joy_data)} joyful records ready to go!")
    print(f"Sample word count of first record: {processed_joy_data[0]['word_count']} words.")

    # ============================================================
    # Module 4: Streaming Demo
    # ============================================================
    print("\n" + "="*60)
    print("Module 4: The Expressway Method (Streaming)")
    print("="*60)

    streamed_emotions = load_dataset("dair-ai/emotion", "split", split="train", streaming=True)
    data_pipeline = iter(streamed_emotions)

    print("🌊 Live streaming data directly from the Hugging Face servers:\n")
    for i in range(4):
        live_record = next(data_pipeline)
        print(f"Data packet {i+1}: {live_record['text']}")

    # ============================================================
    # Module 6: Building a Custom Dataset & Pushing to Hub
    # ============================================================
    print("\n" + "="*60)
    print("Module 6: Building Your Own Dataset")
    print("="*60)

    query_data = {
        "student_message": [
            "Can I get doubt solving support on the WhatsApp number?",
            "Is the GIL entirely removed in Python 3.13?",
            "Will the new series cover Spring AI or Lang4J",
            "How do I access the new Learning Management System?",
            "Where can I find the multithreading tutorial for Python?",
        ],
        "category": [
            "Enrollment (WhatsApp is NOT for doubts)",
            "Technical (Sub-interpreter feature, not entirely removed)",
            "Course Content (Spring AI)",
            "Platform Access",
            "Video Request",
        ],
    }

    df = pd.DataFrame(query_data)
    student_dataset = Dataset.from_pandas(df)

    if WRITE_TOKEN:
        repo_name = "navinreddy20/student-support-queries"
        print(f"🚀 Launching our custom dataset to {repo_name}...")
        student_dataset.push_to_hub(repo_name, token=WRITE_TOKEN)
        print(f"🎉 Dataset is now live at https://huggingface.co/datasets/{repo_name}")
    else:
        print("⚠️  Skipping push to Hub (no write token). Dataset created locally.")
        print(student_dataset)

    # ============================================================
    # Module 7: Tokenization
    # ============================================================
    print("\n" + "="*60)
    print("Module 7: Tokenization")
    print("="*60)

    print("⚙️ Booting up the Tokenizer...")
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print(f"✅ Tokenizer for '{checkpoint}' is locked and loaded!")

    # ============================================================
    # Module 8: Tokenizer Demo
    # ============================================================
    print("\n" + "="*60)
    print("Module 8: Under the Hood of Tokenization")
    print("="*60)

    sample_text = "I absolutely love learning about AI engineering!"
    inputs = tokenizer(sample_text)

    print(f"Original Text: {sample_text}\n")
    print(f"Token IDs (The Math the AI sees): {inputs['input_ids']}")
    print(f"\nChopped Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'])}")

    # ============================================================
    # Module 9: Padding & Truncation Demo
    # ============================================================
    print("\n" + "="*60)
    print("Module 9: Padding & Truncation")
    print("="*60)

    batch_sentences = [
        "I am happy.",
        "I am feeling incredibly joyful today because my code compiled on the first try!",
    ]

    batch_inputs = tokenizer(batch_sentences, padding=True, truncation=True)
    print("Sentence 1 IDs:", batch_inputs["input_ids"][0])
    print("Sentence 2 IDs:", batch_inputs["input_ids"][1])
    print("\nAttention Mask for Sentence 1:", batch_inputs["attention_mask"][0])

    # ============================================================
    # Module 10: Tokenize the Entire Dataset
    # ============================================================
    print("\n" + "="*60)
    print("Module 10: Tokenizing the Entire Dataset")
    print("="*60)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("⏳ Translating our entire dataset into Math... Hang tight!")
    tokenized_datasets = emotion_data.map(tokenize_function, batched=True)

    print("\n🎉 Tokenization Complete!")
    print("Columns:", tokenized_datasets.column_names)

    # ============================================================
    # Module 11: Loading the Model
    # ============================================================
    print("\n" + "="*60)
    print("Module 11: Loading the Brain (DistilBERT)")
    print("="*60)

    print("🚀 Downloading the DistilBERT Model...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Model loaded! Training device: {device.upper()}")
    if device == "cpu":
        print("⚠️  Running on CPU — training will be slower. Use a CUDA GPU for best performance.")

    # ============================================================
    # Module 12: Metrics
    # ============================================================
    print("\n" + "="*60)
    print("Module 12: Defining Metrics")
    print("="*60)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}

    print("✅ Scoreboard ready!")

    # ============================================================
    # Module 13: Training Arguments
    # ============================================================
    print("\n" + "="*60)
    print("Module 13: Training Blueprint (Hyperparameters)")
    print("="*60)

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

    print("✅ Blueprint locked in!")

    # ============================================================
    # Module 14: Fine-Tuning
    # ============================================================
    print("\n" + "="*60)
    print("Module 14: Fine-Tuning 🔥")
    print("="*60)

    split_data = tokenized_datasets.train_test_split(test_size=0.1)
    train_set = split_data["train"]
    test_set = split_data["test"]

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

    print("💾 Saving the final model...")
    trainer.save_model("emotion-distilbert-model")
    tokenizer.save_pretrained("emotion-distilbert-model")
    print("✅ Model saved to emotion-distilbert-model/")

    # ============================================================
    # Module 15: Before vs. After Comparison
    # ============================================================
    print("\n" + "="*60)
    print("Module 15: The Ultimate Test (Before vs. After)")
    print("="*60)

    print("🔌 Booting up the Base Intern (Untrained)...")
    base_classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased",
        tokenizer="distilbert-base-uncased",
    )

    print("🔌 Booting up our Custom Expert (Fine-Tuned)...")
    expert_classifier = pipeline(
        "text-classification",
        model="emotion-distilbert-model",
        tokenizer="emotion-distilbert-model",
    )

    emotion_map = {
        "LABEL_0": "Sadness 😢",
        "LABEL_1": "Joy 😄",
        "LABEL_2": "Love ❤️",
        "LABEL_3": "Anger 😡",
        "LABEL_4": "Fear 😨",
        "LABEL_5": "Surprise 😲",
    }

    test_sentence = "I was so frustrated when my code crashed, but finding the bug made me so incredibly happy!"

    print("\n" + "="*60)
    print(f"📝 TEST SENTENCE: '{test_sentence}'")
    print("="*60)

    base_result = base_classifier(test_sentence)[0]
    print("\n❌ BEFORE FINE-TUNING (Base Model):")
    print(f"Prediction: {base_result['label']} (It has no idea what this means)")
    print(f"Confidence: {base_result['score']*100:.1f}% (Guessing blindly)")

    expert_result = expert_classifier(test_sentence)[0]
    print("\n✅ AFTER FINE-TUNING (Our Custom Model):")
    print(f"Prediction: {emotion_map.get(expert_result['label'], expert_result['label'])}")
    print(f"Confidence: {expert_result['score']*100:.1f}% (Highly confident!)")
    print("="*60)

    print("\n🎯 Masterclass Complete!")


if __name__ == "__main__":
    main()
