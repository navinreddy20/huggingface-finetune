"""
Step 2: Loading the Emotion Dataset 🧠
Pull the dair-ai/emotion dataset from the Hugging Face Hub.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import save_state
from datasets import load_dataset

print("=" * 60)
print("Step 2: Summoning the Data")
print("=" * 60)

print("⏳ Fetching the Emotion dataset...")
emotion_data = load_dataset("dair-ai/emotion", "split", split="train")

print(f"Total records fetched: {len(emotion_data)}\n")

first_record = emotion_data[0]
print("--- 📝 Sample Record ---")
print(f"Text: '{first_record['text']}'")
print(f"Emotion Label ID: {first_record['label']}")
print("(0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise)")

# Save dataset path for later steps (re-load from cache is instant)
save_state({"dataset_name": "dair-ai/emotion"})

print("\n👉 Run next:  uv run python steps/step03_filter_data.py")
