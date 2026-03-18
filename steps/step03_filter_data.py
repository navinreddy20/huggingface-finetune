"""
Step 3: Data Purification & Engineering 🧹
Filter for joyful records and engineer a word_count feature.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import load_state
from datasets import load_dataset

print("=" * 60)
print("Step 3: Data Purification & Engineering")
print("=" * 60)

state = load_state()
emotion_data = load_dataset(state["dataset_name"], "split", split="train")

# Filter: only 'joy' records (label == 1)
def extract_joy(row):
    return row["label"] == 1

joy_dataset = emotion_data.filter(extract_joy)

# Map: add a word_count column
def count_words(row):
    return {"word_count": len(row["text"].split())}

processed_joy_data = joy_dataset.map(count_words)

print(f"🎉 We now have {len(processed_joy_data)} joyful records ready to go!")
print(f"Sample: '{processed_joy_data[0]['text']}'")
print(f"Word count: {processed_joy_data[0]['word_count']} words")

print("\n👉 Run next:  uv run python steps/step04_streaming.py")
