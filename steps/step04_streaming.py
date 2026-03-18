"""
Step 4: The Expressway Method (Streaming) 🛣️
Stream data without loading the entire dataset into memory.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import load_state
from datasets import load_dataset

print("=" * 60)
print("Step 4: The Expressway Method (Streaming)")
print("=" * 60)

state = load_state()

# streaming=True means data flows chunk by chunk — no RAM explosion!
streamed_emotions = load_dataset(state["dataset_name"], "split", split="train", streaming=True)
data_pipeline = iter(streamed_emotions)

print("🌊 Live streaming data directly from the Hugging Face servers:\n")
for i in range(4):
    live_record = next(data_pipeline)
    print(f"Data packet {i+1}: {live_record['text']}")

print("\n👉 Run next:  uv run python steps/step05_custom_dataset.py")
