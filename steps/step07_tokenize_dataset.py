"""
Step 7: Tokenize the Entire Dataset 🏭
Run the tokenizer over every row to produce input_ids and attention_mask.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import load_state, save_state
from datasets import load_dataset
from transformers import AutoTokenizer

print("=" * 60)
print("Step 7: Tokenizing the Entire Dataset")
print("=" * 60)

state = load_state()
checkpoint = state["checkpoint"]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
emotion_data = load_dataset(state["dataset_name"], "split", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("⏳ Translating the entire dataset into math... Hang tight!")
tokenized_datasets = emotion_data.map(tokenize_function, batched=True)

print("\n🎉 Tokenization Complete!")
print("Columns:", tokenized_datasets.column_names)

# Save to disk so training step can load it
tokenized_datasets.save_to_disk("tokenized_emotion")
save_state({"tokenized_path": "tokenized_emotion"})

print("💾 Tokenized data saved to tokenized_emotion/")

print("\n👉 Run next:  uv run python steps/step08_load_model.py")
