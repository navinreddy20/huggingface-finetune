"""
Step 6: Tokenization — The Translator 🔠
Load the DistilBERT tokenizer and see how it converts text into numbers.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import save_state
from transformers import AutoTokenizer

print("=" * 60)
print("Step 6: Tokenization")
print("=" * 60)

checkpoint = "distilbert-base-uncased"

print(f"⚙️ Loading tokenizer for '{checkpoint}'...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("✅ Tokenizer locked and loaded!\n")

# --- Demo: single sentence ---
sample_text = "I absolutely love learning about AI engineering!"
inputs = tokenizer(sample_text)

print(f"Original Text: {sample_text}\n")
print(f"Token IDs: {inputs['input_ids']}")
print(f"Chopped Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'])}")

# --- Demo: padding & truncation ---
print("\n" + "-" * 40)
print("Padding & Truncation Demo:")
print("-" * 40)

batch_sentences = [
    "I am happy.",
    "I am feeling incredibly joyful today because my code compiled on the first try!",
]

batch_inputs = tokenizer(batch_sentences, padding=True, truncation=True)
print("Sentence 1 IDs:", batch_inputs["input_ids"][0])
print("Sentence 2 IDs:", batch_inputs["input_ids"][1])
print("\nNotice the 0s (padding) at the end of Sentence 1!")
print("Attention Mask for Sentence 1:", batch_inputs["attention_mask"][0])

save_state({"checkpoint": checkpoint})

print("\n👉 Run next:  uv run python steps/step07_tokenize_dataset.py")
