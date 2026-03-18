"""
Step 9: The Ultimate Test — Before vs. After 🥊
Compare the raw base model with our fine-tuned expert.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import load_state
from transformers import pipeline

print("=" * 60)
print("Step 9: The Ultimate Test (Before vs. After)")
print("=" * 60)

state = load_state()
model_path = state["model_path"]
checkpoint = state["checkpoint"]

print("🔌 Booting up the Base Intern (Untrained)...")
base_classifier = pipeline(
    "text-classification",
    model=checkpoint,
    tokenizer=checkpoint,
)

print("🔌 Booting up our Custom Expert (Fine-Tuned)...")
expert_classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
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

print("\n" + "=" * 60)
print(f"📝 TEST SENTENCE: '{test_sentence}'")
print("=" * 60)

base_result = base_classifier(test_sentence)[0]
print("\n❌ BEFORE FINE-TUNING (Base Model):")
print(f"Prediction: {base_result['label']} (It has no idea what this means)")
print(f"Confidence: {base_result['score']*100:.1f}% (Guessing blindly)")

expert_result = expert_classifier(test_sentence)[0]
print("\n✅ AFTER FINE-TUNING (Our Custom Model):")
print(f"Prediction: {emotion_map.get(expert_result['label'], expert_result['label'])}")
print(f"Confidence: {expert_result['score']*100:.1f}% (Highly confident!)")
print("=" * 60)

print("\n🎯 Masterclass Complete! 🚀")
