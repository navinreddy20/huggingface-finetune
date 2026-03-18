"""
Step 9: The Ultimate Test — Before vs. After 🥊
Compare the raw base model with our fine-tuned expert.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import load_state, load_env
from transformers import pipeline
from openai import OpenAI

print("=" * 60)
print("Step 9: The Ultimate Test (Before vs. After)")
print("=" * 60)

state = load_state()
model_path = state["model_path"]
checkpoint = state["checkpoint"]

# Set up OpenAI client
env = load_env()
openai_key = env.get("OPENAI_API_KEY")
if openai_key:
    client = OpenAI(api_key=openai_key)
    print("✅ OpenAI GPT-4o ready for verification!\n")
else:
    client = None
    print("⚠️  No OPENAI_API_KEY in .env — skipping GPT-4o verification.\n")

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

# test_sentence = "I was so frustrated when my code crashed, but finding the bug made me so incredibly happy!"
# test_sentence = "I am terrified of the dark and always check under my bed before sleeping."

test_sentences = [
    # --- 4 Straightforward ---
    "I am so happy today, everything is going great!",                          # Joy
    "I love you so much, you mean the world to me.",                            # Love
    "This makes me really angry, I can't believe it.",                          # Anger
    "I feel so lonely and empty inside.",                                       # Sadness

    # --- 4 Tricky / Ambiguous ---
    "I can't believe I won, I was sure I would lose everything.",               # Surprise hidden in negative framing
    "My heart was racing when I opened the letter, I never expected this.",     # Fear or Surprise?
    "She smiled at me and walked away without saying a word.",                  # Love or Sadness?
    "I laughed so hard I started crying and couldn't stop shaking.",            # Joy disguised as distress
]

for test_sentence in test_sentences:
    print("\n" + "=" * 60)
    print(f"📝 TEST: '{test_sentence}'")
    print("=" * 60)

    base_result = base_classifier(test_sentence)[0]
    print(f"  ❌ Base Model:     {base_result['label']} ({base_result['score']*100:.1f}%)")

    expert_result = expert_classifier(test_sentence)[0]
    label = emotion_map.get(expert_result['label'], expert_result['label'])
    print(f"  ✅ Fine-Tuned:     {label} ({expert_result['score']*100:.1f}%)")

    if client:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an emotion classifier. Classify the emotion of the given text into exactly one of: Sadness, Joy, Love, Anger, Fear, Surprise. Reply with ONLY the emotion word, nothing else."},
                {"role": "user", "content": test_sentence},
            ],
            max_tokens=10,
            temperature=0,
        )
        gpt_label = response.choices[0].message.content.strip()
        print(f"  🤖 GPT-4o:        {gpt_label}")

print("\n🎯 Masterclass Complete! 🚀")
