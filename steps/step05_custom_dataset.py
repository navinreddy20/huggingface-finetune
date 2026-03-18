"""
Step 5: Building Your Own Dataset 🚀
Create a custom dataset and optionally push it to the Hub.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from shared import load_state
from datasets import Dataset

print("=" * 60)
print("Step 5: Building Your Own Dataset")
print("=" * 60)

state = load_state()
WRITE_TOKEN = state.get("WRITE_TOKEN")

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
    print("⚠️  Skipping push to Hub (no write token). Dataset created locally:")
    print(student_dataset)

print("\n👉 Run next:  uv run python steps/step06_tokenizer.py")
