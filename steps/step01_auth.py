"""
Step 1: Authentication 🔐
Connect to the Hugging Face Hub using your tokens.
"""

import warnings
warnings.filterwarnings("ignore")

from shared import load_env, save_state, clear_state
from huggingface_hub import login

print("=" * 60)
print("Step 1: The Secret Handshake (Authentication)")
print("=" * 60)

# Start fresh
clear_state()

env = load_env()
READ_TOKEN = env.get("HF_TOKEN")
WRITE_TOKEN = env.get("HF_TOKEN_W")

if WRITE_TOKEN:
    login(token=WRITE_TOKEN)
    print("✅ Authenticated successfully! Welcome to the Hub.")
elif READ_TOKEN:
    login(token=READ_TOKEN)
    print("✅ Authenticated with read token.")
else:
    print("⚠️  No HF tokens found. Add them to the .env file.")
    print("   Continuing without authentication (public datasets will still work).")

# Save tokens for later steps
save_state({"READ_TOKEN": READ_TOKEN, "WRITE_TOKEN": WRITE_TOKEN})

print("\n👉 Run next:  uv run python steps/step02_load_data.py")
