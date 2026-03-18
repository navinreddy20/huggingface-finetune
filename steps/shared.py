"""
Shared utilities for all steps.
Loads tokens from .env and provides a pickle-based state to pass data between steps.
"""

import os
import pickle

STATE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(STATE_DIR, ".state.pkl")


def load_env():
    """Read tokens from .env file in the project root."""
    env = {}
    env_path = os.path.join(STATE_DIR, "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


def save_state(data: dict):
    """Save shared state so the next step can pick up where we left off."""
    existing = load_state()
    existing.update(data)
    with open(STATE_FILE, "wb") as f:
        pickle.dump(existing, f)


def load_state() -> dict:
    """Load shared state from previous steps."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def clear_state():
    """Reset state (start fresh)."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
