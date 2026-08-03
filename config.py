import os
from pathlib import Path

from dotenv import load_dotenv


# Find the .env file in the same folder as config.py
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Load the exact .env file
load_dotenv(dotenv_path=ENV_FILE)

# Read the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        f"GEMINI_API_KEY was not found.\n"
        f"Expected .env file at: {ENV_FILE}"
    )