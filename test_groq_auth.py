from pathlib import Path
import os

from dotenv import load_dotenv
from groq import Groq

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("GROQ_API_KEY", "").strip()

if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")

print("API key loaded:", True)
print("Key format looks correct:", api_key.startswith("gsk_"))

client = Groq(
    api_key=api_key,
    max_retries=0,
    timeout=15
)

try:
    models = client.models.list()

    print("\nAuthentication successful!")
    print("Available models:", len(models.data))

except Exception as error:
    print("\nAuthentication failed!")
    print(repr(error))