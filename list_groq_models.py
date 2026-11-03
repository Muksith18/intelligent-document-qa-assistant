from pathlib import Path
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=0
)

models = client.models.list()

print("\n========== AVAILABLE GROQ MODELS ==========\n")

for model in models.data:
    print(model.id)

print("\n===========================================")