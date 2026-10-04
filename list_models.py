"""One-off script: list Gemini models available to your API key, filtered by capability."""
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Pass "embed" or "generate" as an argument; defaults to "generate".
action = "embedContent" if (len(sys.argv) > 1 and sys.argv[1] == "embed") else "generateContent"

for model in client.models.list():
    if action in (model.supported_actions or []):
        print(model.name)
