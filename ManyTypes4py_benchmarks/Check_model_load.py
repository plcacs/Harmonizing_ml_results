import os
import openai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env in parent directory

openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)
models = openai.models.list()

for model in models:
    print(model.id)