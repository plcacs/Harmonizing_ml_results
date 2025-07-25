import os
import anthropic
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("ANTHROPIC_API_KEY")
print(api_key)
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key="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")

# List available models
models = client.models.list()

# Print model information
print("Available Anthropic Models:")
for model in models.data:
    print(f"- ID: {model.id} | Max Tokens: {model.max_tokens} | Training Cutoff: {model.training_cutoff}")
