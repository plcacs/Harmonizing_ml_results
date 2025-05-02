import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

models = openai.models.list()

for model in models:
    print(model.id)