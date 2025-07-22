import os
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")

client = openai.OpenAI(api_key=api_key)

# Test available models
try:
    models = client.models.list()
    available_models = [model.id for model in models.data if 'gpt' in model.id]
    print("Available GPT models:", available_models)
except Exception as e:
    print(f"Error fetching models: {e}")

# Test simple completion
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Try the most accessible model first
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ API working! Response:", response.choices[0].message.content)
except Exception as e:
    print(f"❌ API test failed: {e}")