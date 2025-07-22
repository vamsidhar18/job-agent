import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Try different models in order of preference
MODELS_TO_TRY = [
    "gpt-4o-mini",     # Newer, cheaper model
    "gpt-3.5-turbo",   # Original choice
    "gpt-4",           # More expensive but widely available
]

def create_llm():
    for model in MODELS_TO_TRY:
        try:
            llm = ChatOpenAI(model=model, temperature=0.2, openai_api_key=openai_api_key)
            # Test with a simple message
            test_response = llm.invoke([HumanMessage(content="test")])
            print(f"✅ Successfully connected using model: {model}")
            return llm
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")
            continue
    
    raise Exception("No available models found. Check your API key and billing status.")

llm = create_llm()

def score_job(job):
    system_prompt = (
        "You are an expert career advisor helping a software engineer "
        "evaluate jobs. Return a JSON with scores and reasons."
    )
    user_prompt = f"""
Job Title: {job['title']}
Company: {job['company']}
Description: {job['description']}

Return a JSON like this:
{{
  "totalScore": 0-100,
  "sponsorshipScore": 0-100,
  "fitScore": 0-100,
  "techScore": 0-100,
  "companyScore": 0-100,
  "shouldApply": true/false,
  "reasoning": "short reason"
}}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content