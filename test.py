# %pip install llama-index-llms-google-genai llama-index
import os
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("api_key")

llm = GoogleGenAI(
    model= "gemini-2.5-flash",
    api_key=api_key
)


async def get_ai_response(prompt):
    """Hàm này sẽ được gọi từ call.py"""
    try:
        # Sử dụng acomplete (bất đồng bộ) để bot không bị treo
        response = await llm.acomplete(prompt)
        return str(response)
    except Exception as e:
        return f"Lỗi AI: {str(e)}"