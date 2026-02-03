# pip install llama-index-llms-gemini llama-index-embeddings-gemini
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
import os
# Cấu hình mô hình
api_key = os.getenv("api_key")
Settings.llm = Gemini(model_name="models/gemini-1.5-flash", api_key=api_key)
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)

# Sau đó dùng code 5 dòng như cũ
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print(query_engine.query("Tài liệu nói gì về OCR?"))