"""
RAG Pipeline – PDF → Chunking → ChromaDB → Q&A
Models  : Qwen 2.5 (3B) and/or Mistral 7B via Ollama (local)
Embeddings: sentence-transformers (local)
"""

import os
import re
import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "pdf_rag"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multilingual, ~120 MB
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks

# Ollama model names (make sure they are pulled: `ollama pull qwen2.5:3b` / `ollama pull mistral`)
QWEN_MODEL    = "qwen2.5:3b"
MISTRAL_MODEL = "mistral"

# ──────────────────────────────────────────────
# 2. EMBEDDING HELPER
# ──────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    return model.encode(texts, show_progress_bar=False).tolist()


# ──────────────────────────────────────────────
# 3. CHROMADB HELPER
# ──────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ──────────────────────────────────────────────
# 4. PDF READING
# ──────────────────────────────────────────────
def read_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)
    full_text = "\n".join(pages_text)
    print(f"[RAG] Extracted {len(full_text):,} characters from {len(reader.pages)} pages.")
    return full_text


# ──────────────────────────────────────────────
# 5. CHUNKING
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    Tries to break at sentence boundaries ('. ', '! ', '? ', '\n').
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to extend to a sentence boundary within the next 200 chars
        if end < length:
            boundary_search = text[end : end + 200]
            match = re.search(r"[.!?\n]", boundary_search)
            if match:
                end = end + match.start() + 1  # include the punctuation

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Next start respects overlap
        next_start = end - overlap
        start = next_start if next_start > start else start + 1

    print(f"[RAG] Created {len(chunks)} chunks (size≈{chunk_size}, overlap={overlap}).")
    return chunks


# ──────────────────────────────────────────────
# 6. INGEST: PDF → ChromaDB
# ──────────────────────────────────────────────
def ingest_pdf(pdf_path: str, source_name: Optional[str] = None) -> int:
    """
    Read PDF, chunk, embed, and store in ChromaDB.
    Returns the number of chunks stored.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source_name = source_name or os.path.basename(pdf_path)
    raw_text = read_pdf(pdf_path)
    chunks = chunk_text(raw_text)

    print(f"[RAG] Embedding {len(chunks)} chunks …")
    embeddings = embed_texts(chunks)

    collection = get_collection()

    # Build IDs & metadata
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=chunks[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print(f"[RAG] ✅ Stored {len(chunks)} chunks from '{source_name}' into ChromaDB.")
    return len(chunks)


# ──────────────────────────────────────────────
# 7. RETRIEVAL
# ──────────────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> List[dict]:
    """
    Embed the query and return top_k most relevant chunks from ChromaDB.
    """
    query_embedding = embed_texts([query])[0]
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


# ──────────────────────────────────────────────
# 8. LLM ANSWER GENERATION via Ollama
# ──────────────────────────────────────────────
def build_prompt(question: str, context_chunks: List[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        [f"[Chunk {i+1}] {c['text']}" for i, c in enumerate(context_chunks)]
    )
    prompt = (
        "Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.\n"
        "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ điều đó.\n\n"
        f"=== NGỮ CẢNH ===\n{context_text}\n\n"
        f"=== CÂU HỎI ===\n{question}\n\n"
        "=== TRẢ LỜI ==="
    )
    return prompt


def ask_ollama(question: str, model: str = QWEN_MODEL, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve top_k relevant chunks from ChromaDB
      2. Build prompt
      3. Call Ollama LLM
    Returns dict with keys: answer, model, chunks_used
    """
    import requests

    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {"answer": "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", "model": model, "chunks_used": []}

    prompt = build_prompt(question, hits)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        answer = (
            "[Lỗi] Không thể kết nối tới Ollama. "
            "Hãy chắc chắn Ollama đang chạy: `ollama serve` "
            f"và model đã được tải: `ollama pull {model}`"
        )
    except Exception as e:
        answer = f"[Lỗi] {str(e)}"

    return {"answer": answer, "model": model, "chunks_used": hits}


async def ask_ollama_async(question: str, model: str = QWEN_MODEL, top_k: int = 5) -> dict:
    """Async wrapper for use with discord bot."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: ask_ollama(question, model, top_k))
