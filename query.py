"""
query.py – CLI tool to ask questions against the ChromaDB knowledge base.

Usage:
    python query.py "Câu hỏi của bạn?"
    python query.py "Câu hỏi?" --model mistral
    python query.py "Câu hỏi?" --model qwen2.5:3b --top-k 3 --show-chunks
"""

import argparse
from rag_pipeline import ask_ollama, retrieve, QWEN_MODEL, MISTRAL_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Đặt câu hỏi và nhận trả lời từ ChromaDB + LLM (Qwen / Mistral)"
    )
    parser.add_argument("question", help="Câu hỏi cần trả lời")
    parser.add_argument(
        "--model", "-m", default=QWEN_MODEL,
        choices=[QWEN_MODEL, MISTRAL_MODEL, "qwen2.5:3b", "mistral"],
        help=f"LLM model Ollama sử dụng (mặc định: {QWEN_MODEL})",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5,
        help="Số lượng chunks liên quan nhất truy vấn từ ChromaDB (mặc định: 5)",
    )
    parser.add_argument(
        "--show-chunks", action="store_true",
        help="Hiển thị các chunks liên quan tìm được từ ChromaDB",
    )
    parser.add_argument(
        "--retrieve-only", action="store_true",
        help="Chỉ tìm chunks liên quan, không gọi LLM",
    )

    args = parser.parse_args()

    print(f"\n🔍 Câu hỏi: {args.question}")
    print(f"   Model  : {args.model}  |  Top-K: {args.top_k}\n")

    if args.retrieve_only:
        hits = retrieve(args.question, top_k=args.top_k)
        print(f"📄 {len(hits)} đoạn văn liên quan nhất từ ChromaDB:\n")
        for i, h in enumerate(hits, 1):
            src = h["metadata"].get("source", "?")
            idx = h["metadata"].get("chunk_index", "?")
            dist = h["distance"]
            print(f"─── Chunk {i}  [nguồn: {src}  idx: {idx}  khoảng cách: {dist:.4f}] ───")
            print(h["text"])
            print()
        return

    result = ask_ollama(args.question, model=args.model, top_k=args.top_k)

    if args.show_chunks:
        print(f"📄 {len(result['chunks_used'])} đoạn văn liên quan nhất từ ChromaDB:\n")
        for i, h in enumerate(result["chunks_used"], 1):
            src = h["metadata"].get("source", "?")
            idx = h["metadata"].get("chunk_index", "?")
            dist = h["distance"]
            print(f"─── Chunk {i}  [nguồn: {src}  idx: {idx}  khoảng cách: {dist:.4f}] ───")
            print(h["text"])
            print()

    print(f"🤖 Trả lời ({result['model']}):\n")
    print(result["answer"])
    print()


if __name__ == "__main__":
    main()
