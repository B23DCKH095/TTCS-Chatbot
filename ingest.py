"""
ingest.py – CLI tool to read a PDF and store it into ChromaDB.

Usage:
    python ingest.py path/to/document.pdf
    python ingest.py path/to/document.pdf --source "Tên tài liệu"
    python ingest.py path/to/document.pdf --chunk-size 600 --overlap 120
"""

import argparse
import sys
from rag_pipeline import ingest_pdf, CHUNK_SIZE, CHUNK_OVERLAP
import rag_pipeline as rp


def main():
    parser = argparse.ArgumentParser(
        description="Đọc PDF, chia nhỏ (chunking) và lưu vào ChromaDB"
    )
    parser.add_argument("pdf", help="Đường dẫn tới file PDF cần xử lý")
    parser.add_argument(
        "--source", "-s", default=None,
        help="Tên nguồn tài liệu (mặc định: tên file)",
    )
    parser.add_argument(
        "--chunk-size", "-c", type=int, default=CHUNK_SIZE,
        help=f"Kích thước mỗi chunk (ký tự, mặc định: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap", "-o", type=int, default=CHUNK_OVERLAP,
        help=f"Độ chồng lấp giữa các chunk (mặc định: {CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    # Override globals if user specified custom values
    rp.CHUNK_SIZE = args.chunk_size
    rp.CHUNK_OVERLAP = args.overlap

    try:
        num_chunks = ingest_pdf(args.pdf, source_name=args.source)
        print(f"\n✅ Hoàn tất! Đã lưu {num_chunks} chunks vào ChromaDB.")
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
