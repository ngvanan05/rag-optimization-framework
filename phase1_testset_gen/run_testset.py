"""
Phase 1 — Entry Point
Chạy toàn bộ pipeline: PDF/DOCX → Markdown → RAGAS Testset → JSON

Chạy:
    cd phase1_testset_gen
    python run_testset.py
"""
import os
import sys

# Patch trước mọi import ragas
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# Fix encoding Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from phase1_testset_gen.config import DATA_DIR, MARKDOWN_DIR, TESTSET_PATH, TESTSET_SIZE
from phase1_testset_gen.step1_convert import convert_pdfs_to_markdown, load_all_documents
from phase1_testset_gen.step2_models import get_ragas_wrappers
from phase1_testset_gen.step3_generate import build_generator, generate_and_save


def main():
    print("=" * 50)
    print("  PHASE 1 — TESTSET GENERATION")
    print("=" * 50)

    # Step 1 — Convert & Load (không cần ragas/torch)
    print("\n[1/3] Convert PDF → Markdown + Load DOCX...")
    convert_pdfs_to_markdown(DATA_DIR, MARKDOWN_DIR)
    docs = load_all_documents(MARKDOWN_DIR, DATA_DIR)

    if not docs:
        raise RuntimeError(f"Không tìm thấy document nào trong {DATA_DIR}")

    # Step 2 — Khởi tạo models
    print("[2/3] Khởi tạo LLM + Embedding...")
    llm_wrapper, emb_wrapper = get_ragas_wrappers()

    # Step 3 — Sinh testset
    print("[3/3] Sinh testset...")
    generator = build_generator(llm_wrapper, emb_wrapper)
    generate_and_save(generator, docs, TESTSET_SIZE, TESTSET_PATH)

    print("\n✅ Hoàn tất! Testset lưu tại:", TESTSET_PATH)


if __name__ == "__main__":
    main()
