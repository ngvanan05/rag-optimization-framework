"""
Phase 1 — Step 1: Convert PDF → Markdown + Load DOCX
Chạy độc lập để test:
    python step1_convert.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pymupdf4llm
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader

from config import DATA_DIR, MARKDOWN_DIR


def convert_pdfs_to_markdown(pdf_dir: str = DATA_DIR, output_dir: str = MARKDOWN_DIR) -> str:
    """Chuyển toàn bộ PDF sang Markdown, lưu vào output_dir."""
    import signal

    pdf_path    = Path(pdf_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Tìm thấy {len(pdf_files)} file PDF")

    converted = 0
    for pdf_file in pdf_files:
        # Bỏ qua nếu đã convert rồi
        md_file = output_path / f"{pdf_file.stem}.md"
        if md_file.exists():
            print(f"  Skip (đã có): {pdf_file.name}")
            converted += 1
            continue
        try:
            print(f"  Converting: {pdf_file.name}")
            md_text = pymupdf4llm.to_markdown(str(pdf_file))
            md_file.write_text(md_text, encoding="utf-8")
            converted += 1
        except Exception as e:
            print(f"  Lỗi {pdf_file.name}: {e}")

    print(f"Đã convert {converted}/{len(pdf_files)} PDF → {output_path}\n")
    return str(output_path)


def load_docx_documents(data_dir: str = DATA_DIR):
    """Load toàn bộ .docx từ data_dir."""
    docx_files = list(Path(data_dir).glob("*.docx"))
    print(f"Tìm thấy {len(docx_files)} file DOCX")

    docs = []
    for docx_file in docx_files:
        try:
            loaded = Docx2txtLoader(str(docx_file)).load()
            docs.extend(loaded)
        except Exception as e:
            print(f"  Lỗi {docx_file.name}: {e}")

    print(f"Đã load {len(docs)} DOCX documents\n")
    return docs


def load_markdown_documents(md_dir: str = MARKDOWN_DIR):
    """Load toàn bộ .md từ md_dir."""
    print(f"Loading Markdown từ: {md_dir}")
    loader = DirectoryLoader(
        md_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Đã load {len(docs)} Markdown documents")
    return docs


def load_all_documents(md_dir: str = MARKDOWN_DIR, data_dir: str = DATA_DIR):
    """Gộp Markdown (từ PDF) + DOCX."""
    md_docs   = load_markdown_documents(md_dir)
    docx_docs = load_docx_documents(data_dir)
    all_docs  = md_docs + docx_docs
    print(f"Tổng: {len(all_docs)} documents (PDF→MD: {len(md_docs)} | DOCX: {len(docx_docs)})\n")
    return all_docs


if __name__ == "__main__":
    convert_pdfs_to_markdown()
    docs = load_all_documents()
    print(f"Sẵn sàng: {len(docs)} documents")
