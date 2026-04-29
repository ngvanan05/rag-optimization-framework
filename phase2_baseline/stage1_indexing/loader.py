"""
Stage 1 - Loader
Read all PDF/DOCX files from the data directory and clean Vietnamese text.
Merged logic from: data_loader.py + text_preprocessing.py
"""
import re
import glob
import unicodedata
from tqdm import tqdm
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

def clean_vietnamese_text(text: str) -> str:
    """
    Normalize Vietnamese text to NFC form and remove noise characters.
    Optimized for text extracted from PDF/DOCX files.
    """
    # 1. Chuẩn hóa Unicode NFC (chuẩn tiếng Việt dựng sẵn)
    text = unicodedata.normalize("NFC", text)

    # 2. Loại bỏ ký tự điều khiển (giữ lại \n và \t)
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("C") or char in "\n\t"
    )

    # 3. Gộp nhiều khoảng trắng/tab thành một (giữ nguyên \n)
    text = re.sub(r"[^\S\n]+", " ", text)

    # 4. Gộp nhiều dòng trống liên tiếp thành một
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Document Loader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """
    Load documents from a directory, supporting .pdf and .docx formats.
    Automatically cleans Vietnamese text after loading.
    """

    def load_pdf(self, pdf_file: str) -> List:
        docs = PyPDFLoader(pdf_file, extract_images=True).load()
        for doc in docs:
            doc.page_content = clean_vietnamese_text(doc.page_content)
        return docs

    def load_docx(self, docx_file: str) -> List:
        docs = Docx2txtLoader(docx_file).load()
        for doc in docs:
            doc.page_content = clean_vietnamese_text(doc.page_content)
        return docs

    def load_dir(self, dir_path: str) -> List:
        pdf_files  = glob.glob(f"{dir_path}/*.pdf")
        docx_files = glob.glob(f"{dir_path}/*.docx")
        all_files  = pdf_files + docx_files

        if not all_files:
            raise ValueError(
                f"No files found (.pdf/.docx) in directory: {dir_path}"
            )

        print(f"Found: {len(pdf_files)} PDF | {len(docx_files)} DOCX")

        all_docs = []
        for file_path in tqdm(all_files, desc="Loading documents"):
            try:
                ext = file_path.lower().rsplit(".", 1)[-1]
                if ext == "pdf":
                    all_docs.extend(self.load_pdf(file_path))
                elif ext == "docx":
                    all_docs.extend(self.load_docx(file_path))
            except Exception as e:
                print(f"\nError reading file {file_path}: {e}")

        return all_docs


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from config import DATA_DIR

    loader = DocumentLoader()
    docs = loader.load_dir(DATA_DIR)
    print(f"\nTotal: {len(docs)} pages/documents loaded.")
    print("\nFirst content sample:")
    print(docs[0].page_content[:300])
