"""
Phase 3 - v1 Semantic Chunking
Splits documents by semantic meaning instead of fixed character count.

Uses BGE-M3 (GPU FP16) to compute sentence similarity and
cuts chunks at points where semantics shift abruptly.
"""
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from underthesea import sent_tokenize
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Import config từ Phase 2
from phase2_baseline.config import EMBEDDING_MODEL_NAME

# --- CẤU HÌNH ---
SEMANTIC_BREAKPOINT_THRESHOLD = 0.5   # Ngưỡng similarity để cắt chunk
MIN_CHUNK_SIZE  = 600                  # Chunk tối thiểu (ký tự)
MAX_CHUNK_SIZE  = 1024                 # Chunk tối đa (ký tự)
CHUNK_OVERLAP   = 128                  # Overlap giữa các chunk


class SemanticChunker:
    """
    Splits text by semantic meaning using BGE-M3 on GPU.
    Replaces the baseline RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        breakpoint_threshold: float = SEMANTIC_BREAKPOINT_THRESHOLD,
        overlap_size: int = CHUNK_OVERLAP,
    ):
        self.breakpoint_threshold = breakpoint_threshold
        self.overlap_size = overlap_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"-> Loading SemanticChunker embedding: {model_name} on {device}...")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
        )
        print("-> SemanticChunker ready.")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits Vietnamese text into sentences using underthesea."""
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 20]

    def _cosine_similarity(self, emb1, emb2) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))

    def _chunk_by_semantic_similarity(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        # Embed toàn bộ câu một lần (batch)
        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity  = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            chunk_text  = " ".join(current_chunk)
            chunk_len   = len(chunk_text)

            if chunk_len >= MAX_CHUNK_SIZE:
                # Chunk đã đủ lớn → cắt bắt buộc
                chunks.append(chunk_text)
                current_chunk = [sentences[i]]
            elif similarity >= self.breakpoint_threshold or chunk_len < MIN_CHUNK_SIZE:
                # Ngữ nghĩa liên tục hoặc chunk còn quá nhỏ → tiếp tục gộp
                current_chunk.append(sentences[i])
            else:
                # Ngữ nghĩa thay đổi → cắt chunk
                chunks.append(chunk_text)
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Accept a list of LangChain Documents and return semantically split chunks.
        Interface matches the baseline TextSplitter for easy swapping.
        """
        all_chunks = []

        for doc in tqdm(documents, desc="Semantic chunking"):
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue

            chunks = self._chunk_by_semantic_similarity(sentences)

            for idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Thêm overlap từ chunk trước
                if idx > 0 and self.overlap_size > 0:
                    prev = chunks[idx - 1]
                    if len(prev) >= self.overlap_size:
                        chunk_text = prev[-self.overlap_size:] + " " + chunk_text

                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy(),
                ))

        return all_chunks

    def free_gpu(self):
        """Free the model from GPU after chunking to release VRAM for embeddings."""
        if hasattr(self, "model"):
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("-> SemanticChunker: GPU memory freed.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from phase2_baseline.stage1_indexing.loader import DocumentLoader
    from phase2_baseline.config import DATA_DIR

    loader  = DocumentLoader()
    docs    = loader.load_dir(DATA_DIR)
    chunker = SemanticChunker()
    chunks  = chunker.split(docs)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk sample:\n{chunks[0].page_content[:300]}")
