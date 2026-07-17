"""
Phase 3 - v3 Hybrid Search + RRF
Combines Dense (Qdrant) + Sparse (BM25) via Reciprocal Rank Fusion.

Replaces the baseline SearchEngine with a hybrid retriever,
keeping Qdrant as the vector store for consistency across the project.
"""
import numpy as np
from typing import List, Optional
from collections import defaultdict

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Import config từ Phase 2
from phase2_baseline.config import (
    QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL_NAME
)

# --- CẤU HÌNH ---
HYBRID_K = 7    # Số doc lấy từ mỗi retriever trước khi fuse
RRF_K    = 40   # Hằng số RRF (thường dùng 40-60)


class HybridVectorDB:
    """
    Manages a dual index: Qdrant (dense) + BM25 (sparse).
    Interface matches the baseline VectorDB for easy swapping.
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embedding=None,
        collection_name: str = COLLECTION_NAME,
        qdrant_url: str = QDRANT_URL,
        api_key: str = None,
    ):
        self.collection_name = collection_name
        self.qdrant_url      = qdrant_url
        self.embedding       = embedding
        self.documents       = documents or []

        print("-> Initializing Qdrant (dense index)...")
        self.client       = QdrantClient(url=qdrant_url, api_key=api_key)
        self.vector_store = self._build_qdrant(documents)

        print("-> Building BM25 (sparse index)...")
        self.bm25 = self._build_bm25(self.documents)
        print("-> HybridVectorDB ready.")

    def _build_qdrant(self, documents: Optional[List[Document]]):
        if documents:
            return QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embedding,
                url=self.qdrant_url,
                collection_name=self.collection_name,
                force_recreate=True,
            )
        else:
            return QdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                url=self.qdrant_url,
                collection_name=self.collection_name,
            )

    def _build_bm25(self, documents: List[Document]):
        if not documents:
            print("-> No documents to build BM25.")
            return None
        tokenized = [
            word_tokenize(doc.page_content.lower())
            for doc in documents
        ]
        return BM25Okapi(tokenized)

    def get_retriever(self, search_kwargs: dict = None) -> "HybridRetriever":
        k = (search_kwargs or {}).get("k", HYBRID_K)
        return HybridRetriever(
            vector_store=self.vector_store,
            bm25=self.bm25,
            documents=self.documents,
            k=k,
            rrf_k=RRF_K,
        )

    def save_bm25(self, path: str):
        """Saves BM25 index + documents to a pickle file for fast reloading."""
        import pickle, pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        print(f"-> BM25 index saved: {path}")

    @classmethod
    def load_bm25(cls, path: str):
        """Loads BM25 index from a pickle file, returns (bm25, documents)."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"-> BM25 index loaded: {path}")
        return data["bm25"], data["documents"]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: Dense (Qdrant) + Sparse (BM25) → RRF fusion.
    """
    vector_store: QdrantVectorStore
    bm25: object
    documents: List[Document]
    k: int = HYBRID_K
    rrf_k: int = RRF_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:

        results_list = []

        # 1. Dense search — Qdrant
        dense_results = self.vector_store.similarity_search(query, k=self.k)
        results_list.append(dense_results)

        # 2. Sparse search — BM25
        if self.bm25 is not None:
            tokenized_query = word_tokenize(query.lower())
            bm25_scores     = self.bm25.get_scores(tokenized_query)
            top_indices     = np.argsort(bm25_scores)[::-1][: self.k]
            bm25_results    = [self.documents[i] for i in top_indices]
            results_list.append(bm25_results)

        # 3. Reciprocal Rank Fusion
        return self._rrf(results_list)[: self.k]

    def _rrf(self, results_list: List[List[Document]]) -> List[Document]:
        """Merges multiple ranked lists using RRF: score = Σ 1/(k + rank)."""
        rrf_scores   = defaultdict(float)
        doc_map      = {}

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.page_content          # dùng content làm key
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
                doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]


if __name__ == "__main__":
    from phase2_baseline.stage1_indexing.loader import DocumentLoader
    from phase2_baseline.stage1_indexing.vector_db import TextSplitter
    from phase2_baseline.models import get_embeddings
    from phase2_baseline.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

    loader    = DocumentLoader()
    splitter  = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs      = loader.load_dir(DATA_DIR)
    chunks    = splitter.split(docs)
    embedding = get_embeddings()

    hdb       = HybridVectorDB(documents=chunks, embedding=embedding)
    retriever = hdb.get_retriever(search_kwargs={"k": 5})

    results = retriever.invoke("Sinh viên khóa 51 cần đăng ký những hoạt động gì?")
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:120]}")
