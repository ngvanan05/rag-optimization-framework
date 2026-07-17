"""
Stage 1 - Vector DB
Split text into chunks (Recursive) and store in Qdrant.
Merged logic from: document_processing.py + vector_database.py
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ---------------------------------------------------------------------------
# Text Splitter
# ---------------------------------------------------------------------------

class TextSplitter:
    """
    Split text using Recursive Character Splitting strategy.
    Avoids breaking sentences/paragraphs thanks to a prioritized separator list.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, documents: List) -> List:
        return self.splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Vector Database (Qdrant)
# ---------------------------------------------------------------------------

class VectorDB:
    """
    Manage vector store using Qdrant.
    Supports creating a new collection from documents or connecting to an existing one.
    """

    def __init__(
        self,
        documents=None,
        embedding=None,
        collection_name: str = "aio_documents",
        qdrant_url: str = "http://localhost:6333",
        api_key: str = None,
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embedding = embedding
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        self.vector_store = self._build_db(documents)

    def _build_db(self, documents):
        if documents and len(documents) > 0:
            print(f"-> Initializing Qdrant with {len(documents)} chunks...")
            return QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embedding,
                url=self.qdrant_url,
                collection_name=self.collection_name,
                force_recreate=True,  # Đảm bảo dữ liệu mới nhất khi dev
            )
        else:
            print(f"-> Connecting to existing Qdrant collection: {self.collection_name}")
            return QdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                url=self.qdrant_url,
                collection_name=self.collection_name,
            )

    def get_retriever(self, search_kwargs: dict = None):
        """Return a similarity search retriever from Qdrant."""
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )


if __name__ == "__main__":
    from phase2_baseline.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_URL, COLLECTION_NAME
    from phase2_baseline.stage1_indexing.loader import DocumentLoader
    from phase2_baseline.models import get_embeddings

    loader   = DocumentLoader()
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs     = loader.load_dir(DATA_DIR)
    chunks   = splitter.split(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = get_embeddings()
    vdb = VectorDB(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
    )
    print("VectorDB ready!")
