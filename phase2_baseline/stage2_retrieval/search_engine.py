"""
Stage 2 - Search Engine
Handles context retrieval logic from Qdrant.
Extracted from: rag_pipeline.py (batch_retrieve section)
"""
from tqdm import tqdm
from typing import List


class SearchEngine:
    """
    Retrieve relevant text passages from Vector DB for a list of questions.
    Supports deduplication and filtering of overly short chunks.
    """

    MIN_CONTENT_LENGTH = 40  # Bỏ qua chunk ngắn hơn ngưỡng này

    def __init__(self, retriever):
        self.retriever = retriever

    def format_docs(self, docs) -> str:
        """
        Merge chunks into a single context string.
        Removes duplicate and overly short chunks.
        """
        formatted = []
        seen = set()
        for doc in docs:
            content = (doc.page_content or "").strip()
            if content and len(content) > self.MIN_CONTENT_LENGTH and content not in seen:
                formatted.append(content)
                seen.add(content)
        return "\n\n".join(formatted)

    def retrieve_one(self, question: str) -> dict:
        """Retrieve context for a single question."""
        docs = self.retriever.invoke(question)
        return {
            "question": question,
            "contexts": [(doc.page_content or "") for doc in docs],
            "formatted_context": self.format_docs(docs),
        }

    def retrieve_batch(self, questions: List[str]) -> List[dict]:
        """Retrieve context for a list of questions."""
        results = []
        for question in tqdm(questions, desc="Retrieving"):
            results.append(self.retrieve_one(question))
        return results


if __name__ == "__main__":
    print("--- SearchEngine ready ---")
