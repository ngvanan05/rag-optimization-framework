"""
Phase 3 - v2 Query Decomposition
Breaks complex questions into sub-questions, retrieves for each,
merges the contexts, then generates an answer.

Replaces SearchEngine + RAGChain from baseline.
"""
import re
from typing import List
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from phase2_baseline.config import BATCH_SIZE
from phase2_baseline.stage3_generation.rag_chain import FocusedAnswerParser


class DecompositionQueryTransformer:
    """Uses an LLM to decompose complex questions into up to 3 sub-questions."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "Hãy phân tách câu hỏi sau thành các câu hỏi con đơn giản hơn.\n"
            "Mỗi câu hỏi con nên tập trung vào một khía cạnh cụ thể của câu hỏi gốc.\n"
            "Trả về tối đa 3 câu hỏi con, mỗi câu trên một dòng.\n"
            "Nếu câu hỏi đã đơn giản, chỉ trả về chính câu hỏi đó.\n\n"
            "Câu hỏi gốc: {question}\n\n"
            "Các câu hỏi con:"
        )

    def decompose(self, question: str) -> List[str]:
        """Breaks a question into up to 3 sub-questions using the LLM."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(
                self.llm.ainvoke(self.prompt.format(question=question))
            )
        finally:
            loop.close()
        raw = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        cleaned = []
        for q in raw:
            q = re.sub(r"^\d+\.\s*", "", q).strip()
            if q:
                cleaned.append(q)
        return cleaned[:3]


class DecompositionBatchRAG:
    """
    RAG pipeline with integrated Query Decomposition.
    Interface matches the baseline RAGChain (answer_with_contexts_batch).
    """

    def __init__(self, llm, batch_size: int = BATCH_SIZE):
        self.llm          = llm
        self.batch_size   = batch_size
        self.decomposer   = DecompositionQueryTransformer(llm)
        self.answer_parser = FocusedAnswerParser()

        self.prompt = PromptTemplate.from_template(
            "Bạn là trợ lý AI phân tích tài liệu tiếng Việt.\n\n"
            "[TÀI LIỆU]:\n{context}\n\n"
            "[CÂU HỎI GỐC]:\n{question}\n\n"
            "[CÁC CÂU HỎI CON ĐÃ PHÂN TÁCH]:\n{sub_questions}\n\n"
            "Hãy trả lời câu hỏi gốc dựa trên tài liệu, sử dụng thông tin từ các câu hỏi con làm hướng dẫn.\n"
            "Nếu tài liệu không có thông tin, nói rõ \"Không có thông tin\".\n"
            "[TRẢ LỜI]:"
        )

    def _format_docs(self, docs) -> str:
        return "\n\n".join(
            (doc.page_content or "").strip()
            for doc in docs
            if (doc.page_content or "").strip()
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def decompose_and_retrieve(self, question: str, retriever) -> dict:
        """Decompose question → retrieve for each sub-question + original → dedup."""
        sub_questions = self.decomposer.decompose(question)

        all_docs      = []
        seen_contents = set()

        for sub_q in sub_questions:
            for doc in retriever.invoke(sub_q):
                if doc.page_content not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)

        for doc in retriever.invoke(question):
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)

        return {
            "question":          question,
            "sub_questions":     sub_questions,
            "contexts":          [doc.page_content for doc in all_docs],
            "formatted_context": self._format_docs(all_docs),
        }

    def batch_retrieve(self, questions: List[str], retriever) -> List[dict]:
        return [
            self.decompose_and_retrieve(q, retriever)
            for q in tqdm(questions, desc="Decomposing & Retrieving")
        ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Batch inference via async gather — uses a separate event loop per batch."""
        import asyncio

        async def _run_batch(batch_prompts):
            messages = [[HumanMessage(content=p)] for p in batch_prompts]
            return await asyncio.gather(*[self.llm.ainvoke(m) for m in messages])

        all_answers = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating (Decomposition)"):
            batch = prompts[i : i + self.batch_size]
            loop  = asyncio.new_event_loop()
            try:
                responses = loop.run_until_complete(_run_batch(batch))
            finally:
                loop.close()
            for resp in responses:
                all_answers.append(self.answer_parser.parse(resp.content))
        return all_answers

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def answer_with_contexts_batch(self, questions: List[str], retriever) -> List[dict]:
        """
        Full pipeline: decompose → retrieve → generate.
        Output format compatible with Ragas: answer + contexts.
        """
        retrieved_data = self.batch_retrieve(questions, retriever)

        prompts = [
            self.prompt.format(
                context=data["formatted_context"],
                question=data["question"],
                sub_questions="\n".join(f"- {sq}" for sq in data["sub_questions"]),
            )
            for data in retrieved_data
        ]

        answers = self.batch_generate(prompts)

        return [
            {
                "answer":        answer,
                "contexts":      data["contexts"],
                "sub_questions": data["sub_questions"],
            }
            for data, answer in zip(retrieved_data, answers)
        ]
