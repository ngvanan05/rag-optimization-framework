"""
Stage 3 - RAG Chain
Combines Context + LLM to generate final answers.
Merged logic from: rag_pipeline.py (BatchRAG + FocusedAnswerParser)
"""
import re
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from phase2_baseline.async_utils import run_prompts_in_batches
from phase2_baseline.stage3_generation.prompt_templates import BASELINE_PROMPT


# ---------------------------------------------------------------------------
# Answer Parser
# ---------------------------------------------------------------------------

class FocusedAnswerParser(StrOutputParser):
    """
    Clean the LLM answer output.
    Removes extraneous content before [TRẢ LỜI]: and leading markdown characters.
    """

    def parse(self, text: str) -> str:
        text = (text or "").strip()

        # Cắt phần thừa trước [TRẢ LỜI]: nếu LLM sinh ra
        if "[TRẢ LỜI]:" in text:
            text = text.split("[TRẢ LỜI]:")[-1].strip()

        # Xóa gạch đầu dòng, dấu chấm tròn markdown
        text = re.sub(r"^\s*[\u2022\-\*]\s*", "", text, flags=re.MULTILINE)

        # Gộp nhiều dòng trống thành một khoảng trắng
        text = re.sub(r"\n+", " ", text).strip()

        return text


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

class RAGChain:
    """
    Combine SearchEngine + LLM to generate answers.
    Supports single-question processing (get_chain) and batch processing (batch_answer).
    """

    def __init__(self, llm, batch_size: int = 32, prompt=BASELINE_PROMPT):
        self.llm = llm
        self.batch_size = batch_size
        self.prompt = prompt
        self.parser = FocusedAnswerParser()

    def get_chain(self, retriever):
        """
        Create LCEL chain for single question processing.
        """
        from phase2_baseline.stage2_retrieval.search_engine import SearchEngine
        engine = SearchEngine(retriever)

        def format_docs(docs):
            return engine.format_docs(docs)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.parser
        )

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Batch inference via async gather — uses a separate event loop per batch."""
        responses = run_prompts_in_batches(self.llm, prompts, self.batch_size, desc="Generating answers")
        return [self.parser.parse(response.content) for response in responses]

    def batch_answer(self, retrieved_data: List[dict]) -> List[dict]:
        """
        Receive SearchEngine results and generate batch answers.
        Returns a list of dicts with answer + contexts.
        """
        prompts = [
            self.prompt.format(
                context=data["formatted_context"],
                question=data["question"]
            )
            for data in retrieved_data
        ]

        answers = self.batch_generate(prompts)

        return [
            {"answer": answer, "contexts": data["contexts"]}
            for data, answer in zip(retrieved_data, answers)
        ]


if __name__ == "__main__":
    print("--- RAGChain and FocusedAnswerParser ready ---")
