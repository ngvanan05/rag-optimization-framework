"""
Stage 3 - RAG Chain
Combines Context + LLM to generate final answers.
Merged logic from: rag_pipeline.py (BatchRAG + FocusedAnswerParser)
"""
import re
import asyncio
from tqdm import tqdm
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stage3_generation.prompt_templates import BASELINE_PROMPT


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
        from stage2_retrieval.search_engine import SearchEngine
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
        async def _run_batch(batch_prompts):
            messages_batch = [[HumanMessage(content=p)] for p in batch_prompts]
            tasks = [self.llm.ainvoke(msgs) for msgs in messages_batch]
            return await asyncio.gather(*tasks)

        all_answers = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating answers"):
            batch = prompts[i:i + self.batch_size]
            loop  = asyncio.new_event_loop()
            try:
                results = loop.run_until_complete(_run_batch(batch))
            finally:
                loop.close()
            for result in results:
                all_answers.append(self.parser.parse(result.content))
        return all_answers

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
