"""
Phase 3 - v5 CoT + Citation Engine
Generates answers with step-by-step reasoning (Chain-of-Thought) and source citations.

Replaces the baseline RAGChain:
- Context is labeled [Doc 1], [Doc 2]... so the LLM can cite sources
- Output includes <think>...</think> (reasoning) + answer with [Doc X] citations
- Supports true batch inference via llm.batch()
"""
import re
from typing import List, Dict
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Import config từ Phase 2
from phase2_baseline.config import BATCH_SIZE

# --- PROMPT: CoT + Citation (dùng cho batch inference / Ragas) ---
COT_PROMPT_TEMPLATE = """Bạn là hệ thống chuyên gia hỏi đáp dựa trên tài liệu nội bộ.

[NGỮ CẢNH]:
{context}

[CÂU HỎI]:
{question}

[YÊU CẦU BẮT BUỘC]:
1. SUY LUẬN NỘI BỘ: Viết suy luận từng bước vào thẻ <think>...</think>. Phải đối chiếu câu hỏi với từng [Doc X] để tìm bằng chứng.
2. NGUYÊN TẮC THÔNG TIN: Chỉ dùng thông tin trong [NGỮ CẢNH]. Không thêm kiến thức ngoài.
3. TRƯỜNG HỢP KHÔNG THẤY: Nếu không có câu trả lời, suy luận lý do trong <think> và kết luận: "Tôi không biết".
4. TRÍCH DẪN (CITATION): Mỗi thông tin đưa ra phải kèm ký hiệu [Doc X] ngay sau ý đó.
5. NGÔN NGỮ: Tiếng Việt, văn phong tự nhiên, rõ ràng, dễ đọc.
6. TRÌNH BÀY: Ưu tiên chia đoạn hoặc dùng bullet nếu có nhiều ý. Không copy nguyên văn — tóm tắt và diễn đạt lại như một bài viết ngắn.
7. NHẤN MẠNH: Làm nổi bật các thông tin quan trọng như thời gian, địa điểm, đường link.

[ĐỊNH DẠNG ĐẦU RA]:
<think>
(Phần suy luận logic của hệ thống)
</think>

Câu trả lời:
(Nội dung câu trả lời có trích dẫn)"""

# --- PROMPT: Citation only (dùng cho app — không có CoT, tiết kiệm token) ---
CITATION_PROMPT_TEMPLATE = """Bạn là hệ thống chuyên gia hỏi đáp dựa trên tài liệu nội bộ.

[NGỮ CẢNH]:
{context}

[CÂU HỎI]:
{question}

[YÊU CẦU BẮT BUỘC]:
1. NGUYÊN TẮC THÔNG TIN: Chỉ dùng thông tin trong [NGỮ CẢNH]. Không thêm kiến thức ngoài.
2. TRƯỜNG HỢP KHÔNG THẤY: Nếu không có câu trả lời, kết luận: "Tôi không biết".
3. TRÍCH DẪN (CITATION): Mỗi thông tin đưa ra phải kèm ký hiệu [Doc X] ngay sau ý đó.
4. NGÔN NGỮ: Tiếng Việt, văn phong tự nhiên, rõ ràng, dễ đọc.
5. TRÌNH BÀY: Ưu tiên chia đoạn hoặc dùng bullet nếu có nhiều ý. Không copy nguyên văn — tóm tắt và diễn đạt lại như một bài viết ngắn.
6. NHẤN MẠNH: Làm nổi bật các thông tin quan trọng như thời gian, địa điểm, đường link.

Câu trả lời:"""


class CotCitationEngine:
    """
    Engine for generating answers with CoT reasoning + Citation.
    Interface compatible with the baseline RAGChain for easy swapping.
    """

    def __init__(self, llm, batch_size: int = BATCH_SIZE):
        self.llm          = llm
        self.batch_size   = batch_size
        self.prompt       = PromptTemplate.from_template(COT_PROMPT_TEMPLATE)
        self.cite_prompt  = PromptTemplate.from_template(CITATION_PROMPT_TEMPLATE)

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def _format_docs_for_citation(self, docs: List) -> str:
        """Label [Doc 1], [Doc 2]... so LLM has citation targets."""
        parts = []
        for i, doc in enumerate(docs, start=1):
            content = (doc.page_content if hasattr(doc, "page_content") else str(doc)).strip()
            parts.append(f"[Doc {i}]: {content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Citation-only inference (dùng cho app — không CoT, tiết kiệm token)
    # ------------------------------------------------------------------

    def cite(self, question: str, docs: List) -> str:
        """Sinh câu trả lời có citation, không có CoT. Dùng cho app."""
        import asyncio
        formatted_prompt = self.cite_prompt.format(
            context=self._format_docs_for_citation(docs),
            question=question,
        )
        response = asyncio.run(self.llm.ainvoke(formatted_prompt))
        answer = (response.content or "").strip()
        answer = re.sub(r"^(Câu trả lời:|Trả lời:)\s*", "", answer, flags=re.IGNORECASE).strip()
        return answer

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, text: str) -> Dict[str, str]:
        """Extract <think> (reasoning) and final answer."""
        text = (text or "").strip()

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thought     = think_match.group(1).strip() if think_match else ""

        # Xóa block <think> để lấy phần trả lời
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Xóa tiền tố "Câu trả lời:" nếu LLM thêm vào
        answer = re.sub(r"^(Câu trả lời:|Trả lời:)\s*", "", answer, flags=re.IGNORECASE).strip()

        return {
            "thought_process": thought,
            "final_answer":    answer,
        }

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------

    def run(self, question: str, retrieved_docs: List) -> Dict[str, str]:
        """Run CoT+Citation for a single question."""
        import asyncio
        context          = self._format_docs_for_citation(retrieved_docs)
        formatted_prompt = self.prompt.format(context=context, question=question)
        response         = asyncio.run(self.llm.ainvoke(formatted_prompt))
        return self._parse_output(response.content)

    # ------------------------------------------------------------------
    # Batch inference (dùng llm.batch() thực sự)
    # ------------------------------------------------------------------

    def batch_run(
        self,
        questions: List[str],
        docs_list: List[List],
    ) -> List[Dict[str, str]]:
        """Batch inference qua async gather."""
        import asyncio

        prompts = [
            self.prompt.format(
                context=self._format_docs_for_citation(docs),
                question=question,
            )
            for question, docs in zip(questions, docs_list)
        ]

        async def _run_batch(batch_prompts):
            messages = [[HumanMessage(content=p)] for p in batch_prompts]
            return await asyncio.gather(*[self.llm.ainvoke(m) for m in messages])

        all_results = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="CoT+Citation generating"):
            batch     = prompts[i : i + self.batch_size]
            responses = asyncio.run(_run_batch(batch))
            for resp in responses:
                all_results.append(self._parse_output(resp.content))

        return all_results

    # ------------------------------------------------------------------
    # Ragas-compatible wrapper
    # ------------------------------------------------------------------

    def answer_with_contexts_batch(
        self,
        questions: List[str],
        retriever,
    ) -> List[Dict[str, str]]:
        """
        Full pipeline: retrieve → CoT+Citation generate.
        Output format compatible with Ragas: answer + contexts.
        """
        from tqdm import tqdm as _tqdm

        docs_list = []
        contexts_list = []
        for q in _tqdm(questions, desc="Retrieving for CoT"):
            docs = retriever.invoke(q)
            docs_list.append(docs)
            contexts_list.append([d.page_content for d in docs])

        results = self.batch_run(questions, docs_list)

        return [
            {
                "answer":   r["final_answer"],
                "contexts": ctx,
                "thought":  r["thought_process"],
            }
            for r, ctx in zip(results, contexts_list)
        ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dotenv import load_dotenv
    from phase2_baseline.models import get_llm
    from langchain_core.documents import Document

    load_dotenv()
    llm    = get_llm()
    engine = CotCitationEngine(llm)

    docs = [
        Document(page_content="Sinh viên khóa 51 cần đăng ký lịch hoạt động Đoàn - Hội theo hướng dẫn của Ban Chấp hành."),
        Document(page_content="Chương trình chào đón tân sinh viên khóa 51 diễn ra vào tháng 9 năm 2025 tại hội trường lớn."),
    ]

    result = engine.run("Sinh viên khóa 51 cần làm gì?", docs)
    print("=== THOUGHT ===")
    print(result["thought_process"])
    print("\n=== ANSWER ===")
    print(result["final_answer"])
