"""
Phase 3 - v4 Reranking
Cross-encoder reranking using Qwen3-Reranker-0.6B (Generative Reranker).

Strategy: Retrieve many (RETRIEVE_K=10) → Rerank → Keep top RERANK_TOP_K=3
"""
import torch
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Import các thành phần từ Baseline (Phase 2)
from phase2_baseline.config import BATCH_SIZE, RETRIEVE_K, TOP_K as RERANK_TOP_K
from phase2_baseline.stage3_generation.rag_chain import FocusedAnswerParser

# --- CẤU HÌNH ---
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


class CrossEncoderReranker:
    """
    Reranker using Qwen3-Reranker-0.6B — generative reranker (LLM-based).
    Takes a query + document and predicts yes/no tokens to score relevance.
    """
    DEFAULT_INSTRUCTION = (
        "Given a Vietnamese document query, retrieve relevant passages "
        "that answer the query"
    )

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        top_k: int = RERANK_TOP_K,
        instruction: str = None,
        use_gpu: bool = False,
    ):
        self.top_k = top_k
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

        print(f"-> Loading Reranker: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        if use_gpu and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16,
            ).eval().cuda()
            print("-> Reranker loaded on GPU (FP16)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32,
            ).eval()
            print("-> Reranker loaded on CPU (FP32)")

        # Token IDs cho yes/no
        self.token_true_id  = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.max_length = 8192

        # Prefix/suffix theo chat template của Qwen3
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens  = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _process_inputs(self, pairs: List[str]):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        # Giữ trên cùng device với model
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_scores(self, inputs) -> List[float]:
        logits = self.model(**inputs).logits[:, -1, :]
        true_vec  = logits[:, self.token_true_id]
        false_vec = logits[:, self.token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()

    def rerank(self, query: str, documents: List) -> List:
        """
        Rerank a list of LangChain Documents.
        Processes one doc at a time to avoid OOM on 6GB GPU.
        """
        if not documents:
            return documents

        scores = []
        for doc in documents:
            pair   = self._format_pair(query, doc.page_content)
            inputs = self._process_inputs([pair])
            score  = self._compute_scores(inputs)[0]
            scores.append(score)
            # Giải phóng cache sau mỗi doc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


class BatchRAG:
    """
    Enhanced BatchRAG with integrated CrossEncoderReranker.
    Interface matches the baseline for easy swapping into the pipeline.
    """

    def __init__(self, llm, reranker: CrossEncoderReranker = None, batch_size: int = BATCH_SIZE):
        self.llm        = llm
        self.reranker   = reranker
        self.batch_size = batch_size

        self.prompt = PromptTemplate.from_template(
            "Bạn là trợ lý AI phân tích tài liệu tiếng Việt.\n\n"
            "[TÀI LIỆU]:\n{context}\n\n"
            "[CÂU HỎI]:\n{question}\n\n"
            "Hãy trả lời dựa trên tài liệu. "
            "Nếu tài liệu không có thông tin, nói rõ \"Không có thông tin\".\n"
            "[TRẢ LỜI]:"
        )
        self.answer_parser = FocusedAnswerParser()

    def _format_docs(self, docs) -> str:
        return "\n\n".join(
            (doc.page_content or "").strip()
            for doc in docs
            if (doc.page_content or "").strip()
        )

    # ------------------------------------------------------------------
    # Retrieval + Reranking
    # ------------------------------------------------------------------

    def batch_retrieve(self, questions: List[str], retriever) -> List[dict]:
        """
        For each question:
          1. Fetch RETRIEVE_K docs from the vector DB
          2. Rerank → keep RERANK_TOP_K docs (if reranker is set)
        """
        all_contexts = []
        for question in tqdm(questions, desc="Retrieving & Reranking"):
            docs = retriever.invoke(question)

            if self.reranker is not None:
                docs = self.reranker.rerank(question, docs)

            all_contexts.append({
                "question":          question,
                "contexts":          [(doc.page_content or "") for doc in docs],
                "formatted_context": self._format_docs(docs),
            })
        return all_contexts

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
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating answers"):
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

    def answer_with_contexts_batch(
        self,
        questions: List[str],
        retriever=None,
        _docs_list: List[List] = None,
    ) -> List[dict]:
        """
        Full pipeline: retrieve → rerank → generate.
        If _docs_list is provided, skips the retriever and uses pre-fetched docs.
        Returns a list of dicts with answer + contexts (for Ragas eval).
        """
        if _docs_list is not None:
            # Dùng pre-retrieved docs (từ runv4_index_retrieve.py)
            retrieved_data = []
            for question, docs in tqdm(
                zip(questions, _docs_list), total=len(questions), desc="Reranking"
            ):
                if self.reranker is not None:
                    docs = self.reranker.rerank(question, docs)
                retrieved_data.append({
                    "question":          question,
                    "contexts":          [(doc.page_content or "") for doc in docs],
                    "formatted_context": self._format_docs(docs),
                })
        else:
            retrieved_data = self.batch_retrieve(questions, retriever)

        prompts = [
            self.prompt.format(
                context=data["formatted_context"],
                question=data["question"],
            )
            for data in retrieved_data
        ]

        answers = self.batch_generate(prompts)

        return [
            {"answer": answer, "contexts": data["contexts"]}
            for data, answer in zip(retrieved_data, answers)
        ]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    reranker = CrossEncoderReranker(top_k=3)

    from langchain_core.documents import Document
    query = "Sinh viên khóa 51 cần đăng ký những hoạt động gì?"
    docs = [
        Document(page_content="Sinh viên khóa 51 cần đăng ký lịch hoạt động Đoàn - Hội theo hướng dẫn."),
        Document(page_content="Chương trình chào đón tân sinh viên khóa 51 diễn ra vào tháng 9 năm 2025."),
        Document(page_content="Thời tiết hôm nay rất đẹp và nắng ấm."),
    ]

    ranked = reranker.rerank(query, docs)
    for i, doc in enumerate(ranked):
        print(f"[{i+1}] {doc.page_content[:100]}")
