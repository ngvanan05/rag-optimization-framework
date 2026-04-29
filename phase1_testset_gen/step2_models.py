"""
Phase 1 — Step 2: Khởi tạo LLM + Embedding cho RAGAS
Chạy độc lập để test:
    python step2_models.py

Patch LangchainLLMWrapper để tương thích với langchain-openai >= 0.2
(method agenerate_prompt đã bị xóa, thay bằng agenerate_prompt wrapper).
"""
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import torch
from dotenv import load_dotenv
load_dotenv()

from config import LLM_MODEL, EMBEDDING_MODEL


def _patch_langchain_llm_wrapper():
    """
    Patch LangchainLLMWrapper để thêm agenerate_prompt tương thích
    với langchain-openai >= 0.2 (đã xóa method này).
    """
    from ragas.llms.base import LangchainLLMWrapper

    if hasattr(LangchainLLMWrapper, "_patched"):
        return  # đã patch rồi

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Compatibility shim: route agenerate_prompt → agenerate_prompt via agenerate_text."""
        from langchain_core.outputs import LLMResult, Generation, ChatGeneration
        results = []
        for prompt in prompts:
            result = await self.agenerate_text(
                prompt=prompt,
                n=1,
                stop=stop,
                callbacks=callbacks,
            )
            results.append(result.generations[0])
        # Merge thành 1 LLMResult
        return LLMResult(generations=results)

    LangchainLLMWrapper.agenerate_prompt = agenerate_prompt
    LangchainLLMWrapper._patched = True


def get_ragas_wrappers():
    """
    Trả về (llm_wrapper, embedding_wrapper) tương thích với RAGAS 0.2.x.
    """
    # Patch trước khi dùng
    _patch_langchain_llm_wrapper()

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    # LLM
    print(f"Khởi tạo LLM: {LLM_MODEL}...")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=8192)
    llm_wrapper = LangchainLLMWrapper(llm)
    print("LLM OK\n")

    # Embedding — BGE-M3 trên GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Khởi tạo Embedding: {EMBEDDING_MODEL} trên {device.upper()}...")
    bge = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    emb_wrapper = LangchainEmbeddingsWrapper(bge)
    print("Embedding OK\n")

    return llm_wrapper, emb_wrapper


if __name__ == "__main__":
    llm_w, emb_w = get_ragas_wrappers()
    print("LLM wrapper:", llm_w)
    print("Embedding wrapper:", emb_w)
    print("Models sẵn sàng!")
