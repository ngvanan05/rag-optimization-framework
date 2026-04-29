import torch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import LLM_MODEL, EMBEDDING_MODEL_NAME, SEED


def get_llm(
    model_name: str = LLM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    seed: int = SEED,
    api_key: str = None
):
    """
    Initialize GPT-4o-mini.
    temperature=0.0 to maximize determinism of outputs.
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        api_key=api_key
    )
    return llm


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME, force_cpu: bool = False):
    """
    Initialize BGE-M3. Use GPU for indexing, CPU for eval to avoid OOM.
    """
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )
    return embeddings


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    print("Initializing LLM (GPT-4o-mini)...")
    llm = get_llm()

    print("Initializing Embeddings (BGE-M3 on GPU)...")
    embeddings = get_embeddings()

    print("--- Both models initialized successfully ---")
