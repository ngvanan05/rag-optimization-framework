"""
Run Pipeline
Initialize the full RAG system and run Batch Inference on the testset.

Run:
    cd phase2_baseline
    python run_pipeline.py                          # default
    python run_pipeline.py --output my_results.csv  # custom output
"""
import os
import time
import json
import argparse
import pandas as pd
from dotenv import load_dotenv

from phase2_baseline.config import (
    SEED, LLM_MODEL, BATCH_SIZE, DATA_DIR, TESTSET_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_URL, COLLECTION_NAME,
    TOP_K, set_global_seed
)
from phase2_baseline.models import get_llm, get_embeddings
from phase2_baseline.stage1_indexing.loader import DocumentLoader
from phase2_baseline.stage1_indexing.vector_db import TextSplitter, VectorDB
from phase2_baseline.stage2_retrieval.search_engine import SearchEngine
from phase2_baseline.stage3_generation.rag_chain import RAGChain


# ---------------------------------------------------------------------------
# Khởi tạo hệ thống
# ---------------------------------------------------------------------------

def build_rag_system():
    """Initialize the full RAG pipeline from end to end."""
    print("=" * 50)
    print("  INITIALIZING BASELINE RAG SYSTEM")
    print("=" * 50)

    set_global_seed(SEED)

    # [1/4] Models
    print("\n[1/4] Loading models...")
    llm        = get_llm(model_name=LLM_MODEL, seed=SEED)
    embeddings = get_embeddings()

    # [2/4] Load & Chunk documents
    print("\n[2/4] Loading and processing documents...")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    loader   = DocumentLoader()
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    raw_docs = loader.load_dir(DATA_DIR)
    chunks   = splitter.split(raw_docs)
    print(f"-> Done: {len(chunks)} chunks.")

    # [3/4] Vector DB
    print("\n[3/4] Initializing Vector Database (Qdrant)...")
    vdb = VectorDB(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
    )
    retriever = vdb.get_retriever(search_kwargs={"k": TOP_K})

    # [4/4] RAG Chain
    print("\n[4/4] Connecting RAG Pipeline...")
    search_engine = SearchEngine(retriever)
    rag_chain     = RAGChain(llm, batch_size=BATCH_SIZE)

    print("\n--- BASELINE RAG SYSTEM READY ---\n")
    return search_engine, rag_chain, retriever


# ---------------------------------------------------------------------------
# Load testset
# ---------------------------------------------------------------------------

def load_testset(testset_path: str = TESTSET_PATH) -> pd.DataFrame:
    """Read testset from JSON and return a DataFrame."""
    print(f"Loading testset: {testset_path}")
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {
            "user_input":         item.get("user_input", ""),
            "reference":          item.get("reference", ""),
            "reference_contexts": item.get("reference_contexts", []),
        }
        for item in data
    ])
    print(f"-> Loaded {len(df)} questions.")
    return df


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(search_engine: SearchEngine, rag_chain: RAGChain, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run Batch Inference on the full testset."""
    set_global_seed(SEED)
    questions = test_df["user_input"].tolist()

    print(f"\nStarting inference on {len(questions)} questions...")
    start = time.time()

    retrieved_data = search_engine.retrieve_batch(questions)
    results        = rag_chain.batch_answer(retrieved_data)

    elapsed = time.time() - start
    print(f"Done! Time: {elapsed:.2f}s")

    eval_df = pd.DataFrame([
        {
            "user_input":         row["user_input"],
            "answer":             result["answer"],        # Ragas cần cột "answer"
            "retrieved_contexts": result["contexts"],
            "reference":          row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts", []),
        }
        for result, (_, row) in zip(results, test_df.iterrows())
    ])

    print("\nSample results (first 5 rows):")
    print(eval_df[["user_input", "answer"]].head())
    return eval_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()

    import pathlib
    DEFAULT_OUTPUT = str(pathlib.Path(__file__).resolve().parent.parent / "outputs" / "phase2_baseline" / "rag_inference_results.csv")

    parser = argparse.ArgumentParser(description="Baseline RAG Pipeline")
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV filename (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    search_engine, rag_chain, retriever = build_rag_system()

    if os.path.exists(TESTSET_PATH):
        test_df = load_testset(TESTSET_PATH)
        eval_df = run_inference(search_engine, rag_chain, test_df)
        eval_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"-> Saved results: {args.output}")
    else:
        print(f"Testset not found at: {TESTSET_PATH}")
        print("Please run phase1 to generate the testset first.")
