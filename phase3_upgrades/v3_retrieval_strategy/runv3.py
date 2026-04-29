"""
Phase 3 - v3 Hybrid Search (BM25 + Qdrant → RRF)
Replaces dense VectorDB with HybridVectorDB.
The rest of the pipeline remains identical to baseline.

Run from project root:
    python phase3_upgrades/v3_retrieval_strategy/runv3.py
"""
import os, sys, time, json, pathlib
import pandas as pd
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase2_baseline"))
load_dotenv()

from phase2_baseline.config import (
    SEED, LLM_MODEL, BATCH_SIZE, DATA_DIR, TESTSET_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_URL, COLLECTION_NAME, TOP_K, set_global_seed,
)
from phase2_baseline.models import get_llm, get_embeddings
from phase2_baseline.stage1_indexing.loader import DocumentLoader
from phase2_baseline.stage1_indexing.vector_db import TextSplitter
from phase2_baseline.stage2_retrieval.search_engine import SearchEngine
from phase2_baseline.stage3_generation.rag_chain import RAGChain
from phase3_upgrades.v3_retrieval_strategy.hybrid_search import HybridVectorDB

OUTPUT_PATH = ROOT / "outputs" / "phase3_upgrades" / "v3_hybrid_search.csv"

def main():
    set_global_seed(SEED)
    print("=" * 55)
    print("  PHASE 3 - v3: HYBRID SEARCH (BM25 + Qdrant + RRF)")
    print("=" * 55)

    print("\n[1/4] Loading models...")
    llm        = get_llm(model_name=LLM_MODEL, seed=SEED)
    embeddings = get_embeddings()

    print("\n[2/4] Loading documents + Chunking (baseline)...")
    raw_docs = DocumentLoader().load_dir(DATA_DIR)
    chunks   = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split(raw_docs)
    print(f"-> {len(chunks)} chunks.")

    print("\n[3/4] Initializing HybridVectorDB (Qdrant + BM25)...")
    retriever = HybridVectorDB(
        documents=chunks, embedding=embeddings,
        collection_name=COLLECTION_NAME, qdrant_url=QDRANT_URL,
    ).get_retriever(search_kwargs={"k": TOP_K})

    print("\n[4/4] Inference...")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_df = pd.DataFrame([{
        "user_input": d.get("user_input", ""),
        "reference":  d.get("reference", ""),
        "reference_contexts": d.get("reference_contexts", []),
    } for d in data])
    print(f"-> {len(test_df)} questions.")

    search_engine = SearchEngine(retriever)
    rag_chain     = RAGChain(llm, batch_size=BATCH_SIZE)

    start     = time.time()
    retrieved = search_engine.retrieve_batch(test_df["user_input"].tolist())
    results   = rag_chain.batch_answer(retrieved)
    print(f"Done! {time.time()-start:.2f}s")

    eval_df = pd.DataFrame([{
        "user_input":         row["user_input"],
        "answer":             r["answer"],
        "retrieved_contexts": r["contexts"],
        "reference":          row.get("reference", ""),
        "reference_contexts": row.get("reference_contexts", []),
    } for r, (_, row) in zip(results, test_df.iterrows())])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"-> Saved: {OUTPUT_PATH}")
    print(f"\nRun eval:\n  python phase2_baseline/run_eval.py --input \"{OUTPUT_PATH}\"")

if __name__ == "__main__":
    main()
