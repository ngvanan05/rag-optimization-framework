"""
Phase 3 - v4 Index + Retrieve (BGE-M3 on GPU)
Step 1/2: Load documents, index into Qdrant, retrieve top-10 docs per question.
Saves results to JSON for runv4_rerank_generate.py to process next.

Run from project root:
    python phase3_upgrades/v4_reranking/runv4_index_retrieve.py
"""
import json, pathlib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

from phase2_baseline.config import (
    SEED, DATA_DIR, TESTSET_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_URL, COLLECTION_NAME, RETRIEVE_K, set_global_seed,
)
from phase2_baseline.models import get_embeddings
from phase2_baseline.stage1_indexing.loader import DocumentLoader
from phase2_baseline.stage1_indexing.vector_db import TextSplitter, VectorDB

RETRIEVED_PATH = ROOT / "outputs" / "phase3_upgrades" / "v4_retrieved.json"

def main():
    set_global_seed(SEED)
    print("=" * 55)
    print("  PHASE 3 - v4a: INDEX + RETRIEVE (BGE-M3 GPU)")
    print("=" * 55)

    print("\n[1/3] Loading documents + Chunking...")
    raw_docs = DocumentLoader().load_dir(DATA_DIR)
    chunks   = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split(raw_docs)
    print(f"-> {len(chunks)} chunks.")

    print("\n[2/3] Initializing Qdrant + Embeddings (GPU)...")
    embeddings = get_embeddings()   # GPU
    retriever  = VectorDB(
        documents=chunks, embedding=embeddings,
        collection_name=COLLECTION_NAME, qdrant_url=QDRANT_URL,
    ).get_retriever(search_kwargs={"k": RETRIEVE_K})

    print("\n[3/3] Retrieving top-10 docs per question...")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_df = pd.DataFrame([{
        "user_input":         d.get("user_input", ""),
        "reference":          d.get("reference", ""),
        "reference_contexts": d.get("reference_contexts", []),
    } for d in data])
    print(f"-> {len(test_df)} questions.")

    from tqdm import tqdm
    retrieved = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Retrieving"):
        docs = retriever.invoke(row["user_input"])
        retrieved.append({
            "user_input":         row["user_input"],
            "reference":          row["reference"],
            "reference_contexts": row["reference_contexts"],
            "retrieved_docs":     [d.page_content for d in docs],
        })

    RETRIEVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVED_PATH, "w", encoding="utf-8") as f:
        json.dump(retrieved, f, ensure_ascii=False, indent=2)

    print(f"\n-> Saved retrieved contexts: {RETRIEVED_PATH}")
    print("\nNext step — shut down terminal, then run:")
    print("  python phase3_upgrades/v4_reranking/runv4_rerank_generate.py")

if __name__ == "__main__":
    main()
