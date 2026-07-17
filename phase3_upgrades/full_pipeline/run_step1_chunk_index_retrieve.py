"""
Full Pipeline - Step 1: Chunk + Index + Retrieve
Combines v1 + v3 + v2:
  - v1: SemanticChunker (BGE-M3 GPU) replacing RecursiveCharacterTextSplitter
  - v3: HybridVectorDB (Qdrant dense + BM25 sparse → RRF)
  - v2: QueryDecomposition (splits question → retrieves for sub-questions → dedup)

Output: JSON containing top-10 retrieved docs per question
→ Run run_step2_rerank_generate.py next

Run from project root:
    python phase3_upgrades/full_pipeline/run_step1_chunk_index_retrieve.py
"""
import json, pathlib
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

from phase2_baseline.config import (
    SEED, DATA_DIR, TESTSET_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_URL, COLLECTION_NAME, set_global_seed,
)
from phase2_baseline.models import get_embeddings, get_llm
from phase2_baseline.stage1_indexing.loader import DocumentLoader
from phase3_upgrades.v1_indexing_chunking.semantic_chunker import SemanticChunker
from phase3_upgrades.v3_retrieval_strategy.hybrid_search import HybridVectorDB
from phase3_upgrades.v2_query_transformation.query_decomposer import DecompositionQueryTransformer
from phase2_baseline.config import LLM_MODEL, TOP_K
from phase3_upgrades.v4_reranking.cross_encoder import RETRIEVE_K

RETRIEVED_PATH = ROOT / "outputs" / "full_pipeline" / "step1_retrieved.json"
BM25_PATH      = ROOT / "database" / "bm25_index.pkl"


def main():
    set_global_seed(SEED)
    print("=" * 60)
    print("  FULL PIPELINE - STEP 1: CHUNK + INDEX + RETRIEVE")
    print("  v1 SemanticChunk + v3 HybridSearch + v2 QueryDecompose")
    print("=" * 60)

    # [1/4] Load documents
    print("\n[1/4] Loading documents...")
    raw_docs = DocumentLoader().load_dir(DATA_DIR)

    # [2/4] v1: Semantic Chunking (BGE-M3 GPU) → free GPU after done
    print("\n[2/4] v1: Semantic Chunking (BGE-M3 GPU)...")
    chunker = SemanticChunker()
    chunks  = chunker.split(raw_docs)
    chunker.free_gpu()
    print(f"-> {len(chunks)} chunks.")

    # [3/4] v3: HybridVectorDB (BGE-M3 GPU + BM25 CPU)
    print("\n[3/4] v3: Initializing HybridVectorDB (Qdrant + BM25)...")
    embeddings = get_embeddings()  # GPU
    vdb = HybridVectorDB(
        documents=chunks, embedding=embeddings,
        collection_name=COLLECTION_NAME, qdrant_url=QDRANT_URL,
    )
    retriever = vdb.get_retriever(search_kwargs={"k": RETRIEVE_K})

    # [4/4] v2: QueryDecomposition + Retrieve
    print("\n[4/4] v2: Query Decomposition + Retrieve...")
    llm         = get_llm(model_name=LLM_MODEL, seed=SEED)
    decomposer  = DecompositionQueryTransformer(llm)

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_df = pd.DataFrame([{
        "user_input":         d.get("user_input", ""),
        "reference":          d.get("reference", ""),
        "reference_contexts": d.get("reference_contexts", []),
    } for d in data])
    print(f"-> {len(test_df)} questions.")

    retrieved = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Decompose & Retrieve"):
        question = row["user_input"]

        # Decompose question into sub-questions
        sub_questions = decomposer.decompose(question)

        # Retrieve for each sub-question + original, dedup
        all_docs  = []
        seen      = set()
        for q in sub_questions + [question]:
            for doc in retriever.invoke(q):
                if doc.page_content not in seen:
                    all_docs.append(doc.page_content)
                    seen.add(doc.page_content)

        retrieved.append({
            "user_input":         question,
            "sub_questions":      sub_questions,
            "reference":          row["reference"],
            "reference_contexts": row["reference_contexts"],
            "retrieved_docs":     all_docs[:RETRIEVE_K],  # limit to top-10
        })

    RETRIEVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVED_PATH, "w", encoding="utf-8") as f:
        json.dump(retrieved, f, ensure_ascii=False, indent=2)

    print(f"\n-> Saved: {RETRIEVED_PATH}")
    print("\nNext step — shut down terminal, then run:")
    print("  python phase3_upgrades/full_pipeline/run_step2_rerank_generate.py")


if __name__ == "__main__":
    main()
