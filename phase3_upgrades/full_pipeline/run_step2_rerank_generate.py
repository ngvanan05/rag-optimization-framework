"""
Full Pipeline - Step 2: Rerank + Generate
Combines v4 + baseline generation:
  - v4: Qwen3-Reranker-0.6B (GPU) reranks top-10 → top-3
  - Generation: baseline RAGChain (GPT-4o-mini)

Input:  JSON from run_step1_chunk_index_retrieve.py
Output: CSV ready for RAGAS eval

Run from project root (after step1 completes):
    python phase3_upgrades/full_pipeline/run_step2_rerank_generate.py
"""
import sys, json, time, pathlib, asyncio
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase2_baseline"))
load_dotenv()

from phase2_baseline.config import SEED, LLM_MODEL, BATCH_SIZE, TOP_K, set_global_seed
from phase2_baseline.models import get_llm
from phase2_baseline.stage3_generation.rag_chain import RAGChain
from phase2_baseline.stage2_retrieval.search_engine import SearchEngine
from phase3_upgrades.v4_reranking.cross_encoder import CrossEncoderReranker

RETRIEVED_PATH = ROOT / "outputs" / "full_pipeline" / "step1_retrieved.json"
OUTPUT_PATH    = ROOT / "outputs" / "full_pipeline" / "full_pipeline_results.csv"


def main():
    set_global_seed(SEED)
    print("=" * 60)
    print("  FULL PIPELINE - STEP 2: RERANK + GENERATE")
    print("  v4 Qwen3 Reranker (GPU) + Baseline RAGChain")
    print("=" * 60)

    if not RETRIEVED_PATH.exists():
        print(f"File not found: {RETRIEVED_PATH}")
        print("Please run run_step1_chunk_index_retrieve.py first.")
        return

    # [1/3] Load retrieved docs
    print("\n[1/3] Reading retrieved contexts...")
    with open(RETRIEVED_PATH, "r", encoding="utf-8") as f:
        retrieved = json.load(f)
    print(f"-> {len(retrieved)} questions.")

    # [2/3] v4: Rerank (Qwen3 GPU)
    print("\n[2/3] v4: Initializing Qwen3 Reranker (GPU)...")
    reranker = CrossEncoderReranker(top_k=TOP_K, use_gpu=True)

    print("-> Reranking...")
    reranked = []
    for item in tqdm(retrieved, desc="Reranking"):
        docs     = [Document(page_content=c) for c in item["retrieved_docs"]]
        top_docs = reranker.rerank(item["user_input"], docs)
        reranked.append({**item, "reranked_docs": top_docs})

    # [3/3] Baseline Generate (GPT-4o-mini)
    print("\n[3/3] Baseline RAGChain Generate (GPT-4o-mini)...")
    llm       = get_llm(model_name=LLM_MODEL, seed=SEED)
    rag_chain = RAGChain(llm, batch_size=BATCH_SIZE)

    # Format retrieved_data theo chuẩn RAGChain.batch_answer
    retrieved_data = [
        {
            "question":          item["user_input"],
            "contexts":          [d.page_content for d in item["reranked_docs"]],
            "formatted_context": "\n\n".join(d.page_content for d in item["reranked_docs"]),
        }
        for item in reranked
    ]

    start   = time.time()
    results = rag_chain.batch_answer(retrieved_data)
    print(f"Done! {time.time()-start:.2f}s")

    eval_df = pd.DataFrame([{
        "user_input":         item["user_input"],
        "answer":             r["answer"],
        "retrieved_contexts": r["contexts"],
        "reference":          item["reference"],
        "reference_contexts": item["reference_contexts"],
    } for item, r in zip(reranked, results)])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n-> Saved: {OUTPUT_PATH}")
    print(f"\nRun eval:\n  python phase2_baseline/run_eval.py --input \"{OUTPUT_PATH}\"")


if __name__ == "__main__":
    main()
