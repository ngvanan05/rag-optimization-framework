"""
Phase 3 - v4 Rerank + Generate (Qwen3 on GPU)
Step 2/2: Reads retrieved contexts from runv4_index_retrieve.py, reranks with Qwen3 GPU, generates answers.

Run from project root (after runv4_index_retrieve.py completes):
    python phase3_upgrades/v4_reranking/runv4_rerank_generate.py
"""
import json, time, pathlib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from phase2_baseline.config import SEED, LLM_MODEL, BATCH_SIZE, TOP_K, set_global_seed
from phase2_baseline.models import get_llm
from phase3_upgrades.v4_reranking.cross_encoder import CrossEncoderReranker, BatchRAG

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RETRIEVED_PATH = ROOT / "outputs" / "phase3_upgrades" / "v4_retrieved.json"
OUTPUT_PATH    = ROOT / "outputs" / "phase3_upgrades" / "v4_reranking.csv"


def main():
    set_global_seed(SEED)
    print("=" * 55)
    print("  PHASE 3 - v4b: RERANK + GENERATE (Qwen3 GPU)")
    print("=" * 55)

    if not RETRIEVED_PATH.exists():
        print(f"File not found: {RETRIEVED_PATH}")
        print("Please run runv4_index_retrieve.py first.")
        return

    print("\n[1/3] Reading retrieved contexts...")
    with open(RETRIEVED_PATH, "r", encoding="utf-8") as f:
        retrieved = json.load(f)
    print(f"-> {len(retrieved)} questions.")

    print("\n[2/3] Initializing LLM + Reranker (Qwen3 GPU)...")
    llm      = get_llm(model_name=LLM_MODEL, seed=SEED)
    reranker = CrossEncoderReranker(top_k=TOP_K, use_gpu=True)
    generator = BatchRAG(llm, reranker=reranker, batch_size=BATCH_SIZE)

    print("\n[3/3] Rerank + Generate...")
    from tqdm import tqdm
    from langchain_core.documents import Document

    start   = time.time()
    results = []
    for item in tqdm(retrieved, desc="Reranking & Generating"):
        docs = [Document(page_content=c) for c in item["retrieved_docs"]]
        results.append({
            "question":      item["user_input"],
            "retrieved_docs": docs,
            "reference":     item["reference"],
            "reference_contexts": item["reference_contexts"],
        })

    # Batch rerank + generate
    final = generator.answer_with_contexts_batch(
        [r["question"] for r in results],
        # Dùng pre-retrieved docs thay vì retriever
        _docs_list=[r["retrieved_docs"] for r in results],
    )
    print(f"Done! {time.time()-start:.2f}s")

    eval_df = pd.DataFrame([{
        "user_input":         r["question"],
        "answer":             f["answer"],
        "retrieved_contexts": f["contexts"],
        "reference":          r["reference"],
        "reference_contexts": r["reference_contexts"],
    } for r, f in zip(results, final)])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"-> Saved: {OUTPUT_PATH}")
    print(f"\nRun eval:\n  python phase2_baseline/run_eval.py --input \"{OUTPUT_PATH}\"")

if __name__ == "__main__":
    main()
