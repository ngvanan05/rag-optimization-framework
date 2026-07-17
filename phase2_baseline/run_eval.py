"""
Run Evaluation
Score with Ragas and export PDF report with charts.
Supports passing a custom inference CSV filename to compare variants.

Run:
    cd phase2_baseline
    python run_eval.py                                      # default
    python run_eval.py --input my_results.csv               # custom input
    python run_eval.py --input my_results.csv --output my_report.pdf
"""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# Monkey-patch broken decorator trước khi import ragas.evaluation
import ragas._analytics as _ra
_ra.track_was_completed = lambda fn: fn

# Patch _analytics_batcher và EvaluationEvent nếu có
try:
    if hasattr(_ra, '_analytics_batcher'):
        _ra._analytics_batcher.flush = lambda self=None: None
        _ra._analytics_batcher.add_evaluation = lambda self=None, *a, **kw: None
except Exception:
    pass

try:
    from ragas._analytics import EvaluationEvent
    # Patch để chấp nhận bất kỳ kwargs nào
    original_init = EvaluationEvent.__init__
    def _patched_init(self, **kwargs):
        # Lọc chỉ giữ các field hợp lệ
        valid = {k: v for k, v in kwargs.items() if k in ('evaluation_id', 'metrics', 'num_samples')}
        try:
            original_init(self, **valid)
        except Exception:
            pass
    EvaluationEvent.__init__ = _patched_init
except Exception:
    pass

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datasets import Dataset

from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from phase2_baseline.config import SEED, EVAL_SAMPLE_SIZE, EVAL_MAX_WORKERS, EVAL_TIMEOUT, LLM_MODEL
from phase2_baseline.models import get_embeddings, get_llm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rag(eval_df: pd.DataFrame):
    """Score the RAG system using Ragas (4 metrics)."""
    print("\n--- Starting Ragas evaluation ---")

    # Chuẩn hóa tên cột về đúng schema Ragas
    col_map = {}
    if "response" in eval_df.columns and "answer" not in eval_df.columns:
        col_map["response"] = "answer"
    if "retrieved_contexts" in eval_df.columns and "contexts" not in eval_df.columns:
        col_map["retrieved_contexts"] = "contexts"
    if col_map:
        eval_df = eval_df.rename(columns=col_map)
        print(f"-> Renamed columns: {col_map}")

    # Lấy mẫu nếu cấu hình
    if EVAL_SAMPLE_SIZE is not None:
        eval_df = eval_df.sample(n=min(EVAL_SAMPLE_SIZE, len(eval_df)), random_state=SEED)
        print(f"-> Evaluating on {len(eval_df)} samples.")
    else:
        print(f"-> Evaluating on all {len(eval_df)} samples.")

    # Parse contexts từ string nếu cần
    import ast
    for col in ["contexts", "reference_contexts"]:
        if col in eval_df.columns and isinstance(eval_df[col].iloc[0], str):
            eval_df[col] = eval_df[col].apply(lambda x: ast.literal_eval(x) if x else [])

    dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    # Judge models — dùng CPU cho embeddings để tránh OOM khi eval
    print("-> Loading Judge Models...")
    wrapped_embeddings = LangchainEmbeddingsWrapper(get_embeddings(force_cpu=True))
    wrapped_llm        = LangchainLLMWrapper(get_llm(model_name=LLM_MODEL))

    # Gán LLM/Embeddings cho từng metric
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = wrapped_llm
        if hasattr(m, "embeddings"):
            m.embeddings = wrapped_embeddings

    run_config = RunConfig(
        max_workers=4,
        timeout=EVAL_TIMEOUT,
        max_retries=3,
        max_wait=60,
    )

    print("-> Scoring... (this may take a few minutes)")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    print("\n=== RAGAS RESULTS ===")
    print(results)
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_report(results, pdf_filename: str = "RAGAS_Evaluation_Report.pdf"):
    """Export evaluation results to PDF and detailed CSV."""
    results_df  = results.to_pandas()
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                   if c in results_df.columns]

    if not metric_cols:
        print("Error: No metric columns found.")
        return

    scores = {col: results_df[col].mean() for col in metric_cols}

    print(f"\nExporting report: {pdf_filename}")
    with PdfPages(pdf_filename) as pdf:
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(10, 6))

        bars = plt.bar(scores.keys(), scores.values(), color="skyblue", edgecolor="navy", alpha=0.7)
        plt.title(f"RAGAS Baseline (LLM: {LLM_MODEL} | Embed: BGE-M3)", fontsize=14, fontweight="bold")
        plt.ylabel("Score (0-1)", fontsize=12)
        plt.ylim(0, 1.1)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02,
                     round(yval, 4), ha="center", va="bottom", fontweight="bold")

        plt.xticks(rotation=15)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"-> Saved: {pdf_filename}")

    csv_filename = pdf_filename.replace(".pdf", "_detailed.csv")
    results_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"-> Saved detailed: {csv_filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _OUTPUT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "outputs" / "phase2_baseline")
    DEFAULT_INPUT  = os.path.join(_OUTPUT_DIR, "rag_inference_results.csv")

    parser = argparse.ArgumentParser(description="RAGAS Evaluation")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to inference CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output PDF report (default: derived from input filename)",
    )
    args = parser.parse_args()

    inference_csv = args.input

    # Auto-derive output PDF cạnh file input
    if args.output is None:
        pdf_out = inference_csv.replace(".csv", "_RAGAS_Report.pdf")
    else:
        pdf_out = args.output

    os.makedirs(os.path.dirname(os.path.abspath(pdf_out)), exist_ok=True)

    if not os.path.exists(inference_csv):
        print(f"{inference_csv} not found. Please run run_pipeline.py first.")
    else:
        eval_df = pd.read_csv(inference_csv)
        results = evaluate_rag(eval_df)
        save_report(results, pdf_filename=pdf_out)
