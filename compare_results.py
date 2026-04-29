"""
Compare RAGAS results across all variants (Baseline + Phase 3 v1-v5).
Exports a summary table + 4 horizontal bar charts (matching the reference figure).

Run from project root:
    python outputs/compare_results.py
"""
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Cấu hình các variant
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent

VARIANTS = [
    {
        "name": "Baseline",
        "csv":  ROOT / "outputs" / "phase2_baseline" / "rag_inference_results_RAGAS_Report_detailed.csv",
    },
    {
        "name": "v1 Semantic Chunking",
        "csv":  ROOT / "outputs" / "phase3_upgrades" / "v1_semantic_chunker_RAGAS_Report_detailed.csv",
    },
    {
        "name": "v2 Query Decomposition",
        "csv":  ROOT / "outputs" / "phase3_upgrades" / "v2_query_decomposition_RAGAS_Report_detailed.csv",
    },
    {
        "name": "v3 Hybrid Search (RRF)",
        "csv":  ROOT / "outputs" / "phase3_upgrades" / "v3_hybrid_search_RAGAS_Report_detailed.csv",
    },
    {
        "name": "v4 Cross-Encoder Reranking",
        "csv":  ROOT / "outputs" / "phase3_upgrades" / "v4_reranking_RAGAS_Report_detailed.csv",
    },
    {
        "name": "v5 CoT + Citation",
        "csv":  ROOT / "outputs" / "phase3_upgrades" / "v5_cot_citation_RAGAS_Report_detailed.csv",
    },
    {
        "name": "Full Pipeline (v1+v2+v3+v4)",
        "csv":  ROOT / "outputs" / "full_pipeline" / "full_pipeline_results_RAGAS_Report_detailed.csv",
    },
    {
        "name": "Combo v1+v2+v4 (Semantic+Decompose+Rerank)",
        "csv":  ROOT / "outputs" / "semantic_decompose_rerank" / "semantic_decompose_rerank_results_RAGAS_Report_detailed.csv",
    },
]

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}

OUTPUT_PDF = ROOT / "outputs" / "comparison_report.pdf"
OUTPUT_CSV = ROOT / "outputs" / "comparison_summary.csv"

# ---------------------------------------------------------------------------
# Load & tính điểm trung bình
# ---------------------------------------------------------------------------

def load_scores() -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        if not v["csv"].exists():
            print(f"[SKIP] Not found: {v['csv']}")
            continue
        df = pd.read_csv(v["csv"])
        row = {"variant": v["name"]}
        for m in METRICS:
            if m in df.columns:
                row[m] = round(df[m].mean(), 4)
            else:
                row[m] = None
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table with delta
# ---------------------------------------------------------------------------

def build_summary_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    baseline = scores_df[scores_df["variant"].str.contains("Baseline")].iloc[0]
    rows = []
    for _, row in scores_df.iterrows():
        r = {"Experiment": row["variant"]}
        for m in METRICS:
            label = METRIC_LABELS[m]
            score = row[m]
            delta = round(score - baseline[m], 4) if score is not None else None
            r[f"{label} Score"] = score
            r[f"{label} Δ"]     = delta
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Horizontal bar chart (matching reference figure)
# ---------------------------------------------------------------------------

def plot_metric_chart(ax, scores_df: pd.DataFrame, metric: str, baseline_score: float):
    """Draw horizontal bar chart for 1 metric, sorted by score descending."""
    df = scores_df[["variant", metric]].dropna().copy()
    df["delta"] = df[metric] - baseline_score
    df = df.sort_values(metric, ascending=True)  # ascending so bars go bottom-up

    colors = ["#2196F3" if d >= 0 else "#F44336" for d in df["delta"]]
    bars = ax.barh(df["variant"], df[metric], color=colors, edgecolor="white", height=0.6)

    # Red baseline line
    ax.axvline(x=baseline_score, color="red", linestyle="--", linewidth=1.5, label=f"Baseline = {baseline_score:.2f}")

    # Delta labels
    for bar, delta in zip(bars, df["delta"]):
        x = bar.get_width()
        sign = "+" if delta >= 0 else ""
        ax.text(x + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.2f}", va="center", ha="left", fontsize=8, fontweight="bold")

    ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
    ax.set_xlabel("Score")
    ax.tick_params(axis="y", labelsize=8)

    # Baseline legend
    ax.legend(loc="lower right", fontsize=8,
              handles=[mpatches.Patch(color="red", label=f"Baseline = {baseline_score:.2f}")])

    # Clamp x-axis
    min_score = max(0, df[metric].min() - 0.05)
    max_score = min(1.0, df[metric].max() + 0.08)
    ax.set_xlim(min_score, max_score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  RAGAS RESULTS COMPARISON — ALL VARIANTS")
    print("=" * 55)

    scores_df = load_scores()
    if scores_df.empty:
        print("\nNo data found. Please run eval for variants first.")
        return

    print(f"\n-> Loaded {len(scores_df)} variants.")

    # Summary table
    summary = build_summary_table(scores_df)
    print("\n" + summary.to_string(index=False))

    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n-> Saved summary table: {OUTPUT_CSV}")

    # Get baseline scores
    baseline_row = scores_df[scores_df["variant"].str.contains("Baseline")].iloc[0]

    # Render PDF
    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Summary table
        fig, ax = plt.subplots(figsize=(14, len(scores_df) * 0.6 + 2))
        ax.axis("off")

        col_labels = ["Experiment"] + [
            f"{METRIC_LABELS[m]}\nScore    Δ" for m in METRICS
        ]

        table_data = []
        for _, row in summary.iterrows():
            r = [row["Experiment"]]
            for m in METRICS:
                label = METRIC_LABELS[m]
                score = row.get(f"{label} Score", "N/A")
                delta = row.get(f"{label} Δ", "N/A")
                sign  = "+" if isinstance(delta, float) and delta >= 0 else ""
                r.append(f"{score:.4f}   {sign}{delta:.2f}" if isinstance(score, float) else "N/A")
            table_data.append(r)

        tbl = ax.table(
            cellText=table_data,
            colLabels=["Experiment", "Faithfulness\nScore  Δ",
                       "Answer Relevancy\nScore  Δ",
                       "Context Precision\nScore  Δ",
                       "Context Recall\nScore  Δ"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.8)

        # Color header row
        for j in range(5):
            tbl[0, j].set_facecolor("#1565C0")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Color baseline row
        for j in range(5):
            tbl[1, j].set_facecolor("#E3F2FD")

        ax.set_title("Table: RAGAS Results and Delta vs Baseline (Δ)",
                     fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2: 4 charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("RAGAS Metrics Comparison — All Variants", fontsize=14, fontweight="bold")

        for ax, metric in zip(axes.flatten(), METRICS):
            plot_metric_chart(ax, scores_df, metric, baseline_row[metric])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    print(f"-> Saved comparison report: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
