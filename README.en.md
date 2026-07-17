

<div align="center">
<h1>RAG Optimization Framework</h1>
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2--0.3-purple.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.15-orange.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

**End-to-end framework for building, optimizing, and evaluating a RAG system on Vietnamese documents.**

[Overview](#overview) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage) · [Results](#experiment-results) · [EXPERIMENT.md](EXPERIMENT.md)

English · **[Tiếng Việt](README.md)**

</div>

---

## Overview

Internal Student Union documents (25 PDF/DOCX files) were previously looked up manually by students — slow and prone to missing information. This project builds an **automated Q&A system (RAG)** over that corpus, while systematically researching **5 upgrade techniques** to identify the best-performing configuration, measured with the [RAGAS](https://docs.ragas.io/) evaluation suite.

> **Problem:** The baseline RAG with fixed-size chunking only reaches an Answer Relevancy of **0.575** — not good enough for real-world deployment.
> **Result:** The v1+v2+v4 combo improves Answer Relevancy by **+23.9%** and Context Recall by **+14.2%**.

The framework is organized into 3 clear phases:

| Phase | Goal |
|-------|------|
| **Phase 1** | Automatically generate a Q&A testset from the documents using RAGAS |
| **Phase 2** | Build the baseline RAG pipeline and evaluate it on 4 metrics |
| **Phase 3** | Experiment with 5 upgrade techniques, combined into composite pipelines |

---

## Architecture

```
PDF/DOCX Documents
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Testset Generation                            │
│  PDF → Markdown → RAGAS TestsetGenerator → JSON         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Baseline RAG                                  │
│  Loader → TextSplitter → BGE-M3 → Qdrant               │
│  Query → Dense Retrieve → GPT-4o-mini → Answer          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3: Upgrades (modular, composable)                │
│  v1: SemanticChunker    (BGE-M3 + underthesea)          │
│  v2: QueryDecomposition (GPT-4o-mini)                   │
│  v3: HybridSearch       (Qdrant + BM25 → RRF)           │
│  v4: CrossEncoder       (Qwen3-Reranker-0.6B)           │
│  v5: CoT + Citation     (GPT-4o-mini)                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  RAGAS Evaluation                                       │
│  Faithfulness | Answer Relevancy                        │
│  Context Precision | Context Recall                     │
└─────────────────────────────────────────────────────────┘
```

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| GPU VRAM | ≥ 6 GB (NVIDIA CUDA) |
| RAM | ≥ 16 GB |
| Docker | To run Qdrant |
| OpenAI API Key | GPT-4o-mini |

> **VRAM note:** BGE-M3 (~1.6 GB) and Qwen3-Reranker-0.6B (~1.2 GB) cannot run simultaneously on a 6 GB GPU. Pipelines that use both models are split into 2 separate steps (step 1 frees the GPU before step 2 runs).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ngvanan05/rag-optimization-framework.git
cd rag-optimization-framework
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Install the project in editable mode to use standard package imports (phase2_baseline.*, ...)
pip install -e .
```

> **PyTorch note:** `pip install -r requirements.txt` installs the CPU-only build.
> To use GPU (CUDA 12.4), also run:
> ```bash
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```

> **Reproducibility:** `requirements.txt` uses version ranges for flexibility. To install the exact tested versions, use `pip install -r requirements-lock.txt` instead of `requirements.txt`. The lock file is generated with `pip-compile requirements.txt -o requirements-lock.txt` (requires `pip install pip-tools`).

### 4. Configure the API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 5. Start Qdrant

```bash
# First time (create container and mount storage)
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/database:/qdrant/storage qdrant/qdrant

# Auto-start with Docker
docker update --restart always qdrant

# Subsequent runs
docker start qdrant
```

---

## Project Structure

```
RAGOptima/
│
├── app.py                          # Streamlit Q&A app (core: v1+v2+v4+v5)
├── compare_results.py              # Compares all variants → CSV + PDF
├── requirements.txt
├── .env                            # API keys (not committed)
│
├── data/
│   ├── markdown/                   # PDFs converted to Markdown
│   └── ragas_testset.json          # 50-question testset (generated by Phase 1)
│
├── database/                       # Qdrant persistent storage (Docker mount)
│
├── phase1_testset_gen/             # Automatic testset generation
│   ├── config.py
│   ├── step1_convert.py            # PDF → Markdown
│   ├── step2_models.py             # LLM/Embedding wrapper for RAGAS
│   ├── step3_generate.py           # Generate Q&A pairs
│   └── run_testset.py              # Entry point
│
├── phase2_baseline/                # Baseline RAG pipeline
│   ├── config.py                   # Global config (hyperparameters)
│   ├── models.py                   # GPT-4o-mini + BGE-M3
│   ├── run_pipeline.py             # Inference → CSV
│   ├── run_eval.py                 # RAGAS evaluation → PDF report
│   ├── stage1_indexing/
│   │   ├── loader.py               # Load PDF/DOCX, clean Vietnamese text
│   │   └── vector_db.py            # Chunk + index into Qdrant
│   ├── stage2_retrieval/
│   │   └── search_engine.py        # Retrieve top-K chunks
│   └── stage3_generation/
│       ├── prompt_templates.py
│       └── rag_chain.py            # LLM + context → answer
│
├── phase3_upgrades/                # Upgrade techniques
│   ├── v1_indexing_chunking/
│   │   ├── semantic_chunker.py     # Semantic chunking (BGE-M3 + underthesea)
│   │   └── runv1.py
│   ├── v2_query_transformation/
│   │   ├── query_decomposer.py     # Query decomposition
│   │   └── runv2.py
│   ├── v3_retrieval_strategy/
│   │   ├── hybrid_search.py        # BM25 + Qdrant → RRF
│   │   └── runv3.py
│   ├── v4_reranking/
│   │   ├── cross_encoder.py        # Qwen3-Reranker-0.6B
│   │   ├── runv4_index_retrieve.py # Step 1: Index + Retrieve (BGE-M3 GPU)
│   │   └── runv4_rerank_generate.py# Step 2: Rerank + Generate (Qwen3 GPU)
│   ├── v5_generation_advanced/
│   │   ├── cot_citation_engine.py  # CoT + Citation [Doc X]
│   │   └── runv5.py
│   ├── full_pipeline/              # Combo v1+v2+v3+v4
│   │   ├── run_step1_chunk_index_retrieve.py
│   │   └── run_step2_rerank_generate.py
│   └── semantic_decompose_rerank/  # Combo v1+v2+v4 (best recall)
│       ├── run_step1_chunk_retrieve.py
│       └── run_step2_rerank_generate.py
│
└── outputs/                        # Experiment results
    ├── images/                     # Charts and screenshots
    ├── phase2_baseline/            # Baseline CSV + PDF report
    ├── phase3_upgrades/            # CSV + PDF report per variant
    ├── full_pipeline/              # CSV + PDF report for the full combo
    └── semantic_decompose_rerank/  # CSV + PDF report for the v1+v2+v4 combo
```

---

## Usage

### Phase 1 — Generate testset

Convert documents and automatically generate 50 Q&A pairs using RAGAS:

```bash
cd phase1_testset_gen
python run_testset.py
```

Output: `data/ragas_testset.json`

> ⚠️ Phase 1 may need a separate Python environment due to a conflict between `transformers` and `sentence-transformers`. If you hit an error, try creating a new venv and installing only this phase's dependencies.

---

### Phase 2 — Baseline RAG

**Inference:**
```bash
python phase2_baseline/run_pipeline.py
```

**Evaluation:**
```bash
python phase2_baseline/run_eval.py
```

Output: `outputs/phase2_baseline/`

---

### Phase 3 — Upgrades

Each version runs independently, replacing exactly 1 component of the baseline:

#### v1 — Semantic Chunking
```bash
python phase3_upgrades/v1_indexing_chunking/runv1.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v1_semantic_chunker.csv
```

#### v2 — Query Decomposition
```bash
python phase3_upgrades/v2_query_transformation/runv2.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v2_query_decomposition.csv
```

#### v3 — Hybrid Search (BM25 + Qdrant + RRF)
```bash
python phase3_upgrades/v3_retrieval_strategy/runv3.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v3_hybrid_search.csv
```

#### v4 — Cross-Encoder Reranking (2 steps due to 6 GB VRAM)
```bash
# Step 1: BGE-M3 GPU — index + retrieve, save intermediate results to file
python phase3_upgrades/v4_reranking/runv4_index_retrieve.py

# Step 2: Qwen3 GPU — rerank + generate (run AFTER step 1 finishes)
python phase3_upgrades/v4_reranking/runv4_rerank_generate.py

python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v4_reranking.csv
```

#### v5 — CoT + Citation
```bash
python phase3_upgrades/v5_generation_advanced/runv5.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v5_cot_citation.csv
```

---

### Combo Pipelines

#### Full Pipeline (v1 + v2 + v3 + v4)
`SemanticChunk → HybridIndex → QueryDecompose → Rerank → Generate`

```bash
# Step 1: BGE-M3 GPU
python phase3_upgrades/full_pipeline/run_step1_chunk_index_retrieve.py

# Step 2: Qwen3 GPU
python phase3_upgrades/full_pipeline/run_step2_rerank_generate.py

python phase2_baseline/run_eval.py --input outputs/full_pipeline/full_pipeline_results.csv
```

#### Semantic Decompose Rerank — **Best Recall** (v1 + v2 + v4)
`SemanticChunk → DenseIndex → QueryDecompose → Rerank → Generate`

```bash
# Step 1: BGE-M3 GPU
python phase3_upgrades/semantic_decompose_rerank/run_step1_chunk_retrieve.py

# Step 2: Qwen3 GPU
python phase3_upgrades/semantic_decompose_rerank/run_step2_rerank_generate.py

python phase2_baseline/run_eval.py --input outputs/semantic_decompose_rerank/semantic_decompose_rerank_results.csv
```

---

### Comparing Results

After running all variants, aggregate and plot the comparison:

```bash
python compare_results.py
```

Output:
- `outputs/comparison_summary.csv` — score table + delta vs. baseline
- `outputs/comparison_report.pdf` — bar chart across 4 metrics

---

### Streamlit App

A real-time Q&A app using the **v1 + v2 + v4 + v5** core (SemanticChunk pre-indexed, QueryDecompose + Rerank + Citation online):

> **Requirement:** Indexing must have been run at least once (`runv1.py` or `run_step1`) so that Qdrant has the `rag_doanhoi_docs` collection.

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

![Streamlit UI — enter a question, view the answer with cited sources](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/DEMO.png)

**Features:**
- Enter a question → automatically decomposed into sub-questions (v2)
- Retrieve from Qdrant → rerank with Qwen3 (v4)
- Generate the answer with source citations `[Doc 1]`, `[Doc 2]`... (v5)
- Display the top-3 referenced document passages

---

## Experiment Results

![RAGAS score comparison table](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/comparison_report_page-0001.jpg)

![4-metric comparison chart](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/comparison_report_page-0002.jpg)

| Variant | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---------|:-----------:|:----------------:|:-----------------:|:--------------:|
| Baseline (Fixed Chunking) | 0.8550 | 0.5751 | 0.8873 | 0.6886 |
| v1 Semantic Chunking | 0.8548 (−0.00) | 0.6564 (+0.08) | **0.9134** (+0.03) | 0.7686 (+0.08) |
| v2 Query Decomposition | 0.8966 (+0.04) | **0.7134** (+0.14) | 0.8395 (−0.05) | 0.7905 (+0.10) |
| v3 Hybrid Search (RRF) | 0.8589 (+0.00) | 0.6261 (+0.05) | 0.9199 (+0.03) | 0.6814 (−0.01) |
| v4 Cross-Encoder Reranking | **0.9112** (+0.06) | 0.6629 (+0.09) | 0.9330 (+0.05) | 0.7673 (+0.08) |
| v5 CoT + Citation | 0.7897 (−0.07) | 0.6588 (+0.08) | 0.9085 (+0.02) | 0.6846 (−0.00) |
| Full Pipeline (v1+v2+v3+v4) | 0.9021 (+0.05) | 0.6784 (+0.10) | 0.9232 (+0.04) | 0.7510 (+0.06) |
| **Combo v1+v2+v4** | 0.8705 (+0.02) | 0.6389 (+0.06) | **0.9330** (+0.05) | **0.7863** (+0.10) |

> **Bold** = best score per metric (tied values are all bolded). Δ = delta vs. Baseline.
> See detailed analysis and per-variant RAGAS reports: [EXPERIMENT.md](EXPERIMENT.md)

---

## Known Limitations

- **v5 CoT+Citation reduces Faithfulness (−6.5%):** The `<think>` block in Qwen3's reasoning output can confuse the RAGAS faithfulness scorer — not recommended when Faithfulness is critical.
- **GPU VRAM:** BGE-M3 (~1.6 GB) and Qwen3-Reranker (~1.2 GB) cannot run in parallel on a GPU with ≤ 8 GB. The v4/full_pipeline scripts are split into 2 steps to work around this.
- **Phase 1 dependency conflict:** `run_testset.py` may run into a conflict between `transformers` and `sentence-transformers`. A separate venv for this phase is recommended.
- **RAGAS judge bias:** GPT-4o-mini serves as both the generator and the judge — this can introduce evaluation bias (self-consistency bias). Results should be viewed as relative, not absolute.

---

## Configuration

All configuration is centralized in `phase2_baseline/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | Answer-generation model |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding model |
| `CHUNK_SIZE` | `1024` | Chunk size (characters) |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks |
| `TOP_K` | `3` | Number of docs returned after retrieve/rerank |
| `RETRIEVE_K` | `10` | Number of docs fetched from the vector DB before reranking |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server address |
| `COLLECTION_NAME` | `rag_doanhoi_docs` | Collection name in Qdrant |
| `BATCH_SIZE` | `32` | Batch size for LLM inference |
| `SEED` | `42` | Random seed for reproducibility |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | GPT-4o-mini (OpenAI API) |
| Embedding | BGE-M3 (BAAI, GPU FP16) |
| Vector DB | Qdrant |
| Reranker | Qwen3-Reranker-0.6B (GPU FP16) |
| Vietnamese NLP | underthesea |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Evaluation | RAGAS 0.2.15 |
| Framework | LangChain 0.2–0.3 |
| App | Streamlit |

---

## License

[MIT](LICENSE) © 2025
