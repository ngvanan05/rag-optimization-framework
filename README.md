

<div align="center">
<h1>RAG Optimization Framework</h1>
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2--0.3-purple.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.15-orange.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

**End-to-end framework xây dựng, tối ưu và đánh giá hệ thống RAG trên tài liệu tiếng Việt.**

[Tổng quan](#tổng-quan) · [Kiến trúc](#kiến-trúc) · [Cài đặt](#cài-đặt) · [Hướng dẫn chạy](#hướng-dẫn-chạy) · [Kết quả](#kết-quả-thực-nghiệm) · [EXPERIMENT.md](EXPERIMENT.md)

</div>

---

## Tổng quan

Tài liệu nội bộ Đoàn-Hội sinh viên (25 file PDF/DOCX) được sinh viên tra cứu thủ công — tốn thời gian và dễ bỏ sót thông tin. Dự án này xây dựng một **hệ thống hỏi đáp tự động (RAG)** trên corpus đó, đồng thời nghiên cứu có hệ thống **5 kỹ thuật nâng cấp** để xác định phương án tối ưu nhất, đo lường bằng bộ đánh giá [RAGAS](https://docs.ragas.io/).

> **Vấn đề:** Baseline RAG với fixed-size chunking chỉ đạt Answer Relevancy **0.575** — không đủ để triển khai thực tế.  
> **Kết quả:** Combo v1+v2+v4 cải thiện Answer Relevancy lên **+23.9%**, Context Recall **+14.2%**.

Framework được tổ chức thành 3 phase rõ ràng:

| Phase | Mục tiêu |
|-------|----------|
| **Phase 1** | Tự động sinh bộ câu hỏi-đáp (testset) từ tài liệu bằng RAGAS |
| **Phase 2** | Xây dựng baseline RAG pipeline và đánh giá 4 metrics |
| **Phase 3** | Thử nghiệm 5 kỹ thuật nâng cấp, kết hợp thành combo pipeline |

---

## Kiến trúc

```
Tài liệu PDF/DOCX
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
│  Phase 3: Upgrades (modular, có thể kết hợp)            │
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

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| Python | 3.10+ |
| GPU VRAM | ≥ 6 GB (NVIDIA CUDA) |
| RAM | ≥ 16 GB |
| Docker | Để chạy Qdrant |
| OpenAI API Key | GPT-4o-mini |

> **Lưu ý VRAM:** BGE-M3 (~1.6 GB) và Qwen3-Reranker-0.6B (~1.2 GB) không thể chạy đồng thời trên GPU 6 GB. Các pipeline có cả 2 model được tách thành 2 bước riêng biệt (bước 1 giải phóng GPU trước khi bước 2 chạy).

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/ngvanan05/rag-optimization-framework.git
cd rag-optimization-framework
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý PyTorch:** `pip install -r requirements.txt` cài CPU-only build.  
> Để dùng GPU (CUDA 12.4), chạy thêm:
> ```bash
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```

### 4. Cấu hình API key

Tạo file `.env` ở thư mục gốc:

```env
OPENAI_API_KEY=sk-...
```

### 5. Khởi động Qdrant

```bash
# Lần đầu (tạo container và mount storage)
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/database:/qdrant/storage qdrant/qdrant

# Tự động khởi động cùng Docker
docker update --restart always qdrant

# Các lần sau
docker start qdrant
```

---

## Cấu trúc dự án

```
RAGOptima/
│
├── app.py                          # Streamlit Q&A app (core: v1+v2+v4+v5)
├── compare_results.py              # So sánh tất cả variant → CSV + PDF
├── requirements.txt
├── .env                            # API keys (không commit)
│
├── data/
│   ├── markdown/                   # PDF đã convert sang Markdown
│   └── ragas_testset.json          # Testset 50 câu hỏi (sinh bởi Phase 1)
│
├── database/                       # Qdrant persistent storage (Docker mount)
│
├── phase1_testset_gen/             # Sinh testset tự động
│   ├── config.py
│   ├── step1_convert.py            # PDF → Markdown
│   ├── step2_models.py             # LLM/Embedding wrapper cho RAGAS
│   ├── step3_generate.py           # Sinh Q&A pairs
│   └── run_testset.py              # Entry point
│
├── phase2_baseline/                # Baseline RAG pipeline
│   ├── config.py                   # Cấu hình toàn cục (hyperparameters)
│   ├── models.py                   # GPT-4o-mini + BGE-M3
│   ├── run_pipeline.py             # Inference → CSV
│   ├── run_eval.py                 # RAGAS evaluation → PDF report
│   ├── stage1_indexing/
│   │   ├── loader.py               # Load PDF/DOCX, clean text tiếng Việt
│   │   └── vector_db.py            # Chunk + index vào Qdrant
│   ├── stage2_retrieval/
│   │   └── search_engine.py        # Retrieve top-K chunks
│   └── stage3_generation/
│       ├── prompt_templates.py
│       └── rag_chain.py            # LLM + context → answer
│
├── phase3_upgrades/                # Các kỹ thuật nâng cấp
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
│   │   ├── runv4_index_retrieve.py # Bước 1: Index + Retrieve (BGE-M3 GPU)
│   │   └── runv4_rerank_generate.py# Bước 2: Rerank + Generate (Qwen3 GPU)
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
└── outputs/                        # Kết quả thực nghiệm
    ├── images/                     # Chart và screenshot
    ├── phase2_baseline/            # CSV + PDF report baseline
    ├── phase3_upgrades/            # CSV + PDF report từng variant
    ├── full_pipeline/              # CSV + PDF report full combo
    └── semantic_decompose_rerank/  # CSV + PDF report combo v1+v2+v4
```

---

## Hướng dẫn chạy

### Phase 1 — Sinh testset

Chuyển đổi tài liệu và sinh 50 câu hỏi Q&A tự động bằng RAGAS:

```bash
cd phase1_testset_gen
python run_testset.py
```

Output: `data/ragas_testset.json`

> ⚠️ Phase 1 có thể cần môi trường Python riêng do conflict giữa `transformers` và `sentence-transformers`. Nếu gặp lỗi, thử tạo venv mới và cài chỉ các dependency của phase này.

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

### Phase 3 — Các nâng cấp

Mỗi version chạy độc lập, chỉ thay thế đúng 1 component so với baseline:

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

#### v4 — Cross-Encoder Reranking (2 bước do VRAM 6 GB)
```bash
# Bước 1: BGE-M3 GPU — index + retrieve, lưu trung gian ra file
python phase3_upgrades/v4_reranking/runv4_index_retrieve.py

# Bước 2: Qwen3 GPU — rerank + generate (chạy SAU khi bước 1 kết thúc)
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
# Bước 1: BGE-M3 GPU
python phase3_upgrades/full_pipeline/run_step1_chunk_index_retrieve.py

# Bước 2: Qwen3 GPU
python phase3_upgrades/full_pipeline/run_step2_rerank_generate.py

python phase2_baseline/run_eval.py --input outputs/full_pipeline/full_pipeline_results.csv
```

#### Semantic Decompose Rerank — **Best Recall** (v1 + v2 + v4)
`SemanticChunk → DenseIndex → QueryDecompose → Rerank → Generate`

```bash
# Bước 1: BGE-M3 GPU
python phase3_upgrades/semantic_decompose_rerank/run_step1_chunk_retrieve.py

# Bước 2: Qwen3 GPU
python phase3_upgrades/semantic_decompose_rerank/run_step2_rerank_generate.py

python phase2_baseline/run_eval.py --input outputs/semantic_decompose_rerank/semantic_decompose_rerank_results.csv
```

---

### So sánh kết quả

Sau khi chạy tất cả các variant, tổng hợp và vẽ biểu đồ so sánh:

```bash
python compare_results.py
```

Output:
- `outputs/comparison_summary.csv` — bảng điểm + delta so với baseline
- `outputs/comparison_report.pdf` — biểu đồ bar chart 4 metrics

---

### Streamlit App

App hỏi đáp real-time sử dụng core **v1 + v2 + v4 + v5** (SemanticChunk đã index sẵn, QueryDecompose + Rerank + Citation online):

> **Yêu cầu:** Đã chạy indexing ít nhất 1 lần (`runv1.py` hoặc `run_step1`) để Qdrant có collection `rag_doanhoi_docs`.

```bash
streamlit run app.py
```

Truy cập: `http://localhost:8501`

![Giao diện Streamlit — nhập câu hỏi, xem câu trả lời kèm nguồn tham khảo](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/DEMO.png)

**Tính năng:**
- Nhập câu hỏi → tự động phân tách thành sub-questions (v2)
- Retrieve từ Qdrant → Rerank bằng Qwen3 (v4)
- Sinh câu trả lời kèm trích dẫn nguồn `[Doc 1]`, `[Doc 2]`... (v5)
- Hiển thị top-3 đoạn tài liệu tham chiếu

---

## Kết quả thực nghiệm

![Bảng so sánh điểm RAGAS](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/comparison_report_page-0001.jpg)

![Biểu đồ so sánh 4 metrics](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/comparison_report_page-0002.jpg)

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

> **Bold** = best score per metric (tied values đều được bôi đậm). Δ = delta vs Baseline.  
> Xem phân tích chi tiết và RAGAS report từng variant: [EXPERIMENT.md](EXPERIMENT.md)

---

## Known Limitations

- **v5 CoT+Citation giảm Faithfulness (−6.5%):** Block `<think>` trong output của Qwen3 reasoning có thể làm nhiễu RAGAS faithfulness scorer — không khuyến nghị dùng khi cần tối đa Faithfulness.
- **GPU VRAM:** BGE-M3 (~1.6 GB) và Qwen3-Reranker (~1.2 GB) không chạy song song trên GPU ≤ 8 GB. Các script v4/full_pipeline được tách 2 bước để giải quyết.
- **Phase 1 dependency conflict:** `run_testset.py` có thể gặp conflict giữa `transformers` và `sentence-transformers`. Khuyến nghị dùng venv riêng cho phase này.
- **RAGAS judge bias:** GPT-4o-mini đóng vai trò vừa là generator vừa là judge — có thể gây thiên lệch đánh giá (self-consistency bias). Kết quả nên được xem là tương đối, không tuyệt đối.

---

## Cấu hình

Tất cả cấu hình tập trung tại `phase2_baseline/config.py`:

| Tham số | Giá trị mặc định | Mô tả |
|---------|-----------------|-------|
| `LLM_MODEL` | `gpt-4o-mini` | Model sinh câu trả lời |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Model embedding |
| `CHUNK_SIZE` | `1024` | Kích thước chunk (ký tự) |
| `CHUNK_OVERLAP` | `128` | Overlap giữa các chunk |
| `TOP_K` | `3` | Số docs trả về sau retrieve/rerank |
| `RETRIEVE_K` | `10` | Số docs lấy từ vector DB trước khi rerank |
| `QDRANT_URL` | `http://localhost:6333` | Địa chỉ Qdrant server |
| `COLLECTION_NAME` | `rag_doanhoi_docs` | Tên collection trong Qdrant |
| `BATCH_SIZE` | `32` | Batch size cho LLM inference |
| `SEED` | `42` | Random seed cho reproducibility |

---

## Stack công nghệ

| Thành phần | Công nghệ |
|------------|-----------|
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
