# Nhật ký thực nghiệm

**[English](EXPERIMENT.md) · Tiếng Việt**

Kết quả và phân tích chi tiết cho tất cả các variant RAG được đánh giá trên corpus tài liệu tiếng Việt  
(tài liệu sinh viên Đoàn-Hội — **50 cặp Q&A**, sinh bởi RAGAS).

> Lệnh chạy lại đầy đủ: [Reproduce](#reproduce) · Biểu đồ tổng hợp: [Comparison](#comparison)

---

## Thiết lập

| Tham số | Giá trị |
|-----------|-------|
| LLM (Generator) | GPT-4o-mini (temperature=0, seed=42) |
| LLM (RAGAS Judge) | GPT-4o-mini ⚠️ *cùng model — xem ghi chú bên dưới* |
| Embedding | BAAI/bge-m3 (GPU FP16) |
| Reranker | Qwen3-Reranker-0.6B (GPU FP16) |
| Chunk size | 1024 ký tự, overlap 128 |
| Top-K retrieve | 3 (baseline) / 10 → 3 (có reranker) |
| Kích thước testset | 50 cặp Q&A |
| Framework đánh giá | RAGAS 0.2.15 — 4 metrics |

> ⚠️ **Ghi chú về judge bias:** GPT-4o-mini đóng vai trò vừa là model sinh câu trả lời vừa là judge của RAGAS. Điều này có thể gây ra thiên lệch tự nhất quán (self-consistency bias) — điểm số nên được xem là mức cải thiện tương đối giữa các variant, không phải chất lượng tuyệt đối.

---

## Kết quả

| Variant | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---------|:-----------:|:----------------:|:-----------------:|:--------------:|
| Baseline (Fixed Chunking) | 0.8550 | 0.5751 | 0.8873 | 0.6886 |
| v1 Semantic Chunking | 0.8548 | 0.6564 | **0.9134** | 0.7686 |
| v2 Query Decomposition | 0.8966 | **0.7134** | 0.8395 | 0.7905 |
| v3 Hybrid Search (RRF) | 0.8589 | 0.6261 | 0.9199 | 0.6814 |
| v4 Cross-Encoder Reranking | **0.9112** | 0.6629 | 0.9330 | 0.7673 |
| v5 CoT + Citation | 0.7897 | 0.6588 | 0.9085 | 0.6846 |
| Full Pipeline (v1+v2+v3+v4) | 0.9021 | 0.6784 | 0.9232 | 0.7510 |
| Combo v1+v2+v4 | 0.8705 | 0.6389 | **0.9330** | **0.7863** |

> **In đậm** = điểm cao nhất theo từng metric (giá trị bằng nhau đều được in đậm).

---

## Delta so với Baseline

| Variant | Δ Faithfulness | Δ Answer Relevancy | Δ Context Precision | Δ Context Recall |
|---------|:--------------:|:------------------:|:-------------------:|:----------------:|
| v1 Semantic Chunking | −0.0002 | +0.0813 | +0.0261 | +0.0800 |
| v2 Query Decomposition | +0.0416 | **+0.1383** | −0.0478 | +0.1019 |
| v3 Hybrid Search (RRF) | +0.0039 | +0.0510 | +0.0326 | −0.0072 |
| v4 Cross-Encoder Reranking | **+0.0562** | +0.0878 | **+0.0457** | +0.0787 |
| v5 CoT + Citation | −0.0653 | +0.0837 | +0.0212 | −0.0040 |
| Full Pipeline (v1+v2+v3+v4) | +0.0471 | +0.1033 | +0.0359 | +0.0624 |
| Combo v1+v2+v4 | +0.0155 | +0.0638 | +0.0457 | **+0.0977** |

---

## Phát hiện chính

**Điểm cao nhất theo từng metric:**
- 🥇 **Faithfulness**: v4 Cross-Encoder Reranking — 0.9112 (+6.6%)
- 🥇 **Answer Relevancy**: v2 Query Decomposition — 0.7134 (+23.9%)
- 🥇 **Context Precision**: v4 & Combo v1+v2+v4 — 0.9330 (+5.1%, đồng hạng)
- 🥇 **Context Recall**: Combo v1+v2+v4 — 0.7863 (+14.2%)

**Nhận xét:**

- **v2 Query Decomposition** mang lại mức cải thiện Answer Relevancy lớn nhất (+13.8%) nhờ tách câu hỏi phức tạp thành các sub-question có mục tiêu rõ ràng, giúp retriever tìm được ngữ cảnh liên quan hơn cho từng phần câu hỏi.

- **v4 Cross-Encoder Reranking** liên tục cải thiện Faithfulness (+5.6%) và Context Precision (+4.6%) bằng cách lọc bỏ các chunk lạc đề trước khi sinh câu trả lời — đây là bản nâng cấp đơn lẻ có tác động lớn nhất.

- **v3 Hybrid Search (RRF)** cải thiện Context Precision (+3.3%) nhưng làm giảm nhẹ Context Recall (−0.7%). BM25 trên văn bản tiếng Việt khi chưa có tokenization chuyên biệt (underthesea chưa được tinh chỉnh) có thể gây nhiễu với các câu hỏi ít từ khóa đặc trưng. Khi kết hợp với v1 SemanticChunk (giúp cải thiện chất lượng tokenization), hiệu ứng trở thành tích cực trong Full Pipeline.

- **v5 CoT+Citation làm giảm Faithfulness (−6.5%):** Block `<think>` được Qwen3 thêm vào đầu output reasoning dường như gây nhiễu cho bộ chấm điểm faithfulness của RAGAS — vốn so sánh các câu trong câu trả lời với ngữ cảnh đã retrieve. Các câu reasoning dài dòng không khớp rõ ràng với các chunk ngữ cảnh.

- **Full Pipeline (v1+v2+v3+v4)** là combo cân bằng nhất trên cả 4 metric — không có metric nào bị hy sinh đáng kể.

- **Combo v1+v2+v4** đạt Context Recall tốt nhất (+14.2%) với ít thành phần hơn Full Pipeline — việc bỏ v3 thực ra lại giúp Recall tốt hơn, nhất quán với việc v3 đứng riêng lẻ làm giảm recall.

**Khuyến nghị:**

| Use case | Variant tốt nhất | Lý do |
|----------|:------------:|--------|
| Ưu tiên độ chính xác (giảm thiểu hallucination) | Full Pipeline (v1+v2+v3+v4) | Cân bằng nhất; Faithfulness + Precision cao nhất khi kết hợp |
| Ưu tiên recall (không bỏ sót thông tin liên quan) | Combo v1+v2+v4 | Context Recall tốt nhất (+14.2%) |
| Triển khai nhanh (độ phức tạp tối thiểu) | Chỉ v4 | Faithfulness tốt nhất chỉ với 1 component thêm vào |
| Tránh dùng | v5 đứng riêng | Làm giảm Faithfulness; định dạng citation nên dùng trong app, không dùng khi eval |

---

## Báo cáo RAGAS

Biểu đồ chi tiết từng variant (xuất từ `run_eval.py`):

### Baseline
![Baseline RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/rag_inference_results_RAGAS_Report_page-0001.jpg)

### v1 — Semantic Chunking
![v1 RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/v1_semantic_chunker_RAGAS_Report_page-0001.jpg)

### v2 — Query Decomposition
![v2 RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/v2_query_decomposition_RAGAS_Report_page-0001.jpg)

### v3 — Hybrid Search (RRF)
![v3 RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/v3_hybrid_search_RAGAS_Report_page-0001.jpg)

### v4 — Cross-Encoder Reranking
![v4 RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/v4_reranking_RAGAS_Report_page-0001.jpg)

### v5 — CoT + Citation
![v5 RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/v5_cot_citation_RAGAS_Report_page-0001.jpg)

### Full Pipeline (v1+v2+v3+v4)
![Full Pipeline RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/full_pipeline_results_RAGAS_Report_page-0001.jpg)

### Combo v1+v2+v4
![Combo v1+v2+v4 RAGAS Report](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/semantic_decompose_rerank_results_RAGAS_Report_page-0001.jpg)

---

## Comparison

![Bảng so sánh điểm RAGAS — tất cả variants](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/comparison_report_page-0001.jpg)

![Biểu đồ bar chart 4 metrics](https://raw.githubusercontent.com/ngvanan05/rag-optimization-framework/main/outputs/images/comparison_report_page-0002.jpg)

---

## Reproduce

```bash
# === Phase 2: Baseline ===
python phase2_baseline/run_pipeline.py
python phase2_baseline/run_eval.py

# === Phase 3: Single-component upgrades ===
python phase3_upgrades/v1_indexing_chunking/runv1.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v1_semantic_chunker.csv

python phase3_upgrades/v2_query_transformation/runv2.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v2_query_decomposition.csv

python phase3_upgrades/v3_retrieval_strategy/runv3.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v3_hybrid_search.csv

# v4: 2 bước do giới hạn VRAM (BGE-M3 + Qwen3 không thể chạy đồng thời trên GPU 6 GB)
python phase3_upgrades/v4_reranking/runv4_index_retrieve.py   # Bước 1: BGE-M3
python phase3_upgrades/v4_reranking/runv4_rerank_generate.py  # Bước 2: Qwen3
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v4_reranking.csv

python phase3_upgrades/v5_generation_advanced/runv5.py
python phase2_baseline/run_eval.py --input outputs/phase3_upgrades/v5_cot_citation.csv

# === Combo: Full Pipeline (v1+v2+v3+v4) ===
python phase3_upgrades/full_pipeline/run_step1_chunk_index_retrieve.py  # Bước 1: BGE-M3
python phase3_upgrades/full_pipeline/run_step2_rerank_generate.py        # Bước 2: Qwen3
python phase2_baseline/run_eval.py --input outputs/full_pipeline/full_pipeline_results.csv

# === Combo: Semantic Decompose Rerank (v1+v2+v4) — Best Recall ===
python phase3_upgrades/semantic_decompose_rerank/run_step1_chunk_retrieve.py  # Bước 1: BGE-M3
python phase3_upgrades/semantic_decompose_rerank/run_step2_rerank_generate.py  # Bước 2: Qwen3
python phase2_baseline/run_eval.py --input outputs/semantic_decompose_rerank/semantic_decompose_rerank_results.csv

# === Generate comparison report ===
python compare_results.py
```
