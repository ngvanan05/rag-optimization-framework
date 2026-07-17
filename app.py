"""
RAG Q&A App — Tài liệu Đoàn-Hội
Core: v1 (SemanticChunk, đã index) + v2 (QueryDecompose) + v4 (Rerank) + v5 (Citation)

Chạy:
    streamlit run app.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Giảm CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import streamlit as st

from phase2_baseline.config import (
    LLM_MODEL, QDRANT_URL, COLLECTION_NAME, SEED, RETRIEVE_K, set_global_seed,
)
from phase2_baseline.models import get_llm, get_embeddings
from phase2_baseline.stage1_indexing.vector_db import VectorDB
from phase3_upgrades.v2_query_transformation.query_decomposer import DecompositionQueryTransformer
from phase3_upgrades.v4_reranking.cross_encoder import CrossEncoderReranker
from phase3_upgrades.v5_generation_advanced.cot_citation_engine import CotCitationEngine

set_global_seed(SEED)

# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def check_prerequisites() -> list[str]:
    """Kiểm tra các điều kiện cần thiết trước khi khởi động."""
    errors = []

    # Kiểm tra API key
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("❌ Thiếu OPENAI_API_KEY — tạo file `.env` với `OPENAI_API_KEY=sk-...`")

    # Kiểm tra Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=3)
        if not client.collection_exists(COLLECTION_NAME):
            errors.append(
                f"❌ Collection `{COLLECTION_NAME}` chưa tồn tại trong Qdrant — "
                f"chạy indexing trước: `python phase3_upgrades/v1_indexing_chunking/runv1.py`"
            )
    except Exception:
        errors.append(
            f"❌ Không kết nối được Qdrant tại `{QDRANT_URL}` — "
            f"khởi động Docker: `docker start qdrant`"
        )

    return errors


# ---------------------------------------------------------------------------
# Load pipeline 1 lần duy nhất (cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Đang khởi tạo pipeline (BGE-M3 + Qwen3)...")
def load_pipeline():
    llm        = get_llm(model_name=LLM_MODEL, seed=SEED)
    embeddings = get_embeddings()  # BGE-M3 GPU

    retriever = VectorDB(
        documents=None,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
    ).get_retriever(search_kwargs={"k": RETRIEVE_K})

    decomposer      = DecompositionQueryTransformer(llm)
    reranker        = CrossEncoderReranker(top_k=3, use_gpu=True)
    citation_engine = CotCitationEngine(llm)  # v5: citation generation

    return decomposer, retriever, reranker, citation_engine


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

def answer_question(question: str, decomposer, retriever, reranker, citation_engine):
    # v2: Decompose câu hỏi thành sub-questions
    sub_questions = decomposer.decompose(question)

    # Retrieve cho từng sub-question + câu gốc, dedup
    all_docs = []
    seen     = set()
    for q in sub_questions + [question]:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)

    # v4: Rerank top-10 → top-3
    top_docs = reranker.rerank(question, all_docs[:RETRIEVE_K])

    # v5: Citation — format [Doc 1], [Doc 2]... và sinh câu trả lời có trích dẫn (không CoT)
    answer = citation_engine.cite(question, top_docs)

    return answer, top_docs, sub_questions


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Q&A — Tài liệu Đoàn-Hội")
st.caption("Hệ thống hỏi đáp thông minh dựa trên tài liệu nội bộ (v1 + v2 + v4 + v5 Citation)")

# --- Kiểm tra prerequisites ---
errors = check_prerequisites()
if errors:
    st.error("Không thể khởi động app. Vui lòng kiểm tra các lỗi sau:")
    for err in errors:
        st.markdown(err)
    st.stop()

# --- Load pipeline ---
try:
    decomposer, retriever, reranker, citation_engine = load_pipeline()
    st.success("Pipeline sẵn sàng!", icon="✅")
except Exception as e:
    st.error(f"Lỗi khởi tạo pipeline: {e}")
    st.stop()

st.divider()

# --- Input ---
question = st.text_input(
    "Nhập câu hỏi của bạn:",
    placeholder="Ví dụ: Sinh viên khóa 51 cần đăng ký những hoạt động gì?",
)

submit = st.button("🔍 Gửi câu hỏi", type="primary")

if submit and question.strip():
    st.info("⏳ Đang xử lý... Quá trình reranking có thể mất vài giây.", icon="ℹ️")

    with st.spinner("Đang phân tích câu hỏi và tìm kiếm tài liệu..."):
        try:
            answer, top_docs, sub_questions = answer_question(
                question, decomposer, retriever, reranker, citation_engine
            )

            # Câu trả lời
            st.subheader("💬 Câu trả lời")
            st.write(answer)

            # Sub-questions
            if sub_questions:
                with st.expander("🔍 Câu hỏi con đã phân tách", expanded=False):
                    for i, sq in enumerate(sub_questions, 1):
                        st.write(f"{i}. {sq}")

            # Nguồn tham khảo
            st.subheader("📄 Nguồn tham khảo")
            for i, doc in enumerate(top_docs, 1):
                with st.expander(f"Tài liệu {i}", expanded=False):
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"Lỗi khi xử lý câu hỏi: {e}")

elif submit and not question.strip():
    st.warning("Vui lòng nhập câu hỏi trước khi gửi.")
