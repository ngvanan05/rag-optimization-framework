"""
Stage 3 - Prompt Templates
Centralized management of Vietnamese prompt templates for the RAG system.
Extracted from: rag_pipeline.py
"""
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Template mặc định cho Baseline RAG
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = """Bạn là trợ lý AI phân tích tài liệu tiếng Việt.

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

Hãy trả lời dựa trên tài liệu. Nếu tài liệu không có thông tin, nói rõ "Không có thông tin".
[TRẢ LỜI]:"""

BASELINE_PROMPT = PromptTemplate.from_template(BASELINE_TEMPLATE)


# ---------------------------------------------------------------------------
# Thêm các template khác tại đây khi thử nghiệm
# ---------------------------------------------------------------------------

# VD: Template yêu cầu trả lời ngắn gọn
# CONCISE_TEMPLATE = """..."""
# CONCISE_PROMPT = PromptTemplate.from_template(CONCISE_TEMPLATE)


if __name__ == "__main__":
    sample = BASELINE_PROMPT.format(
        context="Đây là nội dung tài liệu mẫu.",
        question="Tài liệu nói về gì?"
    )
    print(sample)
