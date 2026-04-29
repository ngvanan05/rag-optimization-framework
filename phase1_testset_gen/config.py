"""
Phase 1 — Config
Tất cả cấu hình tập trung, không import ragas/torch ở đây.
"""
from pathlib import Path

_HERE         = Path(__file__).resolve().parent   # phase1_testset_gen/
_PROJECT_ROOT = _HERE.parent                      # thư mục gốc dự án

DATA_DIR     = str(_PROJECT_ROOT / "data" / "data_raw")
MARKDOWN_DIR = str(_PROJECT_ROOT / "data" / "markdown")
TESTSET_PATH = str(_PROJECT_ROOT / "data" / "ragas_testset.json")
TESTSET_SIZE = 50

LLM_MODEL       = "gpt-4o-mini"
EMBEDDING_MODEL = "BAAI/bge-m3"
