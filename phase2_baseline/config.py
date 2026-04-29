import os
import random
import numpy as np
import nest_asyncio
from dotenv import load_dotenv

# Kích hoạt hỗ trợ chạy async
nest_asyncio.apply()
load_dotenv()

# --- 1. REPRODUCIBILITY ---
SEED = 42

def set_global_seed(seed_value: int = 42):
    """Fix the global random seed to ensure reproducibility of experiments."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

set_global_seed(SEED)

# --- 2. API KEY ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- 3. MODEL ---
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# --- 4. DATA ---
# Dùng path tuyệt đối từ vị trí file config.py để tránh lỗi khi chạy từ thư mục khác
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = str(_PROJECT_ROOT / "data" / "data_raw")
TESTSET_PATH = str(_PROJECT_ROOT / "data" / "ragas_testset.json")

# --- 5. CHUNKING ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# --- 6. VECTOR DB ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_doanhoi_docs"
TOP_K = 3
RETRIEVE_K = 10   # Số docs lấy từ vector DB trước khi rerank (dùng ở v4 và app.py)

# --- 7. INFERENCE ---
BATCH_SIZE = 32
EMBEDDING_BATCH_SIZE = 100

# --- 8. EVALUATION ---
EVAL_SAMPLE_SIZE = None
EVAL_MAX_WORKERS = 64
EVAL_TIMEOUT = 180

if __name__ == "__main__":
    print(f"LLM: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"Data: {DATA_DIR} | Qdrant: {QDRANT_URL}")
