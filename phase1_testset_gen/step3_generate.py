"""
Phase 1 — Step 3: Sinh testset bằng RAGAS TestsetGenerator
Chạy độc lập (sau khi đã có documents từ step1):
    python step3_generate.py

NOTE: Cần OPENAI_API_KEY trong .env
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from dotenv import load_dotenv
load_dotenv()

from ragas.testset import TestsetGenerator

from phase1_testset_gen.config import TESTSET_SIZE, TESTSET_PATH


def build_generator(llm_wrapper, embedding_wrapper) -> TestsetGenerator:
    """Khởi tạo RAGAS TestsetGenerator."""
    print("Khởi tạo TestsetGenerator...")
    generator = TestsetGenerator.from_langchain(
        llm=llm_wrapper,
        embedding_model=embedding_wrapper,
    )
    print("Generator sẵn sàng.\n")
    return generator


def generate_and_save(
    generator: TestsetGenerator,
    docs,
    testset_size: int = TESTSET_SIZE,
    output_file: str  = TESTSET_PATH,
):
    """
    Sinh testset với phân phối:
      50% SingleHop Specific  — câu hỏi 1 bước, 1 nguồn
      25% MultiHop Abstract   — tổng hợp nhiều nguồn
      25% MultiHop Specific   — nhiều bước, sự kiện cụ thể
    """
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 0.50),
        (MultiHopAbstractQuerySynthesizer(llm=generator.llm),  0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator.llm),  0.25),
    ]

    print(f"Sinh {testset_size} mẫu từ {len(docs)} tài liệu...")
    print("Phân phối: 50% single-hop | 25% multi-hop abstract | 25% multi-hop specific\n")

    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
        query_distribution=query_distribution,
    )
    print(f"Sinh xong {len(dataset)} mẫu\n")

    # Lưu JSON — dùng to_list() của RAGAS 0.2.x (không có to_pandas())
    try:
        records = dataset.to_list()
    except Exception:
        try:
            records = [s.model_dump() for s in dataset.samples]
        except Exception:
            records = [{"raw": str(s)} for s in dataset.samples]

    import os
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    print(f"Đã lưu: {output_file} ({len(records)} mẫu)")
    return dataset
