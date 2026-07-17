"""
Shared async batching utility for LLM inference.
Merged logic from: stage3_generation/rag_chain.py, phase3_upgrades/v2_query_transformation/query_decomposer.py,
phase3_upgrades/v4_reranking/cross_encoder.py, phase3_upgrades/v5_generation_advanced/cot_citation_engine.py
"""
import asyncio
from typing import List

from tqdm import tqdm
from langchain_core.messages import HumanMessage


def run_prompts_in_batches(llm, prompts: List[str], batch_size: int, desc: str = "Generating") -> List:
    """
    Runs llm.ainvoke over `prompts` in chunks of `batch_size`, using a fresh
    event loop per chunk. Returns the raw LLM response objects, in order.
    """
    async def _run_batch(batch_prompts):
        messages_batch = [[HumanMessage(content=p)] for p in batch_prompts]
        return await asyncio.gather(*(llm.ainvoke(m) for m in messages_batch))

    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i:i + batch_size]
        loop = asyncio.new_event_loop()
        try:
            responses = loop.run_until_complete(_run_batch(batch))
        finally:
            loop.close()
        all_responses.extend(responses)
    return all_responses
