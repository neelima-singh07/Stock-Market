"""Central configuration: paths, model names, retrieval hyper-parameters."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PRICES_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"
GROUND_TRUTH_PATH = PROCESSED_DIR / "ground_truth.json"
META_PATH = PROCESSED_DIR / "meta.json"
EVAL_RESULTS_PATH = PROCESSED_DIR / "eval_results.json"

TICKERS: tuple[str, ...] = ("NVDA", "AAPL", "TSLA")
COMPANY_NAMES: dict[str, str] = {"NVDA": "Nvidia", "AAPL": "Apple", "TSLA": "Tesla"}

# Retrieval
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # alt: "BAAI/bge-reranker-base"
WINDOW_BEFORE_HOURS = 48
WINDOW_AFTER_HOURS = 12
POOL_SIZE = 50       # per-retriever pool before fusion
CANDIDATE_K = 20     # fused candidates passed to the cross-encoder
FINAL_K = 5          # chunks passed to the LLM
RRF_K = 60

# LLM
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
