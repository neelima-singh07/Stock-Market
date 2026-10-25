"""Loaders for prices, documents and ground truth produced by bootstrap_data.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import DOCUMENTS_PATH, GROUND_TRUTH_PATH, PRICES_DIR, TICKERS
from src.schemas import Document, GroundTruthEvent


def load_prices(ticker: str, prices_dir: Path = PRICES_DIR) -> pd.DataFrame:
    path = prices_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.bootstrap_data")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_documents(path: Path = DOCUMENTS_PATH) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.bootstrap_data")
    with path.open("r", encoding="utf-8") as fh:
        return [Document.model_validate_json(line) for line in fh if line.strip()]


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[GroundTruthEvent]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.bootstrap_data")
    return [GroundTruthEvent.model_validate(x) for x in json.loads(path.read_text("utf-8"))]


def data_ready() -> bool:
    if not DOCUMENTS_PATH.exists() or not all((PRICES_DIR / f"{t}.csv").exists() for t in TICKERS):
        return False

    try:
        documents = load_documents()
        if not documents:
            return False
        price_ranges = {
            ticker: (
                pd.Timestamp(load_prices(ticker)["date"].min(), tz="UTC"),
                pd.Timestamp(load_prices(ticker)["date"].max(), tz="UTC") + pd.Timedelta(days=1),
            )
            for ticker in TICKERS
        }
        truth = load_ground_truth()
    except (OSError, ValueError, KeyError):
        return False

    # A live-news-only corpus can be valid JSON but still be unrelated to the
    # historical prices used by the detector. Require ticker/date overlap.
    has_document_overlap = any(
        price_ranges[d.ticker][0] <= d.published_at < price_ranges[d.ticker][1]
        for d in documents
        if d.ticker in price_ranges
    )
    if not has_document_overlap:
        return False

    # The default corpus also includes labelled historical events. Reject a
    # mixed bootstrap where prices were refreshed but ground-truth dates were not.
    return all(
        price_ranges[event.ticker][0] <= pd.Timestamp(event.date, tz="UTC") < price_ranges[event.ticker][1]
        for event in truth
        if event.ticker in price_ranges
    )
