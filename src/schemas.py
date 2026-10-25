"""Shared Pydantic schemas for corpus documents and evaluation ground truth."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, field_validator


class Document(BaseModel):
    doc_id: str
    ticker: str
    published_at: datetime
    source: str                  # "SEC 8-K", "SEC Form 4", "News Headline", ...
    title: str
    text: str
    item: str | None = None      # e.g. "Item 2.02"
    is_catalyst: bool = False    # ground-truth label (synthetic corpus only)
    event_id: str | None = None  # event this doc is a catalyst for
    synthetic: bool = True
    url: str | None = None

    @field_validator("published_at")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"

    @property
    def unix_ts(self) -> int:
        return int(self.published_at.timestamp())


class GroundTruthEvent(BaseModel):
    event_id: str
    ticker: str
    date: str                    # trading date, YYYY-MM-DD
    expected_move_pct: float
    relevant_doc_ids: list[str]
