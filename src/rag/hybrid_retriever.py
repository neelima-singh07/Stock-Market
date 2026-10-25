"""Two-stage, time-bounded hybrid retrieval.

Stage 1: Qdrant pre-filter (ticker == X AND time window) -> dense (bge-small) + BM25 -> RRF -> top-20.
Stage 2: cross-encoder rerank -> top-5.

Retrieval modes (used by the benchmark for ablations):
  naive          dense only, no metadata filter, no rerank   (baseline)
  time_dense     ticker + time filter, dense only
  hybrid         ticker + time filter, dense + BM25 fused with RRF
  hybrid_rerank  hybrid + cross-encoder rerank               (full pipeline)

Time filtering: every point stores an integer Unix timestamp (`ts`, filtered with `Range`) and an
RFC-3339 string (`published_at`, filtered with `DatetimeRange`). `time_filter="int_range"` is the
default; `"datetime_range"` switches to the native datetime filter.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.anomaly.detector import AnomalyEvent
from src.config import (
    BGE_QUERY_PREFIX, CANDIDATE_K, COMPANY_NAMES, DENSE_MODEL, FINAL_K, POOL_SIZE, RERANK_MODEL, RRF_K,
    WINDOW_AFTER_HOURS, WINDOW_BEFORE_HOURS,
)
from src.schemas import Document

logger = logging.getLogger(__name__)

RetrievalMode = Literal["naive", "time_dense", "hybrid", "hybrid_rerank"]
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Keeps 'item', '2.02', '8-k' and ticker symbols as exact tokens for BM25."""
    return _TOKEN_RE.findall(text.lower())


def build_queries(event: AnomalyEvent) -> tuple[str, str]:
    """Returns (natural-language query for dense + reranker, keyword query for BM25)."""
    name = COMPANY_NAMES.get(event.ticker, event.ticker)
    verb = "surged" if event.direction == "up" else "plunged"
    nl = (f"Why did {name} ({event.ticker}) stock {verb} {abs(event.pct_change):.1f}% on "
          f"{event.timestamp:%B %d, %Y}? Earnings results, guidance, regulatory filing or major announcement.")
    kw = f"{event.ticker} {name} 8-K Item 2.02 Item 7.01 Item 8.01 earnings guidance revenue announcement filing"
    return nl, kw


class RetrievedChunk(BaseModel):
    doc: Document
    final_rank: int = 0
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    event_id: str
    mode: str
    dense_query: str
    sparse_query: str
    window_start: datetime | None = None
    window_end: datetime | None = None
    n_filtered: int | None = None
    candidates: list[RetrievedChunk]
    top: list[RetrievedChunk]


class HybridRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        dense_model_name: str = DENSE_MODEL,
        reranker_model_name: str = RERANK_MODEL,
        qdrant_path: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection: str = "alphashock_docs",
        time_filter: Literal["int_range", "datetime_range"] = "int_range",
        rrf_k: int = RRF_K,
        pool_size: int = POOL_SIZE,
    ) -> None:
        if not documents:
            raise ValueError("HybridRetriever needs at least one document")
        self.documents = list(documents)
        self.collection = collection
        self.time_filter = time_filter
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        self.reranker_model_name = reranker_model_name
        self._reranker: CrossEncoder | None = None
        self.embedder = SentenceTransformer(dense_model_name)
        self._local_qdrant = qdrant_url is None
        if self._local_qdrant:
            self.client = QdrantClient(path=qdrant_path) if qdrant_path else QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._build_index()
        self._bm25 = BM25Okapi([tokenize(d.full_text) for d in self.documents])

    # ------------------------------------------------------------------ indexing
    def _build_index(self) -> None:
        dim = self.embedder.get_sentence_embedding_dimension()
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if not self._local_qdrant:
            for field, schema in (("ticker", models.PayloadSchemaType.KEYWORD),
                                  ("ts", models.PayloadSchemaType.INTEGER),
                                  ("published_at", models.PayloadSchemaType.DATETIME)):
                try:
                    self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
                except Exception as exc:  # pragma: no cover
                    logger.debug("payload index %s skipped: %s", field, exc)

        vectors = self.embedder.encode([d.full_text for d in self.documents], batch_size=32,
                                       normalize_embeddings=True, show_progress_bar=False)
        batch = 128
        for start in range(0, len(self.documents), batch):
            pts = []
            for i in range(start, min(start + batch, len(self.documents))):
                d = self.documents[i]
                pts.append(models.PointStruct(
                    id=i, vector=np.asarray(vectors[i]).tolist(),
                    payload={"doc_id": d.doc_id, "ticker": d.ticker, "ts": d.unix_ts,
                             "published_at": d.published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "source": d.source, "item": d.item},
                ))
            self.client.upsert(collection_name=self.collection, points=pts)

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    # ------------------------------------------------------------------ filters
    def window_for(self, event: AnomalyEvent) -> tuple[datetime, datetime]:
        return (event.timestamp - timedelta(hours=WINDOW_BEFORE_HOURS),
                event.timestamp + timedelta(hours=WINDOW_AFTER_HOURS))

    def build_filter(self, ticker: str, start: datetime, end: datetime) -> models.Filter:
        if self.time_filter == "datetime_range":
            time_cond = models.FieldCondition(key="published_at", range=models.DatetimeRange(gte=start, lte=end))
        else:
            time_cond = models.FieldCondition(
                key="ts", range=models.Range(gte=int(start.timestamp()), lte=int(end.timestamp())))
        return models.Filter(must=[
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker)),
            time_cond,
        ])

    def _filtered_ids(self, flt: models.Filter) -> set[int]:
        ids: set[int] = set()
        offset = None
        while True:
            pts, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=256, offset=offset,
                                             with_payload=False, with_vectors=False)
            ids.update(int(p.id) for p in pts)
            if offset is None:
                return ids

    # ------------------------------------------------------------------ retrievers
    def _dense(self, query: str, flt: models.Filter | None, limit: int) -> list[tuple[int, float]]:
        qv = self.embedder.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True)
        res = self.client.query_points(collection_name=self.collection, query=np.asarray(qv).tolist(),
                                       query_filter=flt, limit=limit, with_payload=False)
        return [(int(p.id), float(p.score)) for p in res.points]

    def _bm25_search(self, query: str, allowed: set[int], limit: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(((i, float(scores[i])) for i in allowed), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked if s > 0][:limit]

    def _rrf(self, *rankings: list[int]) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        return fused

    # ------------------------------------------------------------------ public API
    def retrieve(self, event: AnomalyEvent, mode: RetrievalMode = "hybrid_rerank",
                 candidate_k: int = CANDIDATE_K, final_k: int = FINAL_K) -> RetrievalResult:
        dense_q, sparse_q = build_queries(event)
        base = dict(event_id=event.event_id, mode=mode, dense_query=dense_q, sparse_query=sparse_q)

        if mode == "naive":
            hits = self._dense(dense_q, None, candidate_k)
            cands = [self._chunk(i, dense=(s, r)) for r, (i, s) in enumerate(hits, 1)]
            return RetrievalResult(**base, candidates=cands, top=self._rank(cands[:final_k]))

        start, end = self.window_for(event)
        flt = self.build_filter(event.ticker, start, end)
        allowed = self._filtered_ids(flt)
        base.update(window_start=start, window_end=end, n_filtered=len(allowed))
        if not allowed:
            return RetrievalResult(**base, candidates=[], top=[])

        dense_hits = self._dense(dense_q, flt, self.pool_size)
        dense_map = {i: (s, r) for r, (i, s) in enumerate(dense_hits, 1)}

        if mode == "time_dense":
            cands = [self._chunk(i, dense=dense_map[i]) for i, _ in dense_hits[:candidate_k]]
            return RetrievalResult(**base, candidates=cands, top=self._rank(cands[:final_k]))

        bm_hits = self._bm25_search(sparse_q, allowed, self.pool_size)
        bm_map = {i: (s, r) for r, (i, s) in enumerate(bm_hits, 1)}
        fused = self._rrf([i for i, _ in dense_hits], [i for i, _ in bm_hits])
        order = sorted(fused, key=lambda i: -fused[i])[:candidate_k]
        cands = [self._chunk(i, dense=dense_map.get(i), bm25=bm_map.get(i), rrf=fused[i]) for i in order]

        if mode == "hybrid":
            return RetrievalResult(**base, candidates=cands, top=self._rank(cands[:final_k]))

        # hybrid_rerank
        if cands:
            scores = self.reranker.predict([(dense_q, c.doc.full_text) for c in cands],
                                           batch_size=16, show_progress_bar=False)
            for c, s in zip(cands, scores):
                c.rerank_score = float(s)
        reranked = sorted(cands, key=lambda c: -(c.rerank_score if c.rerank_score is not None else -1e9))
        return RetrievalResult(**base, candidates=cands, top=self._rank(reranked[:final_k]))

    # ------------------------------------------------------------------ helpers
    def _chunk(self, idx: int, dense: tuple[float, int] | None = None,
               bm25: tuple[float, int] | None = None, rrf: float | None = None) -> RetrievedChunk:
        return RetrievedChunk(
            doc=self.documents[idx],
            dense_score=dense[0] if dense else None, dense_rank=dense[1] if dense else None,
            bm25_score=bm25[0] if bm25 else None, bm25_rank=bm25[1] if bm25 else None,
            rrf_score=rrf,
        )

    @staticmethod
    def _rank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        out = []
        for r, c in enumerate(chunks, 1):
            out.append(c.model_copy(update={"final_rank": r}))
        return out
