"""LLM catalyst synthesis with strict citation guardrails.

Providers (auto-selected, override with provider=...):
  gemini   google-genai, structured output via a Pydantic response_schema (GEMINI_API_KEY / GOOGLE_API_KEY)
  litellm  any LiteLLM model string in LITELLM_MODEL (OpenAI, Groq, ...) with JSON-mode output
  offline  deterministic extractive report (no API key needed; also the fallback on LLM errors)

Guardrails applied to every provider's output:
  * citations must be doc_ids from the evidence set; unknown ids are dropped (and counted)
  * findings left with no valid citation are removed
  * no surviving findings -> category 'unexplained', confidence 'low'
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.anomaly.detector import AnomalyEvent
from src.config import DEFAULT_LLM_MODEL
from src.rag.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class Category(str, Enum):
    earnings = "earnings"
    guidance = "guidance"
    regulatory = "regulatory"
    macro_policy = "macro_policy"
    product = "product_announcement"
    legal = "legal"
    m_and_a = "m_and_a"
    sector_sentiment = "sector_sentiment"
    other = "other"
    unexplained = "unexplained"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Finding(BaseModel):
    claim: str = Field(description="One factual statement supported by the cited documents.")
    citations: list[str] = Field(description="doc_ids (exactly as given) that support the claim.")


class LLMReport(BaseModel):
    """Schema the LLM must fill in."""
    summary: str = Field(description="2-4 sentence root-cause explanation using only the evidence.")
    primary_catalyst: str = Field(description="Short phrase naming the main catalyst, or 'Unexplained'.")
    catalyst_category: Category
    confidence: Confidence
    findings: list[Finding]
    market_impact: str = Field(description="How the catalyst plausibly maps to the observed price/volume move.")
    caveats: list[str]


class RCAReport(LLMReport):
    event_id: str
    ticker: str
    anomaly_timestamp: datetime
    z_score: float
    pct_change: float
    volume_spike_ratio: float
    evidence_doc_ids: list[str]
    generated_by: str
    dropped_citations: int = 0


SYSTEM_PROMPT = """You are a market-surveillance analyst writing a root-cause report for a statistical price/volume anomaly.

Rules (strict):
1. Use ONLY the evidence documents provided. Never use outside knowledge, never invent figures, dates or quotes.
2. Every finding must cite one or more doc_ids exactly as written in the evidence headers (e.g. NVDA-20230525-sec).
3. A catalyst must be consistent with the move's direction and timing (published before or around the move).
4. If no document plausibly explains the move, set catalyst_category to "unexplained", confidence to "low",
   primary_catalyst to "Unexplained", and return an empty findings list.
5. Confidence is "high" only when at least two independent documents agree on the same catalyst.
6. Do not give investment advice. Keep claims short and factual."""


def build_user_prompt(event: AnomalyEvent, chunks: list[RetrievedChunk]) -> str:
    stats = (
        f"ticker: {event.ticker}\n"
        f"timestamp (UTC): {event.timestamp:%Y-%m-%d %H:%M}\n"
        f"close-to-close change: {event.pct_change:+.2f}%\n"
        f"return z-score: {event.z_score:+.2f} ({event.method})\n"
        f"volume vs 20d median: {event.volume_spike_ratio:.2f}x (volume z={event.volume_z:+.2f})\n"
        f"trigger: {event.trigger}"
    )
    ev = "\n\n".join(
        f"[{c.doc.doc_id}] {c.doc.source}{' / ' + c.doc.item if c.doc.item else ''} | "
        f"published {c.doc.published_at:%Y-%m-%d %H:%M} UTC\nTitle: {c.doc.title}\nText: {c.doc.text}"
        for c in chunks
    ) or "(no evidence documents retrieved)"
    return f"ANOMALY STATISTICS\n{stats}\n\nEVIDENCE DOCUMENTS\n{ev}\n\nWrite the root-cause report."


_CATEGORY_RULES: list[tuple[str, Category]] = [
    (r"guid|outlook", Category.guidance),
    (r"earnings|quarter|revenue|results of operations", Category.earnings),
    (r"tariff|export|licen[sc]e|regulat|election|administration", Category.macro_policy),
    (r"lawsuit|subpoena|investigation|court", Category.legal),
    (r"acqui|merger", Category.m_and_a),
    (r"unveil|launch|introduc|ai system|product", Category.product),
    (r"sell-off|rout|sector|semiconductor", Category.sector_sentiment),
]


class CatalystSynthesizer:
    def __init__(self, provider: Literal["auto", "gemini", "litellm", "offline"] = "auto",
                 model: str | None = None, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self.model = model or os.getenv("ALPHASHOCK_LLM_MODEL", DEFAULT_LLM_MODEL)
        self.provider = self._resolve(provider)

    @staticmethod
    def _resolve(provider: str) -> str:
        if provider != "auto":
            return provider
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        if os.getenv("LITELLM_MODEL"):
            return "litellm"
        return "offline"

    # ------------------------------------------------------------------ public
    def generate(self, event: AnomalyEvent, chunks: list[RetrievedChunk]) -> RCAReport:
        label = self.provider
        raw: LLMReport | None = None
        if self.provider != "offline":
            try:
                raw = self._call_llm(build_user_prompt(event, chunks))
                label = f"{self.provider}:{self.model if self.provider == 'gemini' else os.getenv('LITELLM_MODEL')}"
            except Exception as exc:
                logger.warning("LLM call failed (%s); using offline extractive report", exc)
                label = f"offline-extractive (fallback after {self.provider} error)"
        if raw is None:
            raw = self._extractive(event, chunks)
            if label == "offline":
                label = "offline-extractive"
        return self._finalize(event, chunks, raw, label)

    # ------------------------------------------------------------------ providers
    def _call_llm(self, user_prompt: str) -> LLMReport:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._gemini(user_prompt) if self.provider == "gemini" else self._litellm(user_prompt)
            except Exception as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM failed after retries: {last}")

    def _gemini(self, user_prompt: str) -> LLMReport:
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model, contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.1,
                response_mime_type="application/json", response_schema=LLMReport),
        )
        return LLMReport.model_validate_json(_strip_fences(resp.text or ""))

    def _litellm(self, user_prompt: str) -> LLMReport:
        import litellm

        schema = json.dumps(LLMReport.model_json_schema())
        resp = litellm.completion(
            model=os.environ["LITELLM_MODEL"], temperature=0.1, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"{user_prompt}\n\nReturn ONLY JSON matching this schema:\n{schema}"}],
        )
        return LLMReport.model_validate_json(_strip_fences(resp.choices[0].message.content or ""))

    @staticmethod
    def _extractive(event: AnomalyEvent, chunks: list[RetrievedChunk]) -> LLMReport:
        if not chunks:
            return LLMReport(
                             summary=(f"{event.ticker} moved {event.pct_change:+.2f}% "
                                      f"(return z-score {event.z_score:+.2f}), but no matching "
                                      "documents were found in the local evidence corpus for "
                                      "the surveillance window."),
                             primary_catalyst="Unexplained", catalyst_category=Category.unexplained,
                             confidence=Confidence.low, findings=[], market_impact="",
                             caveats=["This is an evidence-coverage gap, not a confirmed absence of a real-world catalyst."])
        top = chunks[0].doc
        blob = f"{top.title} {top.text} {top.item or ''}".lower()
        category = next((cat for pat, cat in _CATEGORY_RULES if re.search(pat, blob)), Category.other)
        findings = [Finding(claim=f"{c.doc.source} reported: {c.doc.title}", citations=[c.doc.doc_id])
                    for c in chunks[:3]]
        move = "rose" if event.direction == "up" else "fell"
        return LLMReport(
            summary=(f"{event.ticker} {move} {abs(event.pct_change):.1f}% (z={event.z_score:+.1f}, volume "
                     f"{event.volume_spike_ratio:.1f}x normal). The highest-ranked evidence is "
                     f"\"{top.title}\" [{top.doc_id}]."),
            primary_catalyst=top.title, catalyst_category=category,
            confidence=Confidence.medium if len(chunks) >= 2 else Confidence.low, findings=findings,
            market_impact=f"The top-ranked document was published {top.published_at:%Y-%m-%d %H:%M} UTC, "
                          f"within the surveillance window around the move.",
            caveats=["Extractive report (no LLM): categories come from keyword rules, not language understanding."],
        )

    # ------------------------------------------------------------------ guardrails
    @staticmethod
    def _finalize(event: AnomalyEvent, chunks: list[RetrievedChunk], raw: LLMReport, label: str) -> RCAReport:
        allowed = {c.doc.doc_id for c in chunks}
        dropped = 0
        findings: list[Finding] = []
        for f in raw.findings:
            valid = [c for c in dict.fromkeys(f.citations) if c in allowed]
            dropped += len(f.citations) - len(valid)
            if valid:
                findings.append(Finding(claim=f.claim, citations=valid))
            else:
                dropped += 1
        if not findings and chunks and dropped:
            raw = CatalystSynthesizer._extractive(event, chunks)
            findings = raw.findings
            caveat = "The LLM returned unsupported citations; using the cited extractive evidence report."
        else:
            caveat = None
        data = raw.model_dump()
        data["findings"] = [f.model_dump() for f in findings]
        caveats = list(raw.caveats)
        if caveat:
            caveats.append(caveat)
        if not findings:
            data.update(catalyst_category=Category.unexplained, confidence=Confidence.low)
            caveats.append("No cited evidence survived validation; treating the move as unexplained.")
        if dropped:
            caveats.append(f"{dropped} unsupported citation(s)/finding(s) were removed by the citation guardrail.")
        if len({c for f in findings for c in f.citations}) < 2 and data["confidence"] == Confidence.high:
            data["confidence"] = Confidence.medium
        data["caveats"] = caveats
        return RCAReport(
            **data, event_id=event.event_id, ticker=event.ticker, anomaly_timestamp=event.timestamp,
            z_score=event.z_score, pct_change=event.pct_change, volume_spike_ratio=event.volume_spike_ratio,
            evidence_doc_ids=[c.doc.doc_id for c in chunks], generated_by=label, dropped_citations=dropped,
        )


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t)
