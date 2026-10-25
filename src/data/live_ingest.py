"""Fetch current news and SEC filing metadata for an anomaly window."""
from __future__ import annotations

import html
import logging
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from src.schemas import Document

logger = logging.getLogger(__name__)
_TAG_RE = re.compile(r"<[^>]+>")
_CIK = {"NVDA": "0001045810", "AAPL": "0000320193", "TSLA": "0001318605"}


def _get(url: str, *, accept: str = "application/json", timeout: int = 15) -> bytes:
    user_agent = os.getenv("SEC_USER_AGENT", "AlphaShock research contact@example.com")
    request = Request(url, headers={"User-Agent": user_agent, "Accept": accept})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_google_news(
    ticker: str, start: datetime, end: datetime, *, max_items: int = 20, timeout: int = 15
) -> list[Document]:
    """Fetch headlines/snippets only; the original publisher URL is retained."""
    query = quote(f"{ticker} stock after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    root = ElementTree.fromstring(_get(url, accept="application/xml", timeout=timeout))
    documents: list[Document] = []
    for index, item in enumerate(root.findall("./channel/item")[:max_items]):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        raw_date = (item.findtext("pubDate") or "").strip()
        if not title or not link or not raw_date:
            continue
        try:
            published = parsedate_to_datetime(raw_date).astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            continue
        if not start <= published <= end:
            continue
        description = html.unescape(item.findtext("description") or "")
        text = " ".join(_TAG_RE.sub(" ", description).split()) or title
        documents.append(Document(
            doc_id=f"live-{ticker}-news-{published:%Y%m%d%H%M}-{index}",
            ticker=ticker, published_at=published, source="Google News RSS",
            title=title, text=text[:3000], synthetic=False, url=link,
        ))
    return documents


def fetch_sec_8k(
    ticker: str, start: datetime, end: datetime, *, max_items: int = 20, timeout: int = 15
) -> list[Document]:
    """Fetch recent SEC 8-K submission metadata and filing links."""
    cik = _CIK.get(ticker)
    if cik is None:
        raise ValueError(f"No SEC CIK configured for ticker {ticker}")
    payload = __import__("json").loads(
        _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    )
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    documents: list[Document] = []
    for index, form in enumerate(forms):
        if form != "8-K":
            continue
        date_text = recent["filingDate"][index]
        published = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if not start.date() <= published.date() <= end.date():
            continue
        accession = recent["accessionNumber"][index].replace("-", "")
        primary = recent["primaryDocument"][index]
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary}"
        items = recent.get("items", [""] * len(forms))[index] or "8-K"
        description = (
            f"SEC Form 8-K filed by {ticker} on {date_text}. Reported items: {items}. "
            "Open the filing link for the authoritative filing contents."
        )
        documents.append(Document(
            doc_id=f"sec-{ticker}-{accession}", ticker=ticker, published_at=published,
            source="SEC 8-K", title=f"{ticker} SEC Form 8-K ({date_text})",
            text=description, item=items, synthetic=False, url=filing_url,
        ))
        if len(documents) >= max_items:
            break
    return documents


def fetch_fresh_documents(
    ticker: str, start: datetime, end: datetime, *, max_items: int = 20
) -> tuple[list[Document], list[str]]:
    """Fetch both providers, returning documents and explicit provider errors."""
    documents: list[Document] = []
    errors: list[str] = []
    for name, fetcher in (
        ("Google News", fetch_google_news),
        ("SEC", fetch_sec_8k),
    ):
        try:
            documents.extend(fetcher(ticker, start, end, max_items=max_items))
        except (OSError, ElementTree.ParseError, ValueError, KeyError, IndexError) as exc:
            message = f"{name} fetch failed: {type(exc).__name__}: {exc}"
            logger.warning(message)
            errors.append(message)
    unique = {document.doc_id: document for document in documents}
    return sorted(unique.values(), key=lambda document: document.published_at), errors
