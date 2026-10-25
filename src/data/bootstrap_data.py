"""Bootstrap AlphaShock data so the project runs out of the box.

Creates:
  data/raw/prices/{TICKER}.csv     real OHLCV via yfinance (synthetic GBM fallback if offline)
  data/processed/documents.jsonl   15 catalyst events x 2 documents (headline + SEC 8-K / regulatory
                                   item) aligned to real historical volatility dates, plus 100
                                   noise/chatter documents (35 of them placed inside event windows
                                   as hard decoys)
  data/processed/ground_truth.json event_id -> relevant doc ids (for the benchmark)

IMPORTANT: dates and move directions are real historical episodes, but all document *text* is
synthetic, written to resemble the real filings/headlines. It is not verbatim source material.
Optional: --fnspid streams real historical articles from HuggingFace `Zihan1004/FNSPID`.
Optional: --live-news retrieves current article headlines and publisher snippets from Google News RSS.

Usage:
  python -m src.data.bootstrap_data
  python -m src.data.bootstrap_data --offline-prices
  python -m src.data.bootstrap_data --fnspid --fnspid-max-rows 20000
  python -m src.data.bootstrap_data --live-news --live-news-max-per-ticker 20
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from src.config import (
    COMPANY_NAMES, DOCUMENTS_PATH, GROUND_TRUTH_PATH, META_PATH, PRICES_DIR, PROCESSED_DIR, TICKERS,
)
from src.schemas import Document, GroundTruthEvent

CLOSE_HOUR_UTC = 21  # detector stamps daily anomalies at ~US market close


@dataclass(frozen=True)
class CatalystSpec:
    ticker: str
    date: str            # trading day of the move
    move_pct: float      # approximate historical close-to-close move (used for offline price synthesis)
    news_at: str
    news_title: str
    news_text: str
    second_at: str
    second_source: str
    second_item: str | None
    second_title: str
    second_text: str


SPECS: list[CatalystSpec] = [
    # ------------------------------ NVDA ------------------------------
    CatalystSpec(
        "NVDA", "2023-05-25", 24.4,
        "2023-05-24T21:30:00Z",
        "Nvidia forecasts second-quarter revenue far above Wall Street estimates on surging AI chip demand",
        "Nvidia guided quarterly revenue to roughly $11 billion, well above analyst expectations of about $7 billion, as demand for its data center GPUs used in generative AI accelerates. Shares jumped sharply in extended trading.",
        "2023-05-24T20:20:00Z", "SEC 8-K", "Item 2.02",
        "NVIDIA 8-K: Results for First Quarter Fiscal 2024 and outlook for second quarter",
        "Item 2.02 Results of Operations and Financial Condition. On May 24, 2023, NVIDIA Corporation announced financial results for the first quarter ended April 30, 2023. Outlook for the second quarter of fiscal 2024: revenue is expected to be $11.00 billion, plus or minus 2%, led by data center.",
    ),
    CatalystSpec(
        "NVDA", "2024-05-23", 9.3,
        "2024-05-22T21:15:00Z",
        "Nvidia tops estimates, announces 10-for-1 stock split and higher dividend as data center revenue soars",
        "Nvidia reported record quarterly revenue of about $26 billion and said its board approved a ten-for-one forward stock split and a sizeable dividend increase, sending the shares higher after the bell.",
        "2024-05-22T20:25:00Z", "SEC 8-K", "Item 2.02",
        "NVIDIA 8-K: Results for First Quarter Fiscal 2025; stock split and dividend increase",
        "Item 2.02 Results of Operations and Financial Condition. NVIDIA announced record first quarter fiscal 2025 revenue of $26.0 billion. The Board of Directors approved a ten-for-one forward stock split and an increase to the quarterly cash dividend.",
    ),
    CatalystSpec(
        "NVDA", "2024-08-29", -6.4,
        "2024-08-28T21:30:00Z",
        "Nvidia beats on revenue but shares slide as growth deceleration and Blackwell timing worries weigh",
        "Nvidia posted record quarterly revenue of roughly $30 billion and guided to about $32.5 billion, but the size of the beat narrowed and investors questioned the timing of the next-generation Blackwell ramp.",
        "2024-08-28T20:20:00Z", "SEC 8-K", "Item 2.02",
        "NVIDIA 8-K: Results for Second Quarter Fiscal 2025 and third quarter outlook",
        "Item 2.02 Results of Operations and Financial Condition. NVIDIA announced second quarter fiscal 2025 revenue of $30.0 billion. Outlook for the third quarter: revenue is expected to be $32.5 billion, plus or minus 2%.",
    ),
    CatalystSpec(
        "NVDA", "2025-01-27", -17.0,
        "2025-01-27T14:45:00Z",
        "Nvidia plunges as DeepSeek's efficient AI model raises doubts about the scale of AI infrastructure spending",
        "Nvidia shares tumbled in the worst single-day market value loss on record after a Chinese lab released a competitive reasoning model reportedly trained at a fraction of typical costs, sparking fears of reduced demand for high-end AI accelerators.",
        "2025-01-27T13:00:00Z", "Market News Analysis", None,
        "AI hardware sell-off: DeepSeek model release triggers global tech rout",
        "Technology stocks sold off sharply pre-market as investors reassessed capital expenditure plans for AI data centers following the release of the DeepSeek R1 model. Semiconductor suppliers with heavy data center exposure, led by Nvidia, saw the steepest declines.",
    ),
    CatalystSpec(
        "NVDA", "2025-04-16", -6.9,
        "2025-04-16T12:10:00Z",
        "Nvidia says U.S. now requires licenses for H20 chip exports to China, flags multibillion-dollar charge",
        "Nvidia told investors that U.S. authorities will require a license to export its H20 AI chip to China and several other markets, and that it expects roughly $5.5 billion in charges tied to the restriction in its fiscal first quarter.",
        "2025-04-15T21:05:00Z", "SEC 8-K", "Item 8.01",
        "NVIDIA 8-K: Export license requirement for H20 integrated circuits",
        "Item 8.01 Other Events. On April 9, 2025, the Company was informed by the U.S. government that a license is required for exports of the H20 product to China, including Hong Kong and Macau. The Company expects to incur charges of approximately $5.5 billion in the first quarter of fiscal 2026.",
    ),
    # ------------------------------ AAPL ------------------------------
    CatalystSpec(
        "AAPL", "2023-08-04", -4.8,
        "2023-08-03T21:45:00Z",
        "Apple posts third straight quarterly sales decline as iPhone and Mac weakness offsets record services",
        "Apple reported a slight year-over-year revenue decline for its fiscal third quarter and warned of another decline ahead, with iPad and Mac sales falling even as services revenue hit a record.",
        "2023-08-03T20:30:00Z", "SEC 8-K", "Item 2.02",
        "Apple Inc. 8-K: Results for fiscal third quarter ended July 1, 2023",
        "Item 2.02 Results of Operations and Financial Condition. Apple announced financial results for its fiscal 2023 third quarter: revenue of $81.8 billion, down 1 percent year over year. Services set an all-time revenue record.",
    ),
    CatalystSpec(
        "AAPL", "2024-05-03", 6.0,
        "2024-05-02T21:30:00Z",
        "Apple unveils record $110 billion buyback as iPhone sales decline by less than feared",
        "Apple authorized the largest share repurchase program in U.S. corporate history and reported iPhone sales that fell by less than analysts expected, with services growth offsetting weakness in China.",
        "2024-05-02T20:30:00Z", "SEC 8-K", "Item 2.02",
        "Apple Inc. 8-K: Results for fiscal second quarter ended March 30, 2024; capital return",
        "Item 2.02 Results of Operations and Financial Condition. Apple reported fiscal 2024 second quarter revenue of $90.8 billion. The Board authorized an additional $110 billion for share repurchases and raised the quarterly dividend by 4 percent.",
    ),
    CatalystSpec(
        "AAPL", "2024-06-11", 7.3,
        "2024-06-11T12:30:00Z",
        "Wall Street cheers Apple's AI debut; analysts see potential iPhone upgrade cycle",
        "Apple shares climbed as analysts reacted positively to the company's generative AI strategy, including deeper Siri capabilities and an integration with ChatGPT, saying it could spur a multi-year iPhone upgrade cycle.",
        "2024-06-10T17:45:00Z", "News Headline", None,
        "Apple unveils 'Apple Intelligence' at WWDC, brings ChatGPT into Siri",
        "At its Worldwide Developers Conference keynote, Apple introduced Apple Intelligence, a personal AI system for iPhone, iPad and Mac, and announced an integration with OpenAI's ChatGPT for Siri requests.",
    ),
    CatalystSpec(
        "AAPL", "2025-04-03", -9.2,
        "2025-04-03T13:40:00Z",
        "Apple sheds hundreds of billions in market value as sweeping tariffs hit its Asia-heavy supply chain",
        "Apple shares fell sharply in pre-market trading as investors weighed the impact of newly announced reciprocal tariffs on a supply chain concentrated in China, India and Vietnam.",
        "2025-04-02T20:30:00Z", "Regulatory News", None,
        "White House announces sweeping reciprocal tariffs on imports from major trading partners",
        "The administration announced broad reciprocal tariffs, including steep duties on goods from China, Vietnam and India, where consumer electronics makers source much of their production. Equity futures fell sharply after the announcement.",
    ),
    CatalystSpec(
        "AAPL", "2025-04-09", 15.3,
        "2025-04-09T19:30:00Z",
        "Apple surges in biggest one-day gain in decades after tariff pause",
        "Apple stock soared as the administration paused most reciprocal tariffs for 90 days, easing fears about the company's exposure to Asian manufacturing, although duties on China remained elevated.",
        "2025-04-09T17:20:00Z", "Regulatory News", None,
        "U.S. pauses reciprocal tariffs for 90 days for most countries, raises duties on China",
        "The administration announced a 90-day pause on elevated reciprocal tariffs for most trading partners while raising duties on Chinese imports. Equities rallied broadly on the announcement.",
    ),
    # ------------------------------ TSLA ------------------------------
    CatalystSpec(
        "TSLA", "2024-04-24", 12.1,
        "2024-04-23T21:40:00Z",
        "Tesla says it will launch cheaper models early, shares jump after-hours despite earnings miss",
        "Tesla missed first-quarter profit expectations but said it would accelerate the launch of new, more affordable vehicles, and reiterated its autonomy and robotaxi ambitions, lifting the stock in extended trading.",
        "2024-04-23T20:25:00Z", "SEC 8-K", "Item 2.02",
        "Tesla, Inc. 8-K: First quarter 2024 financial results and update letter",
        "Item 2.02 Results of Operations and Financial Condition. Tesla furnished its first quarter 2024 update. The Company stated it is accelerating the launch of new vehicles, including more affordable models, using existing manufacturing lines.",
    ),
    CatalystSpec(
        "TSLA", "2024-07-24", -12.3,
        "2024-07-23T21:50:00Z",
        "Tesla profit beats but robotaxi delay and thin auto margins disappoint investors",
        "Tesla reported second-quarter results with automotive gross margin pressure and said its robotaxi unveiling would be pushed back to October, prompting a sell-off after earnings.",
        "2024-07-23T20:20:00Z", "SEC 8-K", "Item 2.02",
        "Tesla, Inc. 8-K: Second quarter 2024 financial results and update letter",
        "Item 2.02 Results of Operations and Financial Condition. Tesla furnished its second quarter 2024 update. Automotive gross margin excluding regulatory credits declined; the Company noted the robotaxi product unveil has been rescheduled to October.",
    ),
    CatalystSpec(
        "TSLA", "2024-10-24", 22.1,
        "2024-10-23T21:35:00Z",
        "Tesla jumps after-hours as margin improvement and 2025 growth outlook revive the bull case",
        "Tesla beat third-quarter profit expectations and said it expects vehicle growth of 20% to 30% next year, helped by new products and cost reductions, driving a surge in the shares.",
        "2024-10-23T20:25:00Z", "SEC 8-K", "Item 2.02",
        "Tesla, Inc. 8-K: Third quarter 2024 financial results and update letter",
        "Item 2.02 Results of Operations and Financial Condition. Tesla furnished its third quarter 2024 update reporting improved gross margin. The Company indicated a return to vehicle volume growth in 2025.",
    ),
    CatalystSpec(
        "TSLA", "2024-11-06", 14.8,
        "2024-11-06T14:30:00Z",
        "Tesla soars as election result lifts outlook for autonomy rules and EV policy",
        "Tesla shares leapt as investors bet that the CEO's close ties to the incoming administration would ease the path for autonomous driving regulation, even as consumer EV tax credits face uncertainty.",
        "2024-11-06T07:30:00Z", "Regulatory News", None,
        "U.S. election outcome shifts expectations for EV subsidies and federal autonomous-vehicle framework",
        "Markets reacted to the election result by repricing sectors sensitive to regulation. Analysts said a lighter-touch federal framework for self-driving vehicles could benefit Tesla, while EV purchase incentives may be reduced.",
    ),
    CatalystSpec(
        "TSLA", "2025-03-10", -15.4,
        "2025-03-10T15:10:00Z",
        "Tesla stock suffers worst day in years as brand backlash and tariff fears compound",
        "Tesla shares sank in their steepest one-day drop since 2020 amid weakening sales in Europe and China, brand-related protests and a broad market sell-off on recession and tariff concerns.",
        "2025-03-10T13:45:00Z", "Market News Analysis", None,
        "Wall Street sell-off deepens on recession fears; Tesla hit by weak demand reports",
        "U.S. equity futures extended losses on growth worries and trade-policy uncertainty. Tesla was among the weakest large caps pre-market following reports of declining registrations in key overseas markets.",
    ),
]

# (source, item, title template, body template). First 16 -> noise. HARD_IDX share vocabulary
# (earnings / guidance / 8-K / Item numbers) with real catalysts but carry no causal information.
NOISE_TEMPLATES: list[tuple[str, str | None, str, str]] = [
    ("News Headline", None, "{co} shares edge higher as analysts reiterate Buy ratings",
     "Several brokerages reiterated bullish ratings on {co} ({t}) this week, citing long-term growth, though none changed price targets materially. Trading volume stayed near its recent average."),
    ("News Headline", None, "Options traders position for {co} ahead of next earnings report",
     "Implied volatility on {t} options has risen modestly as investors speculate about upcoming quarterly earnings and guidance. No company announcement has been made."),
    ("News Headline", None, "Weekly market wrap: semiconductor and megacap tech stocks trade mixed",
     "Technology shares moved in a narrow range as investors awaited inflation data. {co} ({t}) was little changed."),
    ("SEC Form 4", "Form 4", "{t} director sells shares under pre-arranged 10b5-1 trading plan",
     "A Form 4 filing shows a director of {co} sold a small number of shares under a pre-arranged trading plan, a minor fraction of the director's holdings."),
    ("SEC 8-K", "Item 5.07", "{co} files 8-K reporting annual shareholder meeting voting results",
     "Item 5.07 Submission of Matters to a Vote of Security Holders. {co} held its annual meeting of shareholders; all director nominees were elected and the independent auditor was ratified."),
    ("SEC 8-K", "Item 5.02", "{co} 8-K: routine compensation committee arrangements for officers",
     "Item 5.02 Departure of Directors or Certain Officers. The compensation committee approved customary annual equity awards for executive officers of {co}."),
    ("News Headline", None, "Opinion: Is {co} stock too expensive after a strong run?",
     "A columnist argues that {co} ({t}) trades at a premium multiple and that investors should weigh valuation against growth. The piece contains no new company information."),
    ("News Headline", None, "{co} unveils minor product update; analysts see limited financial impact",
     "{co} announced an incremental product refresh. Analysts said the update was largely expected and would not change earnings estimates for {t}."),
    ("News Headline", None, "Retail investors flock to {t}: social media mentions climb",
     "Mentions of {t} on investor forums rose this week, though professional investors noted little change in fundamentals or guidance."),
    ("News Headline", None, "Analyst raises {co} price target modestly ahead of quarter-end",
     "An analyst nudged the price target on {co} ({t}) higher by a few percent, citing sector momentum. The firm left its earnings forecast unchanged."),
    ("SEC 8-K", "Item 7.01", "{co} furnishes investor presentation for upcoming industry conference",
     "Item 7.01 Regulation FD Disclosure. {co} furnished slides to be used at an industry conference; the presentation reiterates previously disclosed information."),
    ("News Headline", None, "Volatility index ticks up; {co} among most actively traded stocks",
     "Broad market volatility rose slightly as bond yields moved. {co} ({t}) ranked among the most actively traded names, with moves in line with the broader index."),
    ("News Headline", None, "{co} supplier chatter: industry newsletter notes steady component orders",
     "An industry newsletter cited unnamed sources describing steady component orders for {co}. The report was unconfirmed and included no figures."),
    ("SEC Form 4", "Form 4", "Form 4: {co} executive exercises stock options and retains shares",
     "A Form 4 filing shows a {co} executive exercised options and held the resulting shares. No open-market sale was reported."),
    ("News Headline", None, "Fund manager trims {co} stake in quarterly 13F filing",
     "A 13F filing showed a mid-sized fund reduced its position in {co} ({t}) last quarter. The change was small relative to the company's float."),
    ("News Headline", None, "{co} earnings preview: what Wall Street expects this quarter",
     "Analysts expect {co} to report quarterly revenue and earnings roughly in line with prior guidance. The report is scheduled for later this month."),
]
HARD_IDX = [1, 4, 5, 9, 10, 15]


def event_timestamp(date: str) -> datetime:
    return datetime.strptime(date, "%Y-%m-%d").replace(hour=CLOSE_HOUR_UTC, tzinfo=timezone.utc)


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def event_id_for(ticker: str, date: str) -> str:
    return f"{ticker}-{date.replace('-', '')}"


def build_catalyst_documents() -> tuple[list[Document], list[GroundTruthEvent]]:
    docs: list[Document] = []
    truth: list[GroundTruthEvent] = []
    for s in SPECS:
        eid = event_id_for(s.ticker, s.date)
        news = Document(
            doc_id=f"{eid}-news", ticker=s.ticker, published_at=_parse_iso(s.news_at),
            source="News Headline", title=s.news_title, text=s.news_text,
            is_catalyst=True, event_id=eid,
        )
        second = Document(
            doc_id=f"{eid}-{'sec' if s.second_source == 'SEC 8-K' else 'ctx'}",
            ticker=s.ticker, published_at=_parse_iso(s.second_at), source=s.second_source,
            item=s.second_item, title=s.second_title, text=s.second_text,
            is_catalyst=True, event_id=eid,
        )
        docs += [news, second]
        truth.append(GroundTruthEvent(
            event_id=eid, ticker=s.ticker, date=s.date, expected_move_pct=s.move_pct,
            relevant_doc_ids=[news.doc_id, second.doc_id],
        ))
    return docs, truth


def build_noise_documents(
    rng: random.Random, start: str, end: str, n_total: int = 100, n_decoys: int = 35,
) -> list[Document]:
    """n_decoys land inside the [-48h, +12h] window of a real event (same ticker) so time filtering
    alone cannot separate them from catalysts; the rest are scattered over random business days."""
    days = pd.bdate_range(start, end)
    docs: list[Document] = []
    for i in range(n_total):
        if i < n_decoys:
            spec = SPECS[i % len(SPECS)]
            ticker = spec.ticker
            published = event_timestamp(spec.date) + timedelta(hours=rng.uniform(-46, 10))
            tpl = NOISE_TEMPLATES[rng.choice(HARD_IDX)]
        else:
            ticker = rng.choice(TICKERS)
            day = days[rng.randrange(len(days))]
            published = datetime(day.year, day.month, day.day, rng.randint(13, 21),
                                 rng.randint(0, 59), tzinfo=timezone.utc)
            tpl = NOISE_TEMPLATES[rng.randrange(len(NOISE_TEMPLATES))]
        source, item, title_t, body_t = tpl
        co = COMPANY_NAMES[ticker]
        docs.append(Document(
            doc_id=f"noise-{i:03d}", ticker=ticker, published_at=published, source=source, item=item,
            title=title_t.format(co=co, t=ticker), text=body_t.format(co=co, t=ticker),
        ))
    return docs


def load_fnspid(
    tickers: tuple[str, ...], max_rows: int = 20000, start: str | None = None, end: str | None = None,
    data_file: str = "Stock_news/nasdaq_exteral_data.csv", max_scan: int = 2_000_000,
) -> list[Document]:
    """Stream real articles from HuggingFace `Zihan1004/FNSPID` (no 20GB download).

    Expected columns: Date, Article_title, Stock_symbol, Article, Lsa_summary. FNSPID ends in 2023,
    and its articles are NOT labelled against our synthetic events, so treat them as extra chatter.
    """
    from datasets import load_dataset

    ds = load_dataset("Zihan1004/FNSPID", data_files=data_file, split="train", streaming=True)
    wanted = set(tickers)
    lo = pd.Timestamp(start, tz="UTC") if start else None
    hi = pd.Timestamp(end, tz="UTC") if end else None
    docs: list[Document] = []
    for scanned, row in enumerate(ds):
        if scanned >= max_scan or len(docs) >= max_rows:
            break
        sym = str(row.get("Stock_symbol") or "").upper()
        if sym not in wanted:
            continue
        ts = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
        if pd.isna(ts) or (lo is not None and ts < lo) or (hi is not None and ts > hi):
            continue
        title = str(row.get("Article_title") or "").strip()
        if not title:
            continue
        body = str(row.get("Lsa_summary") or row.get("Article") or "").strip()[:1500] or title
        docs.append(Document(
            doc_id=f"fnspid-{sym}-{len(docs):06d}", ticker=sym, published_at=ts.to_pydatetime(),
            source="FNSPID News", title=title, text=body, synthetic=False,
        ))
    return docs


def load_live_news(
    tickers: tuple[str, ...], max_per_ticker: int = 20, timeout: int = 15,
) -> list[Document]:
    """Retrieve recent real article entries from Google News RSS.

    RSS supplies the publisher's headline, a short description, publication time,
    and a link to the original article. Full article text is not copied, avoiding
    paywall and copyright issues; the link is retained for human follow-up.
    """
    docs: list[Document] = []
    tag_re = re.compile(r"<[^>]+>")
    for ticker in tickers:
        query = quote(f"{ticker} stock")
        feed_url = (
            "https://news.google.com/rss/search?q="
            f"{query}&hl=en-US&gl=US&ceid=US:en"
        )
        request = Request(feed_url, headers={"User-Agent": "AlphaShock/1.0"})
        with urlopen(request, timeout=timeout) as response:
            root = ElementTree.fromstring(response.read())
        items = root.findall("./channel/item")
        for index, item in enumerate(items[:max_per_ticker]):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            description = html.unescape(item.findtext("description") or "")
            description = tag_re.sub(" ", description)
            description = " ".join(description.split())
            published_raw = (item.findtext("pubDate") or "").strip()
            if not title or not link or not published_raw:
                continue
            try:
                published_at = parsedate_to_datetime(published_raw).astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                logging.warning("Skipping live article with invalid date: %s", published_raw)
                continue
            docs.append(Document(
                doc_id=f"live-news-{ticker}-{index:04d}",
                ticker=ticker,
                published_at=published_at,
                source="Google News RSS",
                title=title,
                text=description or title,
                synthetic=False,
                url=link,
            ))
    return docs


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance logs malformed/blocked Yahoo responses before returning an empty
    # frame. Bootstrap handles that case below, so keep the user-facing output
    # to the explicit fallback warning.
    yf_logger = logging.getLogger("yfinance")
    previous_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False,
            threads=False, timeout=10,
        )
    finally:
        yf_logger.setLevel(previous_level)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns=lambda c: str(c).lower())
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.reset_index(drop=True)


_SYNTH_PARAMS = {"NVDA": (150.0, 0.024, 4.0e8), "AAPL": (125.0, 0.013, 7.0e7), "TSLA": (110.0, 0.032, 1.3e8)}


def synthesize_prices(ticker: str, start: str, end: str, shocks: dict[str, float], seed: int) -> pd.DataFrame:
    """Offline fallback: fat-tailed GBM-like OHLCV with the historical shocks injected."""
    rng = np.random.default_rng(seed + sum(map(ord, ticker)))
    dates = pd.bdate_range(start, end)
    p0, vol, base_vol = _SYNTH_PARAMS[ticker]
    rets = rng.standard_t(df=5, size=len(dates)) * vol / np.sqrt(5 / 3) + 0.0004
    vols = base_vol * np.exp(rng.normal(0, 0.25, len(dates)))
    for i, d in enumerate(dates):
        key = d.strftime("%Y-%m-%d")
        if key in shocks:
            rets[i] = shocks[key] / 100.0
            vols[i] *= rng.uniform(2.5, 4.0)
    close = p0 * np.cumprod(1.0 + rets)
    prev = np.concatenate([[p0], close[:-1]])
    open_ = prev * (1 + rng.normal(0, vol / 4, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, len(dates))))
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close,
                         "volume": vols.astype("int64")})


def bootstrap(start: str = "2023-01-01", end: str = "2025-06-30", seed: int = 7, offline_prices: bool = False,
              use_fnspid: bool = False, fnspid_max_rows: int = 20000, n_noise: int = 100,
              use_live_news: bool = False, live_news_max_per_ticker: int = 20,
              real_only: bool = False) -> dict:
    if real_only and not (use_fnspid or use_live_news):
        raise ValueError("--real-only requires --fnspid or --live-news")
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    meta: dict = {"price_source": {}, "start": start, "end": end, "seed": seed}
    for ticker in TICKERS:
        df: pd.DataFrame | None = None
        if not offline_prices:
            try:
                df = fetch_prices(ticker, start, end)
                meta["price_source"][ticker] = "yfinance"
            except Exception as exc:  # network / rate-limit / schema drift
                print(f"[warn] yfinance failed for {ticker}: {exc}; using synthetic prices")
        if df is None:
            shocks = {s.date: s.move_pct for s in SPECS if s.ticker == ticker}
            df = synthesize_prices(ticker, start, end, shocks, seed)
            meta["price_source"][ticker] = "synthetic-gbm"
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(PRICES_DIR / f"{ticker}.csv", index=False)
        print(f"[prices] {ticker}: {len(out)} rows ({meta['price_source'][ticker]})")

    catalysts, truth = build_catalyst_documents()
    rng = random.Random(seed)
    noise = build_noise_documents(rng, start, end, n_total=n_noise, n_decoys=min(35, n_noise))
    if real_only:
        catalysts, noise, truth = [], [], []
    docs = catalysts + noise
    if use_fnspid:
        try:
            extra = load_fnspid(TICKERS, max_rows=fnspid_max_rows, start=start, end=end)
            print(f"[fnspid] loaded {len(extra)} real articles")
            docs += extra
        except Exception as exc:
            print(f"[warn] FNSPID load failed: {exc}")
    if use_live_news:
        try:
            live = load_live_news(TICKERS, max_per_ticker=live_news_max_per_ticker)
            print(f"[live-news] loaded {len(live)} real article entries")
            docs += live
        except (OSError, ElementTree.ParseError, ValueError) as exc:
            print(f"[warn] live news load failed: {exc}")
    docs.sort(key=lambda d: d.published_at)

    with DOCUMENTS_PATH.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(d.model_dump_json() + "\n")
    GROUND_TRUTH_PATH.write_text(json.dumps([t.model_dump() for t in truth], indent=2), encoding="utf-8")
    meta.update(n_documents=len(docs), n_catalyst_docs=len(catalysts), n_noise_docs=len(noise),
                n_events=len(truth))
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[docs] {len(docs)} documents ({len(catalysts)} catalyst, {len(noise)} noise) -> {DOCUMENTS_PATH}")
    print(f"[truth] {len(truth)} events -> {GROUND_TRUTH_PATH}")
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Bootstrap AlphaShock prices, documents and ground truth.")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-06-30")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--offline-prices", action="store_true", help="skip yfinance, synthesize prices")
    p.add_argument("--fnspid", action="store_true", help="also stream real FNSPID articles")
    p.add_argument("--fnspid-max-rows", type=int, default=20000)
    p.add_argument("--n-noise", type=int, default=100)
    p.add_argument("--live-news", action="store_true",
                   help="retrieve current real article headlines/snippets from Google News RSS")
    p.add_argument("--live-news-max-per-ticker", type=int, default=20)
    p.add_argument("--real-only", action="store_true",
                   help="exclude the synthetic benchmark corpus; use with --fnspid or --live-news")
    a = p.parse_args(argv)
    bootstrap(
        a.start, a.end, a.seed, a.offline_prices, a.fnspid, a.fnspid_max_rows, a.n_noise,
        a.live_news, a.live_news_max_per_ticker, a.real_only,
    )


if __name__ == "__main__":
    main()
