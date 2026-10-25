# AlphaShock: Market Volatility & SEC Filing Surveillance Engine

Anomaly detection (Prophet + rolling Z) -> time-bounded hybrid retrieval (Qdrant + BM25 + RRF) ->
cross-encoder rerank -> cited root-cause report, plus a retrieval benchmark and a Streamlit dashboard.

```
alphashock/
├── app.py                      # Streamlit + Plotly dashboard
├── requirements.txt
├── .env.example
└── src/
    ├── config.py  schemas.py
    ├── data/       bootstrap_data.py  loaders.py
    ├── anomaly/    detector.py
    ├── rag/        hybrid_retriever.py  synthesizer.py
    └── eval/       benchmark.py
```

## Run on Windows
Open PowerShell in the project root:

```powershell
cd C:\Users\<you>\Downloads\alphashock\alphashock
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data.bootstrap_data --offline-prices
python -m src.eval.benchmark --no-prophet
python -m streamlit run app.py
```

The default bootstrap creates the aligned historical synthetic benchmark corpus.
Do not run `--live-news --real-only` afterward unless you intentionally want a
current-news-only corpus; it replaces the historical documents and cannot be used
with the 2023-2025 ground-truth benchmark.

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

When the dashboard displays an anomaly, the default **Fetch fresh news + SEC
filings** switch fetches Google News RSS headlines and SEC 8-K submission
metadata for that anomaly's `-48h/+12h` window. Those documents are merged
with the local corpus and indexed before retrieval. If a provider is unavailable
or returns no matching documents, the dashboard explicitly reports the error
and uses the local corpus as a fallback.

For SEC requests, optionally set a descriptive contact value in `.env`:

```text
SEC_USER_AGENT=AlphaShock research your-email@example.com
```

The live fetch stores only headlines, snippets, filing metadata, and links; it
does not copy full copyrighted article text.

If Yahoo Finance is unavailable or blocked by your network, run
`python -m src.data.bootstrap_data --offline-prices` to skip the network request
and generate the deterministic synthetic price series directly. Without that
flag, the bootstrap command tries Yahoo Finance first and automatically uses
the same synthetic fallback when no data is returned.
Prophet requires the CmdStan executable in addition to the Python packages. If
Prophet reports that its Stan backend is unavailable, install it once in the
active virtual environment with:

```powershell
python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
```
Set `GEMINI_API_KEY` in `.env` for LLM reports; without a key the app produces an extractive report.

### Real article retrieval

The default corpus is synthetic and remains suitable for the labelled benchmark. To build a
corpus from real article entries instead, run:

```powershell
python -m src.data.bootstrap_data --live-news --real-only
```

This retrieves recent headlines, publisher snippets, timestamps, and links from Google News RSS
for NVDA, AAPL, and TSLA. The RAG index uses the headline and snippet; the original article URL
is stored in each document as `url` for provenance and follow-up. RSS does not copy full
paywalled article bodies. Use `--fnspid --real-only` instead for a larger historical corpus, or
combine both flags. Live retrieval requires network access and can be rerun to refresh
`data/processed/documents.jsonl`.

The dashboard prefers anomaly dates that have matching documents in the local evidence corpus.
If an anomaly has no documents inside its ticker and time window, the report is intentionally
labelled "Unexplained"; this means the local corpus has an evidence-coverage gap, not that no
real-world catalyst existed. The retrieval debug panel shows the exact window and document count.

## Data honesty
Event dates and move directions are real historical episodes; the default catalyst/noise document
text is synthetic. The benchmark number measures retrieval on this constructed corpus (15 events,
100 noise docs, 35 of them decoys inside event windows). Quote it as "on a 15-event synthetic
benchmark", or re-run with labelled real data before presenting it as general performance. Live
RSS/FNSPID documents are real, but they are not labelled against the synthetic ground truth.
