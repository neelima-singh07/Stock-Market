# AlphaShock

AlphaShock is a Streamlit dashboard for detecting unusual movements in
NVDA, AAPL, and TSLA prices and finding possible causes in news and SEC
filings.

It includes:

- Anomaly detection with rolling Z-scores and optional Prophet forecasting
- Time-filtered hybrid search (dense embeddings + BM25 + reranking)
- Cited root-cause reports
- A retrieval benchmark

## Quick start (Windows)

From the project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data.bootstrap_data --offline-prices
python -m streamlit run app.py
```

Open the URL shown by Streamlit in your browser.

To run the retrieval benchmark:

```powershell
python -m src.eval.benchmark --no-prophet
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Optional configuration

Create a `.env` file to enable Gemini-generated reports:

```text
GEMINI_API_KEY=your_api_key
```

For SEC requests, you can also set a descriptive contact:

```text
SEC_USER_AGENT=AlphaShock research your-email@example.com
```

Without a Gemini key, AlphaShock uses an extractive report. The dashboard
can fetch fresh news and SEC metadata when network access is available.

## Data note

The default bootstrap creates deterministic price data when
`--offline-prices` is used and a synthetic evidence corpus based on real
historical market events. The benchmark is therefore a 15-event synthetic
benchmark, not a general measure of real-world retrieval performance.

To build a corpus from recent real article entries instead:

```powershell
python -m src.data.bootstrap_data --live-news --real-only
```
