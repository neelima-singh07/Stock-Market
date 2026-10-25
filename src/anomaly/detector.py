"""Prophet + rolling Z-score anomaly detector for returns and volume.

Price channel
  * Prophet is fit on daily log returns (99% interval). Sigma is recovered from the interval width,
    so the flagging band is  yhat +/- z_threshold * sigma  (default 3 sigma; Prophet's native 99%
    interval is ~2.58 sigma, we widen it to 3 sigma so the band equals the flag rule).
  * Two-pass fit: pass 1 finds outliers, pass 2 refits with them masked so shocks don't inflate sigma.
  * Rows Prophet can't cover (too little history / import or fit failure) fall back to a rolling
    60-day Z-score computed with a 1-day lag (no look-ahead).
Volume channel
  * volume_spike_ratio = volume / rolling 20-day median (lagged); flagged when the log-volume
    rolling Z >= vol_z_threshold AND ratio >= min_volume_ratio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import norm

logger = logging.getLogger(__name__)


class AnomalyEvent(BaseModel):
    event_id: str
    ticker: str
    timestamp: datetime                       # UTC, daily bars stamped at ~market close
    z_score: float                            # signed return Z-score
    pct_change: float                         # close-to-close % change
    volume_spike_ratio: float                 # volume / 20d median volume
    direction: Literal["up", "down"]
    trigger: Literal["price", "volume", "price+volume"]
    method: Literal["prophet", "rolling_z"]
    close: float
    volume: float
    volume_z: float = 0.0
    price_band_lower: float | None = None
    price_band_upper: float | None = None


class DetectorConfig(BaseModel):
    interval_width: float = Field(0.99, gt=0.5, lt=1.0)
    z_threshold: float = 3.0
    vol_z_threshold: float = 3.0
    min_volume_ratio: float = 2.0
    rolling_window: int = 60
    volume_window: int = 20
    min_prophet_rows: int = 120
    use_prophet: bool = True
    refit_masked: bool = True
    close_hour_utc: int = 21


@dataclass
class DetectionResult:
    ticker: str
    frame: pd.DataFrame
    events: list[AnomalyEvent]


def _num(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if (np.isnan(v) or np.isinf(v)) else v


class AnomalyDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()

    # ------------------------------------------------------------------ public
    def analyze(self, prices: pd.DataFrame, ticker: str) -> DetectionResult:
        cfg = self.cfg
        df = prices.sort_values("date").reset_index(drop=True).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["prev_close"] = df["close"].shift(1)
        df["log_ret"] = np.log(df["close"] / df["prev_close"])
        df["pct_change"] = (df["close"] / df["prev_close"] - 1.0) * 100.0

        # rolling stats, lagged by one day to avoid look-ahead
        lag = df["log_ret"].shift(1)
        w = cfg.rolling_window
        roll_mu = lag.rolling(w, min_periods=w // 2).mean()
        roll_sd = lag.rolling(w, min_periods=w // 2).std(ddof=1)
        df["z_roll"] = (df["log_ret"] - roll_mu) / roll_sd.replace(0, np.nan)

        # volume channel
        vw = cfg.volume_window
        med = df["volume"].shift(1).rolling(vw, min_periods=vw // 2).median()
        df["volume_spike_ratio"] = df["volume"] / med.replace(0, np.nan)
        lv = np.log(df["volume"].clip(lower=1))
        lv_lag = lv.shift(1)
        vmu = lv_lag.rolling(60, min_periods=20).mean()
        vsd = lv_lag.rolling(60, min_periods=20).std(ddof=1)
        df["volume_z"] = (lv - vmu) / vsd.replace(0, np.nan)

        # expected return + sigma: Prophet where possible, rolling otherwise
        df["exp_ret"] = np.nan
        df["sigma"] = np.nan
        df["method"] = "rolling_z"
        valid = df.index[df["log_ret"].notna()]
        if cfg.use_prophet and len(valid) >= cfg.min_prophet_rows:
            fc = self._prophet_expected(df.loc[valid, "date"], df.loc[valid, "log_ret"])
            if fc is not None:
                crit = norm.ppf(0.5 + cfg.interval_width / 2.0)
                df.loc[valid, "exp_ret"] = fc["yhat"].to_numpy()
                df.loc[valid, "sigma"] = ((fc["yhat_upper"] - fc["yhat_lower"]) / (2 * crit)).to_numpy()
                df.loc[valid, "method"] = "prophet"
        fb = df["sigma"].isna() | (df["sigma"] <= 0)
        df.loc[fb, "exp_ret"] = roll_mu[fb]
        df.loc[fb, "sigma"] = roll_sd[fb]
        df.loc[fb, "method"] = "rolling_z"
        df["sigma"] = df["sigma"].replace(0, np.nan)

        df["z_score"] = (df["log_ret"] - df["exp_ret"]) / df["sigma"]
        thr = cfg.z_threshold
        df["ret_lower"] = df["exp_ret"] - thr * df["sigma"]
        df["ret_upper"] = df["exp_ret"] + thr * df["sigma"]
        df["price_expected"] = df["prev_close"] * np.exp(df["exp_ret"])
        df["price_lower"] = df["prev_close"] * np.exp(df["ret_lower"])
        df["price_upper"] = df["prev_close"] * np.exp(df["ret_upper"])

        df["price_flag"] = df["z_score"].abs() >= thr
        df["volume_flag"] = (df["volume_z"] >= cfg.vol_z_threshold) & (df["volume_spike_ratio"] >= cfg.min_volume_ratio)
        df["is_anomaly"] = df["price_flag"] | df["volume_flag"]

        events = [self.event_from_row(ticker, row) for _, row in df[df["is_anomaly"]].iterrows()]
        logger.info("%s: %d anomalies (%s)", ticker, len(events), df["method"].iloc[-1])
        return DetectionResult(ticker=ticker, frame=df, events=events)

    def event_from_row(self, ticker: str, row: pd.Series) -> AnomalyEvent:
        """Build an AnomalyEvent from an analyzed frame row (flagged or not; used by the benchmark)."""
        d = pd.Timestamp(row["date"])
        ts = datetime(d.year, d.month, d.day, self.cfg.close_hour_utc, tzinfo=timezone.utc)
        pct = _num(row["pct_change"])
        pf, vf = bool(row.get("price_flag", False)), bool(row.get("volume_flag", False))
        trigger = "price+volume" if (pf and vf) else ("volume" if vf and not pf else "price")
        lo, hi = row.get("price_lower"), row.get("price_upper")
        return AnomalyEvent(
            event_id=f"{ticker}-{d:%Y%m%d}", ticker=ticker, timestamp=ts,
            z_score=_num(row["z_score"]), pct_change=pct,
            volume_spike_ratio=_num(row["volume_spike_ratio"]),
            direction="up" if pct >= 0 else "down", trigger=trigger,
            method="prophet" if row["method"] == "prophet" else "rolling_z",
            close=_num(row["close"]), volume=_num(row["volume"]), volume_z=_num(row["volume_z"]),
            price_band_lower=None if pd.isna(lo) else float(lo),
            price_band_upper=None if pd.isna(hi) else float(hi),
        )

    # ----------------------------------------------------------------- internal
    def _prophet_expected(self, ds: pd.Series, y: pd.Series) -> pd.DataFrame | None:
        try:
            from prophet import Prophet
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("Prophet unavailable (%s); using rolling Z-score fallback", exc)
            return None
        for name in ("cmdstanpy", "prophet"):
            logging.getLogger(name).setLevel(logging.ERROR)

        cfg = self.cfg
        train = pd.DataFrame({"ds": ds.to_numpy(), "y": y.to_numpy()})

        def fit_predict(frame: pd.DataFrame) -> pd.DataFrame:
            model = Prophet(
                interval_width=cfg.interval_width, daily_seasonality=False, weekly_seasonality=True,
                yearly_seasonality=False, changepoint_prior_scale=0.01, uncertainty_samples=1000,
            )
            model.fit(frame)
            np.random.seed(42)  # Prophet's interval simulation uses the global RNG
            return model.predict(train[["ds"]])

        try:
            fc = fit_predict(train)
            if cfg.refit_masked:
                crit = norm.ppf(0.5 + cfg.interval_width / 2.0)
                sigma = (fc["yhat_upper"] - fc["yhat_lower"]) / (2 * crit)
                z0 = (train["y"] - fc["yhat"]) / sigma
                masked = train.copy()
                masked.loc[z0.abs() >= cfg.z_threshold, "y"] = np.nan
                if masked["y"].notna().sum() >= cfg.min_prophet_rows:
                    fc = fit_predict(masked)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if "stan_backend" in str(exc) or "CmdStan" in str(exc):
                detail += (
                    ". Prophet's CmdStan backend is unavailable; install CmdStan "
                    "with `python -c \"from cmdstanpy import install_cmdstan; "
                    "install_cmdstan()\"`."
                )
            logger.warning("Prophet fit failed (%s); using rolling Z-score fallback", detail)
            return None
        out = fc[["yhat", "yhat_lower", "yhat_upper"]].copy()
        out.index = ds.index
        return out


def detect_events(prices: pd.DataFrame, ticker: str, config: DetectorConfig | None = None) -> list[AnomalyEvent]:
    return AnomalyDetector(config).analyze(prices, ticker).events


if __name__ == "__main__":
    from src.config import TICKERS
    from src.data.loaders import load_prices

    logging.basicConfig(level=logging.INFO)
    for t in TICKERS:
        res = AnomalyDetector().analyze(load_prices(t), t)
        print(f"{t}: {len(res.events)} anomalies")
        for e in res.events[:5]:
            print("  ", e.model_dump_json())
