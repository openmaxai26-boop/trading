"""
Fundamental data collector — financial ratios, macro indicators.
Sources: Alpha Vantage, FRED API, Yahoo Finance fundamentals.
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
AV_BASE = "https://www.alphavantage.co/query"

# Key macro series to track
MACRO_SERIES = {
    "VIX": "VIXCLS",            # Volatility index
    "DXY": "DTWEXBGS",          # Dollar index
    "US10Y": "DGS10",           # 10-year Treasury yield
    "US2Y": "DGS2",             # 2-year Treasury yield
    "FEDFUNDS": "FEDFUNDS",     # Fed funds rate
    "CPI": "CPIAUCSL",          # CPI
    "UNEMPLOYMENT": "UNRATE",   # Unemployment rate
    "GDP_GROWTH": "A191RL1Q225SBEA",  # Real GDP growth
    "M2": "M2SL",               # M2 money supply
}


class FundamentalDataCollector:
    """Fetches fundamental and macro economic data."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.fred_key = self.settings.data.fred_api_key
        self.av_key = self.settings.data.alpha_vantage_key

    # ------------------------------------------------------------------ #
    #  Macro data via FRED                                                 #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def fetch_fred_series(self, series_id: str, start_date: str = "2010-01-01") -> pd.Series:
        """Fetch a FRED data series."""
        if not self.fred_key:
            logger.warning("FRED API key not set; skipping macro data")
            return pd.Series(dtype=float, name=series_id)

        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        resp = requests.get(FRED_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float, name=series_id)

        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        logger.debug(f"FRED {series_id}: {len(df)} observations")
        return df["value"].rename(series_id)

    def fetch_all_macro(self, start_date: str = "2010-01-01") -> pd.DataFrame:
        """Fetch all key macro series and combine into a DataFrame."""
        series_list = []
        for name, series_id in MACRO_SERIES.items():
            try:
                s = self.fetch_fred_series(series_id, start_date=start_date)
                if not s.empty:
                    series_list.append(s.rename(name))
            except Exception as e:
                logger.warning(f"Failed to fetch FRED {name} ({series_id}): {e}")

        if not series_list:
            return pd.DataFrame()

        macro_df = pd.concat(series_list, axis=1)
        macro_df = macro_df.resample("D").ffill()
        logger.info(f"Macro data: {macro_df.shape} ({start_date} → today)")
        return macro_df

    def compute_yield_curve_slope(self, macro_df: pd.DataFrame) -> pd.Series:
        """10Y - 2Y yield spread (inversion = recession risk)."""
        if "US10Y" in macro_df.columns and "US2Y" in macro_df.columns:
            return (macro_df["US10Y"] - macro_df["US2Y"]).rename("yield_curve_slope")
        return pd.Series(dtype=float, name="yield_curve_slope")

    # ------------------------------------------------------------------ #
    #  Equity fundamentals via Alpha Vantage                              #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=30))
    def fetch_equity_overview(self, ticker: str) -> dict:
        """Fetch company overview / fundamental ratios."""
        if not self.av_key:
            logger.warning("Alpha Vantage key not set; skipping equity fundamentals")
            return {}

        params = {"function": "OVERVIEW", "symbol": ticker, "apikey": self.av_key}
        resp = requests.get(AV_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Symbol" not in data:
            logger.debug(f"No overview data for {ticker}")
            return {}

        # Extract key ratios
        float_fields = [
            "PERatio", "PEGRatio", "BookValue", "DividendYield",
            "EPS", "RevenuePerShareTTM", "ProfitMargin", "OperatingMarginTTM",
            "ReturnOnAssetsTTM", "ReturnOnEquityTTM", "RevenueTTM",
            "GrossProfitTTM", "DilutedEPSTTM", "QuarterlyEarningsGrowthYOY",
            "QuarterlyRevenueGrowthYOY", "AnalystTargetPrice", "TrailingPE",
            "ForwardPE", "PriceToSalesRatioTTM", "PriceToBookRatio",
            "EVToRevenue", "EVToEBITDA", "Beta", "52WeekHigh", "52WeekLow",
        ]
        result = {"ticker": ticker, "sector": data.get("Sector", ""), "industry": data.get("Industry", "")}
        for f in float_fields:
            try:
                result[f] = float(data.get(f, "None") or "nan")
            except ValueError:
                result[f] = float("nan")
        return result

    def fetch_batch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch fundamentals for multiple tickers."""
        rows = []
        for i, ticker in enumerate(tickers):
            try:
                row = self.fetch_equity_overview(ticker)
                if row:
                    rows.append(row)
            except Exception as e:
                logger.warning(f"Fundamentals failed for {ticker}: {e}")
            if (i + 1) % 5 == 0:
                logger.info(f"Fundamentals progress: {i+1}/{len(tickers)}")

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("ticker")
