from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from config import DATETIME_COLUMN, INTERVAL, OUTPUT_SIZE, ROOT_DIR, TIMEZONE


class TwelveDataClient:
    """Minimal Twelve Data istemcisi."""

    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        env_path = ROOT_DIR / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError(
                "TWELVE_DATA_API_KEY bulunamadi. "
                f"Beklenen dosya: {env_path}. "
                "Repo kokunde .env olusturup anahtari ekleyin veya ortam degiskeni olarak tanimlayin."
            )

    def fetch_time_series(
        self,
        symbol: str,
        interval: str = INTERVAL,
        outputsize: int = OUTPUT_SIZE,
        timezone: str = TIMEZONE,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "timezone": timezone,
            "order": "ASC",
            "apikey": self.api_key,
            "format": "JSON",
        }

        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()

        payload = response.json()
        self._raise_for_api_error(payload)

        values = payload.get("values")
        if not values:
            raise ValueError(f"{symbol} icin Twelve Data bos veri dondu.")

        frame = pd.DataFrame(values)
        return self._normalize_frame(frame=frame, symbol=symbol, interval=interval)

    @staticmethod
    def _raise_for_api_error(payload: dict[str, Any]) -> None:
        if payload.get("status") == "error":
            message = payload.get("message", "Bilinmeyen API hatasi")
            raise ValueError(f"Twelve Data API hatasi: {message}")

        if "code" in payload and "message" in payload and "values" not in payload:
            raise ValueError(f"Twelve Data API hatasi: {payload['message']}")

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        normalized = frame.copy()

        if "datetime" not in normalized.columns:
            raise ValueError(f"{symbol} cevabinda datetime kolonu yok.")

        normalized[DATETIME_COLUMN] = pd.to_datetime(normalized["datetime"], utc=True)

        for column in ("open", "high", "low", "close", "volume"):
            if column not in normalized.columns:
                normalized[column] = pd.NA
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized = (
            normalized[[DATETIME_COLUMN, "open", "high", "low", "close", "volume"]]
            .drop_duplicates(subset=[DATETIME_COLUMN])
            .sort_values(DATETIME_COLUMN)
            .reset_index(drop=True)
        )
        normalized["symbol"] = symbol
        normalized["interval"] = interval

        return normalized[[DATETIME_COLUMN, "open", "high", "low", "close", "volume", "symbol", "interval"]]
