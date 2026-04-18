from __future__ import annotations

import argparse
import logging

from config import ensure_directories, raw_data_path, resolve_symbols
from twelvedata_client import TwelveDataClient


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Twelve Data uzerinden ham OHLCV verisi cek.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklari cek.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_raw_dataset(symbol: str, client: TwelveDataClient) -> None:
    frame = client.fetch_time_series(symbol)
    output_path = raw_data_path(symbol)
    frame.to_parquet(output_path, index=False)

    LOGGER.info(
        "Ham veri kaydedildi | symbol=%s | rows=%s | cols=%s | start=%s | end=%s | path=%s",
        symbol,
        len(frame),
        list(frame.columns),
        frame.iloc[0]["datetime"],
        frame.iloc[-1]["datetime"],
        output_path,
    )


def main() -> None:
    configure_logging()
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)
    client = TwelveDataClient()

    failures: list[str] = []
    for symbol in symbols:
        try:
            build_raw_dataset(symbol=symbol, client=client)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Ham veri olusturulamadi | symbol=%s | error=%s", symbol, exc)
            failures.append(symbol)

    if failures:
        raise RuntimeError(f"Ham veri asamasi bazi semboller icin basarisiz oldu: {failures}")


if __name__ == "__main__":
    main()
