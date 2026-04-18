from __future__ import annotations

import argparse
import logging

import pandas as pd

from config import LABEL_RETURN_THRESHOLD, FEATURE_COLUMNS, ensure_directories, processed_data_path, raw_data_path, resolve_symbols
from features import build_feature_frame


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ham veriden feature eklenmis veri seti olustur.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklari isle.")
    parser.add_argument(
        "--return-threshold",
        type=float,
        default=LABEL_RETURN_THRESHOLD,
        help="Etiketleme icin simetrik getiri esigi. 0 ise klasik binary target kullanilir.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_processed_dataset(symbol: str, return_threshold: float) -> None:
    input_path = raw_data_path(symbol)
    if not input_path.exists():
        raise FileNotFoundError(f"Ham veri bulunamadi: {input_path}")

    raw_frame = pd.read_parquet(input_path)
    processed_frame = build_feature_frame(raw_frame, return_threshold=return_threshold)
    output_path = processed_data_path(symbol)
    processed_frame.to_parquet(output_path, index=False)

    LOGGER.info(
        "Islenmis veri kaydedildi | symbol=%s | rows=%s | feature_count=%s | threshold=%s | features=%s | path=%s",
        symbol,
        len(processed_frame),
        len(FEATURE_COLUMNS),
        return_threshold,
        FEATURE_COLUMNS,
        output_path,
    )


def main() -> None:
    configure_logging()
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)

    failures: list[str] = []
    for symbol in symbols:
        try:
            build_processed_dataset(symbol=symbol, return_threshold=args.return_threshold)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Islenmis veri olusturulamadi | symbol=%s | error=%s", symbol, exc)
            failures.append(symbol)

    if failures:
        raise RuntimeError(f"Islenmis veri asamasi bazi semboller icin basarisiz oldu: {failures}")


if __name__ == "__main__":
    main()
