from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd


def validate_stock_symbol(symbol: str) -> str:
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"Invalid six-digit stock symbol: {symbol!r}")
    return symbol


def normalize_ohlcv_frame(
    frame: pd.DataFrame,
    symbol: str,
    *,
    volume_unit: str = "shares",
) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "股票代码": "raw_symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    data = frame.rename(columns=rename_map).copy()
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"{symbol} missing required columns: {missing_columns}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if volume_unit == "hands":
        data["volume"] = data["volume"] * 100.0
    elif volume_unit != "shares":
        raise ValueError("volume_unit must be 'shares' or 'hands'")

    data["symbol"] = symbol
    row_count_before_cleaning = len(data)
    data = data.dropna(subset=required_columns)
    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
        & (data["high"] >= data["low"])
        & (data["high"] >= data[["open", "close"]].max(axis=1))
        & (data["low"] <= data[["open", "close"]].min(axis=1))
    ]
    data = data.sort_values(["symbol", "date"])
    data = data.drop_duplicates(subset=["symbol", "date"], keep="last")
    data = data.reset_index(drop=True)
    data.attrs["dropped_rows"] = row_count_before_cleaning - len(data)
    return data[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _cache_file(
    cache_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> Path:
    return cache_dir / f"stock_zh_a_hist_{symbol}_{start_date}_{end_date}_{adjust}.csv"


def fetch_symbol_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path,
    adjust: str,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(cache_dir, symbol, start_date, end_date, adjust)
    if cache_path.exists() and not refresh:
        cached_data = pd.read_csv(cache_path)
        return normalize_ohlcv_frame(cached_data, symbol, volume_unit="hands")

    ak_symbol = validate_stock_symbol(symbol)
    try:
        raw_data = ak.stock_zh_a_hist(
            symbol=ak_symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            timeout=20,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {symbol} from AKShare") from exc

    if raw_data.empty:
        raise RuntimeError(f"AKShare returned empty data for {symbol}")

    raw_data.to_csv(cache_path, index=False)
    return normalize_ohlcv_frame(raw_data, symbol, volume_unit="hands")


def load_stock_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path,
    adjust: str,
    refresh: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    symbol_list = list(symbols)
    if not symbol_list:
        return pd.DataFrame(
            columns=["date", "symbol", "open", "high", "low", "close", "volume"]
        )

    worker_count = max(1, min(max_workers, len(symbol_list)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                fetch_symbol_daily,
                symbol,
                start_date,
                end_date,
                cache_dir=cache_dir,
                adjust=adjust,
                refresh=refresh,
            )
            for symbol in symbol_list
        ]
        for future in as_completed(futures):
            frames.append(future.result())

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)
