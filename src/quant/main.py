from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import PeriodResult, run_period_backtest
from .data_prepare import load_stock_data
from .result_prepare import write_records
from .strategy import POSITIVE_LOW_RETURN, WEAK_REVERSAL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liquor-stock AKQuant backtest task")
    parser.add_argument("--records-dir", default="artifacts/task_records")
    parser.add_argument("--cache-dir", default="artifacts/task_records/data_cache")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args(argv)


def _fetch_window(periods: tuple[tuple[str, str], ...]) -> tuple[str, str]:
    period_starts = [pd.Timestamp(start) for start, _ in periods]
    period_ends = [pd.Timestamp(end) for _, end in periods]
    fetch_start = (min(period_starts) - pd.Timedelta(days=90)).strftime("%Y%m%d")
    fetch_end = max(period_ends).strftime("%Y%m%d")
    return fetch_start, fetch_end


def _period_key(period_start: str, period_end: str) -> str:
    return f"{period_start}_to_{period_end}".replace("-", "")


def _result_key(strategy_name: str, period_start: str, period_end: str) -> str:
    return f"{strategy_name}_{_period_key(period_start, period_end)}"


def _collect_backtest_output(result: Any) -> dict[str, Any]:
    return {
        "orders": result.orders_df,
        "trades": result.trades_df,
        "equity": result.equity_curve,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    symbols = (
        "000860",
        "002646",
        "600199",
        "603919",
        "000799",
        "603198",
        "600519",
        "600559",
        "002304",
        "000568",
        "600779",
        "600809",
        "000596",
        "603589",
        "600197",
        "603369",
        "000858",
    )
    periods = (
        ("2020-12-31", "2021-12-31"),
        ("2021-12-31", "2022-06-30"),
    )
    akshare_interface = "stock_zh_a_hist"
    akshare_adjust = "qfq"
    lookback_days = 10
    holding_count = 5
    lot_size = 100
    strategies = (
        (
            WEAK_REVERSAL,
            "rank all tradable stocks by T-1 lookback return and buy the lowest returns",
        ),
        (
            POSITIVE_LOW_RETURN,
            "rank only stocks with positive T-1 lookback return and buy the lowest positive returns",
        ),
    )
    fill_policy = {
        "price_basis": "close",
        "bar_offset": 0,
        "temporal": "same_cycle",
    }
    slippage = {"type": "zero", "value": 0.0}

    fetch_start, fetch_end = _fetch_window(periods)
    data = load_stock_data(
        symbols,
        fetch_start,
        fetch_end,
        cache_dir=Path(args.cache_dir),
        adjust=akshare_adjust,
        refresh=bool(args.refresh),
        max_workers=int(args.max_workers),
    )

    period_results: list[PeriodResult] = []
    signal_tables: dict[str, pd.DataFrame] = {}
    rebalance_records: dict[str, list[dict[str, Any]]] = {}
    backtest_outputs: dict[str, dict[str, Any]] = {}

    for strategy_name, _strategy_rule in strategies:
        for period_start, period_end in periods:
            period_result, result, strategy, signals = run_period_backtest(
                data,
                period_start=period_start,
                period_end=period_end,
                symbols=symbols,
                initial_cash=float(args.initial_cash),
                lot_size=lot_size,
                lookback_days=lookback_days,
                holding_count=holding_count,
                fill_policy=fill_policy,
                slippage=slippage,
                signal_mode=strategy_name,
                strategy_name=strategy_name,
            )
            result_key = _result_key(strategy_name, period_start, period_end)
            period_results.append(period_result)
            signal_tables[result_key] = signals
            rebalance_records[result_key] = strategy.rebalance_records
            backtest_outputs[result_key] = _collect_backtest_output(result)

    write_records(
        Path(args.records_dir),
        data=data,
        period_results=period_results,
        signal_tables=signal_tables,
        rebalance_records=rebalance_records,
        backtest_outputs=backtest_outputs,
        akshare_interface=akshare_interface,
        akshare_adjust=akshare_adjust,
        signal_rule={
            strategy_name: (
                f"{strategy_rule}; use {lookback_days}-trading-day returns from "
                "T-1 close; execute at T close"
            )
            for strategy_name, strategy_rule in strategies
        },
        fill_policy=fill_policy,
        lot_size=lot_size,
        slippage=slippage,
    )

    print("strategy_name,period_start,period_end,total_return_pct,final_equity")
    for item in period_results:
        print(
            f"{item.strategy_name},{item.period_start},{item.period_end},"
            f"{item.total_return * 100:.6f},{item.final_equity:.2f}"
        )


if __name__ == "__main__":
    main()
