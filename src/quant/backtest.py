from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from akquant import run_backtest

from .strategy import (
    MyStrategy,
    build_market_maps,
    build_signal_table,
    signal_dict,
)


@dataclass(frozen=True)
class PeriodResult:
    strategy_name: str
    period_start: str
    period_end: str
    initial_cash: float
    final_equity: float
    total_return: float
    order_count: int
    filled_order_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "initial_cash": self.initial_cash,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return * 100.0,
            "order_count": self.order_count,
            "filled_order_count": self.filled_order_count,
        }


def run_period_backtest(
    data: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
    symbols: Iterable[str],
    initial_cash: float,
    lot_size: int,
    lookback_days: int,
    holding_count: int,
    fill_policy: dict[str, Any],
    slippage: dict[str, Any],
    signal_mode: str = "weak_reversal",
    strategy_name: str | None = None,
) -> tuple[PeriodResult, Any, MyStrategy, pd.DataFrame]:
    resolved_strategy_name = strategy_name or signal_mode
    clean_data = data.copy()
    clean_data["date"] = pd.to_datetime(clean_data["date"])
    warmup_start = pd.Timestamp(period_start) - pd.Timedelta(days=90)
    period_end_timestamp = pd.Timestamp(period_end)
    backtest_data = clean_data[
        (clean_data["date"] >= warmup_start)
        & (clean_data["date"] <= period_end_timestamp)
    ].reset_index(drop=True)
    signal_table = build_signal_table(
        backtest_data,
        period_start=period_start,
        period_end=period_end,
        lookback_days=lookback_days,
        holding_count=holding_count,
        signal_mode=signal_mode,
    )
    close_by_date, tradable_by_date = build_market_maps(backtest_data)
    strategy = MyStrategy(
        selected_by_date=signal_dict(signal_table),
        close_by_date=close_by_date,
        tradable_by_date=tradable_by_date,
        period_start=period_start,
        period_end=period_end,
        lot_size=lot_size,
    )
    result = run_backtest(
        backtest_data,
        strategy=strategy,
        symbols=list(symbols),
        initial_cash=initial_cash,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        slippage=slippage,
        volume_limit_pct=1.0,
        timezone="Asia/Shanghai",
        t_plus_one=True,
        lot_size=lot_size,
        fill_policy=fill_policy,
        show_progress=False,
    )

    equity = result.equity_curve.dropna()
    period_end_date = pd.Timestamp(period_end).date()
    equity_dates = pd.Index(pd.to_datetime(equity.index).date)
    period_equity = equity[equity_dates <= period_end_date]
    if period_equity.empty:
        raise RuntimeError(f"No equity point found on or before {period_end}")

    orders = result.orders_df
    filled_order_count = 0
    if not orders.empty and "status" in orders:
        filled_order_count = int(
            (orders["status"].astype(str).str.lower() == "filled").sum()
        )

    final_equity = float(period_equity.iloc[-1])
    period_result = PeriodResult(
        strategy_name=resolved_strategy_name,
        period_start=period_start,
        period_end=period_end,
        initial_cash=initial_cash,
        final_equity=final_equity,
        total_return=final_equity / initial_cash - 1.0,
        order_count=int(len(orders)),
        filled_order_count=filled_order_count,
    )
    return period_result, result, strategy, signal_table
