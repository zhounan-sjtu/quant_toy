from __future__ import annotations

import math
from typing import Any

import pandas as pd
from akquant import Strategy


def build_signal_table(
    data: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
    lookback_days: int,
    holding_count: int,
    signal_mode: str = "weak_reversal",
) -> pd.DataFrame:
    if signal_mode == "weak_reversal":
        positive_only = False
    elif signal_mode in {"tf_reversal", "positive_low_return"}:
        positive_only = True
    else:
        raise ValueError(f"Unsupported signal_mode: {signal_mode!r}")

    signal_table_columns = [
        "signal_mode",
        "execution_date",
        "signal_date",
        "base_date",
        "rank",
        "symbol",
        "lookback_return",
        "execution_close",
    ]

    clean_data = data.copy()
    clean_data["date"] = pd.to_datetime(clean_data["date"])
    clean_data = clean_data.sort_values(["date", "symbol"])

    close = clean_data.pivot(
        index="date",
        columns="symbol",
        values="close",
    ).sort_index()
    volume = clean_data.pivot(
        index="date",
        columns="symbol",
        values="volume",
    ).sort_index()
    tradable = (close > 0) & (volume > 0)

    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    rows: list[dict[str, Any]] = []
    all_dates = close.index

    # generate signals for each execution date
    for execution_index, execution_date in enumerate(all_dates):
        if execution_date < start or execution_date > end:
            continue

        signal_index = execution_index - 1
        base_index = signal_index - lookback_days
        if base_index < 0:
            continue

        signal_date = all_dates[signal_index]
        base_date = all_dates[base_index]
        returns = close.loc[signal_date] / close.loc[base_date] - 1.0
        candidates = pd.DataFrame(
            {
                "lookback_return": returns,
                "tradable": tradable.loc[execution_date],
                "execution_close": close.loc[execution_date],
            }
        )
        candidates = candidates.dropna(subset=["lookback_return", "execution_close"])
        candidates = candidates[candidates["tradable"]]
        if positive_only:
            candidates = candidates[candidates["lookback_return"] > 0]
        candidates = candidates.sort_values(
            ["lookback_return"],
            kind="mergesort",
        )

        for rank, symbol in enumerate(candidates.head(holding_count).index, start=1):
            rows.append(
                {
                    "signal_mode": signal_mode,
                    "execution_date": execution_date,
                    "signal_date": signal_date,
                    "base_date": base_date,
                    "rank": rank,
                    "symbol": symbol,
                    "lookback_return": float(candidates.loc[symbol, "lookback_return"]),
                    "execution_close": float(candidates.loc[symbol, "execution_close"]),
                }
            )

    return pd.DataFrame(rows, columns=signal_table_columns)


def build_market_maps(
    data: pd.DataFrame,
) -> tuple[dict[Any, dict[str, float]], dict[Any, set[str]]]:
    close_by_date: dict[Any, dict[str, float]] = {}
    tradable_by_date: dict[Any, set[str]] = {}
    clean_data = data.copy()
    clean_data["date"] = pd.to_datetime(clean_data["date"])

    for date, group in clean_data.groupby(clean_data["date"].dt.date):
        close_by_date[date] = {
            str(row.symbol): float(row.close)
            for row in group.itertuples()
            if float(row.close) > 0
        }
        tradable_by_date[date] = {
            str(row.symbol)
            for row in group.itertuples()
            if float(row.close) > 0 and float(row.volume) > 0
        }

    return close_by_date, tradable_by_date


def signal_dict(signal_table: pd.DataFrame) -> dict[Any, list[str]]:
    if signal_table.empty:
        return {}

    signals: dict[Any, list[str]] = {}
    for execution_date, group in signal_table.groupby("execution_date"):
        sorted_group = group.sort_values("rank")
        signals[pd.Timestamp(execution_date).date()] = list(
            sorted_group["symbol"].astype(str)
        )

    return signals


class MyStrategy(Strategy):
    def __init__(
        self,
        *,
        selected_by_date: dict[Any, list[str]],
        close_by_date: dict[Any, dict[str, float]],
        tradable_by_date: dict[Any, set[str]],
        period_start: str,
        period_end: str,
        lot_size: int,
    ) -> None:
        super().__init__()
        self.selected_by_date = selected_by_date
        self.close_by_date = close_by_date
        self.tradable_by_date = tradable_by_date
        self.period_start = pd.Timestamp(period_start).date()
        self.period_end = pd.Timestamp(period_end).date()
        self.rebalance_records: list[dict[str, Any]] = []
        self._lot_size = int(lot_size)

    def on_daily_rebalance_after_bar(self, trading_date: Any, timestamp: int) -> None:
        del timestamp

        current_date = pd.Timestamp(trading_date).date()
        if current_date < self.period_start or current_date > self.period_end:
            return

        prices = self.close_by_date.get(current_date, {})
        tradable = self.tradable_by_date.get(current_date, set())
        selected = [
            symbol
            for symbol in self.selected_by_date.get(current_date, [])
            if symbol in tradable
        ]
        positions = {
            symbol: float(quantity)
            for symbol, quantity in self.get_positions().items()
            if float(quantity) > 0
        }

        sell_orders: list[Any] = []
        skipped_sells: list[str] = []
        sell_proceeds = 0.0
        for symbol, quantity in sorted(positions.items()):
            price = prices.get(symbol)
            if symbol not in tradable or price is None or price <= 0:
                skipped_sells.append(symbol)
                continue

            order_id = self.sell(
                symbol,
                quantity=quantity,
                tag=f"sell:{current_date}",
            )
            if order_id:
                sell_orders.append(order_id)
                sell_proceeds += quantity * float(price)

        cash_after_sells = self.get_cash() + sell_proceeds
        buy_orders: list[Any] = []
        target_value = cash_after_sells / len(selected) if selected else 0.0
        for symbol in selected:
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            lot_count = math.floor(target_value / float(price) / self._lot_size)
            quantity = lot_count * self._lot_size
            if quantity <= 0:
                continue

            order_id = self.buy(
                symbol,
                quantity=quantity,
                tag=f"buy:{current_date}",
            )
            if order_id:
                buy_orders.append(order_id)

        self.rebalance_records.append(
            {
                "date": str(current_date),
                "selected": selected,
                "sold": sorted(set(positions).difference(skipped_sells)),
                "skipped_sells": skipped_sells,
                "cash_before": float(self.get_cash()),
                "planned_sell_proceeds": float(sell_proceeds),
                "cash_after_planned_sells": float(cash_after_sells),
                "target_value_per_position": float(target_value),
                "buy_order_count": len(buy_orders),
                "sell_order_count": len(sell_orders),
            }
        )
