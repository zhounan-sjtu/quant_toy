from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .backtest import PeriodResult


def build_data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    clean_data = data.copy()
    clean_data["date"] = pd.to_datetime(clean_data["date"])
    all_dates = sorted(clean_data["date"].dt.date.unique())
    rows: list[dict[str, Any]] = []

    for symbol, group in clean_data.groupby("symbol"):
        symbol_dates = set(group["date"].dt.date)
        first_date = min(symbol_dates)
        last_date = max(symbol_dates)
        expected_dates = [
            date for date in all_dates if first_date <= date <= last_date
        ]
        missing_dates = sorted(set(expected_dates).difference(symbol_dates))
        rows.append(
            {
                "symbol": symbol,
                "row_count": int(len(group)),
                "first_date": str(first_date),
                "last_date": str(last_date),
                "zero_volume_rows": int((group["volume"] <= 0).sum()),
                "duplicate_date_rows_after_cleaning": int(
                    group.duplicated(subset=["date"]).sum()
                ),
                "missing_trading_days_vs_union": int(len(missing_dates)),
                "first_missing_dates": ",".join(
                    str(item) for item in missing_dates[:10]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def write_records(
    records_dir: Path,
    *,
    data: pd.DataFrame,
    period_results: list["PeriodResult"],
    signal_tables: dict[str, pd.DataFrame],
    rebalance_records: dict[str, list[dict[str, Any]]],
    backtest_outputs: dict[str, dict[str, Any]],
    akshare_interface: str,
    akshare_adjust: str,
    signal_rule: str | dict[str, str],
    fill_policy: dict[str, Any],
    lot_size: int,
    slippage: dict[str, Any],
) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    data_quality = build_data_quality_report(data)
    initial_cash = period_results[0].initial_cash if period_results else None
    summary = {
        "akshare_interface": akshare_interface,
        "akshare_adjust": akshare_adjust,
        "signal_rule": signal_rule,
        "fill_policy": fill_policy,
        "initial_cash": initial_cash,
        "lot_size": lot_size,
        "transaction_costs": {
            "commission_rate": 0.0,
            "stamp_tax_rate": 0.0,
            "transfer_fee_rate": 0.0,
            "min_commission": 0.0,
            "slippage": slippage,
        },
        "data_rows": int(len(data)),
        "date_min": str(pd.to_datetime(data["date"]).min().date()),
        "date_max": str(pd.to_datetime(data["date"]).max().date()),
        "symbol_count": int(data["symbol"].nunique()),
        "data_quality_file": "data_quality.csv",
        "period_results": [item.as_dict() for item in period_results],
    }
    (records_dir / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([item.as_dict() for item in period_results]).to_csv(
        records_dir / "period_returns.csv",
        index=False,
    )
    data_quality.to_csv(records_dir / "data_quality.csv", index=False)

    for period_key, table in signal_tables.items():
        table.to_csv(records_dir / f"signals_{period_key}.csv", index=False)

    for period_key, outputs in backtest_outputs.items():
        orders = outputs.get("orders")
        trades = outputs.get("trades")
        equity = outputs.get("equity")
        if isinstance(orders, pd.DataFrame):
            orders.to_csv(records_dir / f"orders_{period_key}.csv", index=False)
        if isinstance(trades, pd.DataFrame):
            trades.to_csv(records_dir / f"trades_{period_key}.csv", index=False)
        if isinstance(equity, pd.Series):
            equity.rename("equity").to_csv(records_dir / f"equity_{period_key}.csv")

    (records_dir / "rebalance_records.json").write_text(
        json.dumps(rebalance_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
