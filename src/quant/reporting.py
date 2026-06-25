from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd

from .data_prepare import normalize_ohlcv_frame


STRATEGY_LABELS = {
    "weak_reversal": "Weak reversal",
    "tf_reversal": "Trend-filtered reversal",
    "positive_low_return": "Trend-filtered reversal",
}
STRATEGY_COLORS = {
    "weak_reversal": "#2f6f95",
    "tf_reversal": "#d05a4e",
    "positive_low_return": "#d05a4e",
}
STRATEGY_ORDER = {
    "weak_reversal": 0,
    "tf_reversal": 1,
    "positive_low_return": 1,
}
PERIOD_LABELS = {
    ("2020-12-31", "2021-12-31"): "2021",
    ("2021-12-31", "2022-06-30"): "2022H1",
}
FUND_BENCHMARKS = {
    "hs300_etf": {
        "label": "HS300 ETF",
        "full_label": "沪深300ETF华泰柏瑞",
        "symbol": "sh510300",
        "display_symbol": "510300",
        "color": "#6b7280",
        "source": "sina_etf_close",
    },
    "csi_baijiu_lof": {
        "label": "Baijiu LOF",
        "full_label": "招商中证白酒指数(LOF)A",
        "symbol": "161725",
        "display_symbol": "161725",
        "color": "#9a6b30",
        "source": "open_fund_nav",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README report figures")
    parser.add_argument("--records-dir", default="artifacts/task_records")
    parser.add_argument("--figures-dir", default="docs/figures")
    return parser.parse_args(argv)


def _period_key(period_start: str, period_end: str) -> str:
    return f"{period_start}_to_{period_end}".replace("-", "")


def _nice_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 13,
    fill: str = "#27313a",
    anchor: str = "start",
    weight: str = "400",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{html.escape(str(text))}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}" />'
    )


def _dashed_line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}" stroke-dasharray="6 5" />'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    *,
    rx: float = 2.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" opacity="{opacity:.2f}" />'
    )


def _polyline(points: Iterable[tuple[float, float]], color: str, width: float = 2.4) -> str:
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{data}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.1f}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def _document(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            *body,
            "</svg>",
            "",
        ]
    )


def _scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if math.isclose(domain_min, domain_max):
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def _load_equity(records_dir: Path, strategy_name: str, period_start: str, period_end: str) -> pd.DataFrame:
    path = records_dir / f"equity_{strategy_name}_{_period_key(period_start, period_end)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity curve: {path}")

    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["timestamp"].astype(str).str.slice(0, 10))
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    mask = (data["date"] >= pd.Timestamp(period_start)) & (data["date"] <= pd.Timestamp(period_end))
    return data.loc[mask, ["date", "equity"]].dropna().drop_duplicates("date")


def _equity_stats(equity: pd.Series) -> tuple[float, float]:
    daily_returns = equity.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
    volatility = float(daily_returns.std(ddof=1))
    sharpe = 0.0
    if volatility > 0:
        sharpe = float(daily_returns.mean() / volatility * math.sqrt(252))

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown_pct = float(-drawdown.min() * 100.0)
    return sharpe, max_drawdown_pct


def build_strategy_metrics(records_dir: Path) -> pd.DataFrame:
    returns_path = records_dir / "period_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing period returns: {returns_path}")

    rows: list[dict[str, object]] = []
    returns = pd.read_csv(returns_path)
    for item in returns.itertuples(index=False):
        equity = _load_equity(
            records_dir,
            str(item.strategy_name),
            str(item.period_start),
            str(item.period_end),
        )
        sharpe, max_drawdown_pct = _equity_stats(equity["equity"])
        rows.append(
            {
                "strategy_order": STRATEGY_ORDER.get(str(item.strategy_name), 99),
                "strategy_name": item.strategy_name,
                "strategy_label": STRATEGY_LABELS.get(str(item.strategy_name), str(item.strategy_name)),
                "period_start": item.period_start,
                "period_end": item.period_end,
                "period_label": PERIOD_LABELS.get((str(item.period_start), str(item.period_end)), str(item.period_end)),
                "initial_cash": float(item.initial_cash),
                "final_equity": float(item.final_equity),
                "total_return_pct": float(item.total_return_pct),
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": max_drawdown_pct,
                "order_count": int(item.order_count),
                "filled_order_count": int(item.filled_order_count),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["period_start", "strategy_order"])
    return metrics.drop(columns=["strategy_order"]).reset_index(drop=True)


def _load_cached_market_data(records_dir: Path) -> pd.DataFrame:
    cache_dir = records_dir / "data_cache"
    pattern = re.compile(r"stock_zh_a_hist_(\d{6})_.*_qfq\.csv$")
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("stock_zh_a_hist_*_qfq.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        frames.append(normalize_ohlcv_frame(pd.read_csv(path), match.group(1), volume_unit="hands"))

    if not frames:
        raise FileNotFoundError(f"No cached AKShare qfq files found in {cache_dir}")

    data = pd.concat(frames, ignore_index=True)
    return data.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])


def _sector_curve(data: pd.DataFrame, period_start: str, period_end: str) -> pd.DataFrame:
    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    window = close.loc[(close.index >= pd.Timestamp(period_start)) & (close.index <= pd.Timestamp(period_end))]
    base = window.ffill().bfill().iloc[0]
    normalized = window.divide(base) * 100.0
    return pd.DataFrame(
        {
            "date": normalized.index,
            "mean_index": normalized.mean(axis=1),
            "median_index": normalized.median(axis=1),
        }
    )


def _load_fund_history(records_dir: Path, benchmark_key: str) -> pd.DataFrame:
    benchmark = FUND_BENCHMARKS[benchmark_key]
    cache_dir = records_dir / "fund_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{benchmark_key}_{benchmark['symbol']}.csv"
    if cache_path.exists():
        raw_data = pd.read_csv(cache_path)
    else:
        if benchmark["source"] == "sina_etf_close":
            raw_data = ak.fund_etf_hist_sina(symbol=str(benchmark["symbol"]))
        elif benchmark["source"] == "open_fund_nav":
            raw_data = ak.fund_open_fund_info_em(
                symbol=str(benchmark["symbol"]),
                indicator="单位净值走势",
            )
        else:
            raise ValueError(f"Unsupported benchmark source: {benchmark['source']!r}")
        raw_data.to_csv(cache_path, index=False)

    if "净值日期" in raw_data:
        raw_data = raw_data.rename(columns={"净值日期": "date", "单位净值": "close"})

    raw_data["date"] = pd.to_datetime(raw_data["date"])
    raw_data["close"] = pd.to_numeric(raw_data["close"], errors="coerce")
    return raw_data[["date", "close"]].dropna().drop_duplicates("date").sort_values("date")


def load_fund_benchmarks(records_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        benchmark_key: _load_fund_history(records_dir, benchmark_key)
        for benchmark_key in FUND_BENCHMARKS
    }


def _fund_benchmark_curve(fund_data: pd.DataFrame, period_start: str, period_end: str) -> pd.DataFrame:
    window = fund_data.loc[
        (fund_data["date"] >= pd.Timestamp(period_start))
        & (fund_data["date"] <= pd.Timestamp(period_end))
    ].copy()
    if window.empty:
        raise RuntimeError(f"No fund benchmark data for {period_start} to {period_end}")

    benchmark = window["close"] / float(window["close"].iloc[0]) * 100.0
    return pd.DataFrame(
        {
            "date": list(window["date"]),
            "benchmark_index": list(benchmark.to_numpy()),
        }
    )


def add_fund_benchmarks(metrics: pd.DataFrame, fund_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in metrics.to_dict("records"):
        updated = dict(row)
        for benchmark_key, frame in fund_data.items():
            curve = _fund_benchmark_curve(
                frame,
                str(row["period_start"]),
                str(row["period_end"]),
            )
            benchmark_return_pct = float(curve["benchmark_index"].iloc[-1] / curve["benchmark_index"].iloc[0] - 1.0) * 100.0
            updated[f"{benchmark_key}_return_pct"] = benchmark_return_pct
            updated[f"alpha_vs_{benchmark_key}_pct"] = float(row["total_return_pct"]) - benchmark_return_pct
        rows.append(updated)
    return pd.DataFrame(rows)


def build_sector_summary(data: pd.DataFrame) -> pd.DataFrame:
    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    rows: list[dict[str, object]] = []
    periods = [
        ("2020Q4 available", "2020-10-09", "2020-12-31"),
        ("2021", "2021-01-04", "2021-12-31"),
        ("2022H1", "2021-12-31", "2022-06-30"),
    ]
    for label, start, end in periods:
        start_prices = close.loc[close.index >= pd.Timestamp(start)].iloc[0]
        end_prices = close.loc[close.index <= pd.Timestamp(end)].iloc[-1]
        returns = (end_prices / start_prices - 1.0) * 100.0
        rows.append(
            {
                "period_label": label,
                "start_date": str(close.loc[close.index >= pd.Timestamp(start)].index[0].date()),
                "end_date": str(close.loc[close.index <= pd.Timestamp(end)].index[-1].date()),
                "average_return_pct": float(returns.mean()),
                "median_return_pct": float(returns.median()),
                "positive_count": int((returns > 0).sum()),
                "negative_count": int((returns < 0).sum()),
                "stock_count": int(returns.count()),
            }
        )
    return pd.DataFrame(rows)


def write_return_chart(metrics: pd.DataFrame, path: Path) -> None:
    width, height = 980, 460
    left, right, top, bottom = 74, 34, 78, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = list(metrics["total_return_pct"])
    y_min = min(-5.0, math.floor(min(values) / 5.0) * 5.0)
    y_max = max(35.0, math.ceil(max(values) / 5.0) * 5.0)
    zero_y = _scale(0.0, y_min, y_max, top + plot_h, top)
    bar_w = 90
    gap = (plot_w - bar_w * len(values)) / max(1, len(values) - 1)

    body = [
        _svg_text(left, 36, "Strategy Returns", size=24, weight="700"),
        _svg_text(left, 60, "Total return is the primary task metric", size=14, fill="#596773"),
        _line(left, zero_y, left + plot_w, zero_y, "#9aa8b3", 1.2),
    ]
    for tick in range(int(y_min), int(y_max) + 1, 10):
        y = _scale(tick, y_min, y_max, top + plot_h, top)
        body.append(_line(left, y, left + plot_w, y, "#e6ebef", 1.0))
        body.append(_svg_text(left - 10, y + 4, f"{tick}%", size=12, fill="#64727e", anchor="end"))

    for idx, row in enumerate(metrics.itertuples(index=False)):
        x = left + idx * (bar_w + gap)
        value = float(row.total_return_pct)
        y = _scale(value, y_min, y_max, top + plot_h, top)
        bar_y = min(y, zero_y)
        bar_h = abs(zero_y - y)
        color = STRATEGY_COLORS.get(str(row.strategy_name), "#667085")
        body.append(_rect(x, bar_y, bar_w, max(bar_h, 1.0), color))
        body.append(_svg_text(x + bar_w / 2, bar_y - 10, _nice_pct(value), size=13, fill=color, anchor="middle", weight="700"))
        body.append(_svg_text(x + bar_w / 2, top + plot_h + 28, row.period_label, size=12, fill="#53606b", anchor="middle"))
        body.append(_svg_text(x + bar_w / 2, top + plot_h + 48, row.strategy_label, size=12, fill="#27313a", anchor="middle"))

    body.append(_line(left, top, left, top + plot_h, "#bcc7cf", 1.0))
    body.append(_line(left, top + plot_h, left + plot_w, top + plot_h, "#bcc7cf", 1.0))
    path.write_text(_document(width, height, body), encoding="utf-8")


def _write_two_panel_line_chart(
    path: Path,
    title: str,
    subtitle: str,
    panels: list[tuple[str, pd.DataFrame]],
    series: list[tuple[str, str, str]],
    *,
    y_label: str,
    baseline: float | None = None,
    baseline_label: str | None = None,
) -> None:
    width, height = 1100, 520
    margin_left, margin_right, top, bottom = 70, 32, 92, 68
    panel_gap = 54
    panel_w = (width - margin_left - margin_right - panel_gap) / 2
    panel_h = height - top - bottom
    body = [
        _svg_text(margin_left, 38, title, size=24, weight="700"),
        _svg_text(margin_left, 62, subtitle, size=14, fill="#596773"),
    ]

    for panel_idx, (panel_title, data) in enumerate(panels):
        x0 = margin_left + panel_idx * (panel_w + panel_gap)
        y0 = top
        dates = pd.to_datetime(data["date"])
        date_min, date_max = dates.min(), dates.max()
        all_values: list[float] = []
        for column, _label, _color in series:
            all_values.extend(float(v) for v in data[column].dropna())
        value_min = math.floor((min(all_values) - 4.0) / 5.0) * 5.0
        value_max = math.ceil((max(all_values) + 4.0) / 5.0) * 5.0

        body.append(_svg_text(x0, y0 - 18, panel_title, size=15, weight="700"))
        for tick in range(int(value_min), int(value_max) + 1, 10):
            y = _scale(tick, value_min, value_max, y0 + panel_h, y0)
            body.append(_line(x0, y, x0 + panel_w, y, "#e8edf1", 1.0))
            body.append(_svg_text(x0 - 9, y + 4, f"{tick:.0f}", size=11, fill="#66737e", anchor="end"))

        body.append(_line(x0, y0, x0, y0 + panel_h, "#bcc7cf", 1.0))
        body.append(_line(x0, y0 + panel_h, x0 + panel_w, y0 + panel_h, "#bcc7cf", 1.0))
        body.append(_svg_text(x0, y0 + panel_h + 30, str(date_min.date()), size=11, fill="#66737e"))
        body.append(_svg_text(x0 + panel_w, y0 + panel_h + 30, str(date_max.date()), size=11, fill="#66737e", anchor="end"))
        if baseline is not None and value_min <= baseline <= value_max:
            baseline_y = _scale(baseline, value_min, value_max, y0 + panel_h, y0)
            body.append(_dashed_line(x0, baseline_y, x0 + panel_w, baseline_y, "#111827", 1.2))

        for column, label, color in series:
            points: list[tuple[float, float]] = []
            for row in data[["date", column]].dropna().itertuples(index=False):
                x = _scale(pd.Timestamp(row.date).toordinal(), date_min.toordinal(), date_max.toordinal(), x0, x0 + panel_w)
                y = _scale(float(getattr(row, column)), value_min, value_max, y0 + panel_h, y0)
                points.append((x, y))
            body.append(_polyline(points, color, 2.5))

        legend_x = x0 + 14
        legend_y = y0 + 18
        legend_step = min(158.0, max(80.0, (panel_w - 60.0) / max(1, len(series))))
        for idx, (_column, label, color) in enumerate(series):
            lx = legend_x + idx * legend_step
            body.append(_line(lx, legend_y, lx + 24, legend_y, color, 3.0))
            body.append(_svg_text(lx + 32, legend_y + 4, label, size=12, fill="#394752"))
        if baseline is not None and baseline_label:
            lx = legend_x + len(series) * legend_step
            body.append(_dashed_line(lx, legend_y, lx + 24, legend_y, "#111827", 1.2))
            body.append(_svg_text(lx + 32, legend_y + 4, baseline_label, size=12, fill="#394752"))

    body.append(_svg_text(20, top + panel_h / 2, y_label, size=12, fill="#64727e", anchor="middle", rotate=-90))
    path.write_text(_document(width, height, body), encoding="utf-8")


def write_equity_chart(records_dir: Path, metrics: pd.DataFrame, path: Path) -> None:
    panels: list[tuple[str, pd.DataFrame]] = []
    for (period_start, period_end), period_label in PERIOD_LABELS.items():
        merged: pd.DataFrame | None = None
        for strategy_name in ("weak_reversal", "tf_reversal"):
            equity = _load_equity(records_dir, strategy_name, period_start, period_end)
            column = f"{strategy_name}_index"
            equity[column] = equity["equity"] / float(equity["equity"].iloc[0]) * 100.0
            current = equity[["date", column]]
            merged = current if merged is None else pd.merge(merged, current, on="date", how="outer")
        assert merged is not None
        panels.append((period_label, merged.sort_values("date")))

    write_series = [
        ("weak_reversal_index", STRATEGY_LABELS["weak_reversal"], STRATEGY_COLORS["weak_reversal"]),
        ("tf_reversal_index", STRATEGY_LABELS["tf_reversal"], STRATEGY_COLORS["tf_reversal"]),
    ]
    _write_two_panel_line_chart(
        path,
        "Strategy Equity Curves",
        "Each strategy is normalized to 100 at the period start",
        panels,
        write_series,
        y_label="Equity index",
        baseline=100.0,
        baseline_label="Initial capital",
    )
    del metrics


def write_benchmark_comparison_chart(metrics: pd.DataFrame, path: Path) -> None:
    width, height = 1120, 470
    left, right, top, bottom = 74, 34, 78, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    bars: list[dict[str, object]] = []
    for period_label, group in metrics.groupby("period_label", sort=False):
        first = group.iloc[0]
        bars.append(
            {
                "period_label": period_label,
                "label": "HS300 ETF",
                "value": float(first["hs300_etf_return_pct"]),
                "color": FUND_BENCHMARKS["hs300_etf"]["color"],
            }
        )
        bars.append(
            {
                "period_label": period_label,
                "label": "Baijiu LOF",
                "value": float(first["csi_baijiu_lof_return_pct"]),
                "color": FUND_BENCHMARKS["csi_baijiu_lof"]["color"],
            }
        )
        for row in group.itertuples(index=False):
            short_label = "WR" if str(row.strategy_name) == "weak_reversal" else "TFR"
            bars.append(
                {
                    "period_label": period_label,
                    "label": short_label,
                    "value": float(row.total_return_pct),
                    "color": STRATEGY_COLORS.get(str(row.strategy_name), "#667085"),
                }
            )

    values = [float(item["value"]) for item in bars]
    y_min = min(-25.0, math.floor(min(values) / 5.0) * 5.0)
    y_max = max(35.0, math.ceil(max(values) / 5.0) * 5.0)
    zero_y = _scale(0.0, y_min, y_max, top + plot_h, top)
    period_count = metrics["period_label"].nunique()
    group_gap = 72
    bar_gap = 16
    bar_w = 58
    group_w = (plot_w - group_gap * (period_count - 1)) / period_count

    body = [
        _svg_text(left, 36, "Strategy Returns vs ETF/LOF", size=24, weight="700"),
        _svg_text(left, 60, "Benchmarks: 510300 HS300 ETF close and 161725 Baijiu LOF NAV", size=14, fill="#596773"),
        _line(left, zero_y, left + plot_w, zero_y, "#9aa8b3", 1.2),
    ]
    for tick in range(int(y_min), int(y_max) + 1, 10):
        y = _scale(tick, y_min, y_max, top + plot_h, top)
        body.append(_line(left, y, left + plot_w, y, "#e6ebef", 1.0))
        body.append(_svg_text(left - 10, y + 4, f"{tick}%", size=12, fill="#64727e", anchor="end"))

    for group_idx, (period_label, group) in enumerate(metrics.groupby("period_label", sort=False)):
        group_x = left + group_idx * (group_w + group_gap)
        group_bars = [
            ("HS300 ETF", float(group.iloc[0]["hs300_etf_return_pct"]), FUND_BENCHMARKS["hs300_etf"]["color"]),
            ("Baijiu LOF", float(group.iloc[0]["csi_baijiu_lof_return_pct"]), FUND_BENCHMARKS["csi_baijiu_lof"]["color"]),
        ]
        for row in group.itertuples(index=False):
            label = "WR" if str(row.strategy_name) == "weak_reversal" else "TFR"
            group_bars.append((label, float(row.total_return_pct), STRATEGY_COLORS.get(str(row.strategy_name), "#667085")))

        total_bars_w = len(group_bars) * bar_w + (len(group_bars) - 1) * bar_gap
        start_x = group_x + (group_w - total_bars_w) / 2
        for idx, (label, value, color) in enumerate(group_bars):
            x = start_x + idx * (bar_w + bar_gap)
            y = _scale(value, y_min, y_max, top + plot_h, top)
            bar_y = min(y, zero_y)
            bar_h = abs(zero_y - y)
            body.append(_rect(x, bar_y, bar_w, max(bar_h, 1.0), color, opacity=0.9))
            body.append(_svg_text(x + bar_w / 2, bar_y - 10, _nice_pct(value), size=12, fill=color, anchor="middle", weight="700"))
            body.append(_svg_text(x + bar_w / 2, top + plot_h + 25, label, size=11, fill="#27313a", anchor="middle"))
        body.append(_svg_text(group_x + group_w / 2, top + plot_h + 52, str(period_label), size=13, fill="#53606b", anchor="middle", weight="700"))

    body.append(_line(left, top, left, top + plot_h, "#bcc7cf", 1.0))
    body.append(_line(left, top + plot_h, left + plot_w, top + plot_h, "#bcc7cf", 1.0))
    path.write_text(_document(width, height, body), encoding="utf-8")


def write_risk_chart(metrics: pd.DataFrame, path: Path) -> None:
    width, height = 1050, 500
    left, top, bottom = 68, 118, 78
    gap = 58
    panel_w = (width - left - 34 - gap) / 2
    panel_h = height - top - bottom
    body = [
        _svg_text(left, 36, "Risk Metrics", size=24, weight="700"),
        _svg_text(left, 62, "Annualized Sharpe ratio (rf=0) and max drawdown from period equity curves", size=14, fill="#596773"),
    ]

    panels = [
        ("Sharpe ratio", "sharpe_ratio", -0.2, 2.8, ""),
        ("Max drawdown", "max_drawdown_pct", 0.0, 45.0, "%"),
    ]
    for panel_idx, (title, column, y_min, y_max, suffix) in enumerate(panels):
        x0 = left + panel_idx * (panel_w + gap)
        body.append(_svg_text(x0, top - 18, title, size=15, weight="700"))
        zero_y = _scale(0.0, y_min, y_max, top + panel_h, top)
        for tick in range(math.floor(y_min), math.ceil(y_max) + 1):
            if column == "max_drawdown_pct" and tick % 10 != 0:
                continue
            if column == "sharpe_ratio" and tick not in [0, 1, 2]:
                continue
            y = _scale(float(tick), y_min, y_max, top + panel_h, top)
            body.append(_line(x0, y, x0 + panel_w, y, "#e8edf1", 1.0))
            body.append(_svg_text(x0 - 9, y + 4, f"{tick}{suffix}", size=11, fill="#66737e", anchor="end"))
        body.append(_line(x0, zero_y, x0 + panel_w, zero_y, "#9aa8b3", 1.1))
        body.append(_line(x0, top, x0, top + panel_h, "#bcc7cf", 1.0))
        body.append(_line(x0, top + panel_h, x0 + panel_w, top + panel_h, "#bcc7cf", 1.0))

        bar_w = 58
        inner_gap = (panel_w - bar_w * len(metrics)) / max(1, len(metrics) - 1)
        for idx, row in enumerate(metrics.itertuples(index=False)):
            x = x0 + idx * (bar_w + inner_gap)
            value = float(getattr(row, column))
            y = _scale(value, y_min, y_max, top + panel_h, top)
            bar_y = min(y, zero_y)
            bar_h = abs(zero_y - y)
            color = STRATEGY_COLORS.get(str(row.strategy_name), "#667085")
            body.append(_rect(x, bar_y, bar_w, max(bar_h, 1.0), color, opacity=0.9))
            label = f"{value:.2f}{suffix}" if suffix else f"{value:.2f}"
            body.append(_svg_text(x + bar_w / 2, bar_y - 9, label, size=11, fill=color, anchor="middle", weight="700"))
            body.append(_svg_text(x + bar_w / 2, top + panel_h + 26, row.period_label, size=11, fill="#53606b", anchor="middle"))
            short_label = "WR" if str(row.strategy_name) == "weak_reversal" else "TFR"
            body.append(_svg_text(x + bar_w / 2, top + panel_h + 44, short_label, size=10, fill="#27313a", anchor="middle"))

    path.write_text(_document(width, height, body), encoding="utf-8")


def write_sector_chart(data: pd.DataFrame, path: Path) -> None:
    summary = build_sector_summary(data)
    width, height = 980, 500
    left, right, top, bottom = 80, 40, 116, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.0, 17.0
    body = [
        _svg_text(left, 36, "Liquor Stock Breadth", size=24, weight="700"),
        _svg_text(left, 60, "How many of the 17 liquor stocks rose or fell in each period", size=14, fill="#596773"),
    ]
    for tick in [0, 5, 10, 15, 17]:
        y = _scale(float(tick), y_min, y_max, top + plot_h, top)
        body.append(_line(left, y, left + plot_w, y, "#e8edf1", 1.0))
        body.append(_svg_text(left - 10, y + 4, str(tick), size=12, fill="#64727e", anchor="end"))

    group_count = len(summary)
    group_gap = 52
    group_w = (plot_w - group_gap * (group_count - 1)) / group_count
    bar_w = 76
    bar_gap = 18
    up_color = "#2f6f95"
    down_color = "#d05a4e"
    for idx, row in enumerate(summary.itertuples(index=False)):
        group_x = left + idx * (group_w + group_gap)
        bars = [
            ("Up", int(row.positive_count), up_color),
            ("Down", int(row.negative_count), down_color),
        ]
        total_w = len(bars) * bar_w + (len(bars) - 1) * bar_gap
        start_x = group_x + (group_w - total_w) / 2
        for bar_idx, (label, value, color) in enumerate(bars):
            x = start_x + bar_idx * (bar_w + bar_gap)
            y = _scale(float(value), y_min, y_max, top + plot_h, top)
            body.append(_rect(x, y, bar_w, top + plot_h - y, color, opacity=0.9))
            body.append(_svg_text(x + bar_w / 2, y - 10, value, size=14, fill=color, anchor="middle", weight="700"))
            body.append(_svg_text(x + bar_w / 2, top + plot_h + 25, label, size=12, fill="#27313a", anchor="middle"))
        body.append(_svg_text(group_x + group_w / 2, top + plot_h + 52, row.period_label, size=13, fill="#53606b", anchor="middle", weight="700"))
        body.append(
            _svg_text(
                group_x + group_w / 2,
                top + plot_h + 74,
                f"avg {row.average_return_pct:.1f}%, med {row.median_return_pct:.1f}%",
                size=12,
                fill="#53606b",
                anchor="middle",
            )
        )

    legend_x = left + plot_w - 230
    legend_y = 54
    body.append(_rect(legend_x, legend_y - 11, 13, 13, up_color, rx=1.5))
    body.append(_svg_text(legend_x + 20, legend_y, "Rising stocks", size=12, fill="#394752"))
    body.append(_rect(legend_x + 116, legend_y - 11, 13, 13, down_color, rx=1.5))
    body.append(_svg_text(legend_x + 136, legend_y, "Falling stocks", size=12, fill="#394752"))
    body.append(_line(left, top, left, top + plot_h, "#bcc7cf", 1.0))
    body.append(_line(left, top + plot_h, left + plot_w, top + plot_h, "#bcc7cf", 1.0))
    body.append(_svg_text(22, top + plot_h / 2, "Stock count", size=12, fill="#64727e", anchor="middle", rotate=-90))
    path.write_text(_document(width, height, body), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records_dir = Path(args.records_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    market_data = _load_cached_market_data(records_dir)
    fund_data = load_fund_benchmarks(records_dir)
    metrics = add_fund_benchmarks(build_strategy_metrics(records_dir), fund_data)
    sector_summary = build_sector_summary(market_data)

    metrics.to_csv(figures_dir / "strategy_metrics.csv", index=False)
    sector_summary.to_csv(figures_dir / "sector_summary.csv", index=False)
    write_return_chart(metrics, figures_dir / "strategy_returns.svg")
    write_equity_chart(records_dir, metrics, figures_dir / "equity_curves.svg")
    write_benchmark_comparison_chart(metrics, figures_dir / "benchmark_comparison.svg")
    write_risk_chart(metrics, figures_dir / "risk_metrics.svg")
    write_sector_chart(market_data, figures_dir / "liquor_sector.svg")

    print(f"Wrote report figures to {figures_dir}")
    print(
        metrics[
            [
                "strategy_name",
                "period_label",
                "total_return_pct",
                "hs300_etf_return_pct",
                "alpha_vs_hs300_etf_pct",
                "csi_baijiu_lof_return_pct",
                "alpha_vs_csi_baijiu_lof_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
            ]
        ].to_string(index=False)
    )
    print(sector_summary.to_string(index=False))


if __name__ == "__main__":
    main()
