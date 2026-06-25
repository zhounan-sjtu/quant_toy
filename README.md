# Quant

白酒股 AKQuant 回测项目。

## Project Layout

- `src/quant/`: 核心源码，包括数据准备、策略、回测和结果输出。
- `tests/`: 单元测试和回测逻辑校验。
- `docs/`: 任务说明和项目文档。
- `notebooks/`: 交互式实验 notebook。
- `artifacts/`: 本地运行产物、缓存和回测记录。

## Commands

```bash
uv sync
uv run quant-backtest
uv run python -m quant
uv run python -m unittest discover -s tests
```
