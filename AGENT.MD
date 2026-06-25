# Project Agent Notes

## Package Management

- This project uses `uv` to manage Python dependencies and the local virtual environment.
- Keep dependency declarations in `pyproject.toml`.
- Keep the resolved dependency lockfile in `uv.lock`.
- Do not install project dependencies with plain `pip install`; use `uv` instead.

Common commands:

```bash
uv sync
uv add <package>
uv remove <package>
```

## Running Python

- Run Python code through `uv run` so commands use the project's managed environment.
- Prefer:

```bash
uv run quant-backtest
uv run python -m quant
uv run python <script.py>
uv run python -m <module>
```

- If tests are added later, run them through `uv run` as well, for example:

```bash
uv run python -m unittest discover -s tests
```

## Development Conventions

- When adding a new runtime dependency, use `uv add <package>` so both `pyproject.toml` and `uv.lock` stay in sync.
- When adding a development-only dependency, use `uv add --dev <package>`.
- Before running scripts or notebooks that depend on project packages, run `uv sync` if the environment may be stale.
- Avoid committing local virtual environment contents; `.venv/` is local infrastructure managed by `uv`.
