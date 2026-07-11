# Itzi flood model

## Common commands
- Run a single test: `uv run pytest tests/my_test.py`
- Enforce code formatting: `uvx ruff format .`
- After editing a *.pyx file, recompile with `uv pip install -e .`; if not the old binary will continue to be used.

## Code style
- Since the arguments types and return types are already documented by the hints, there's no need to duplicate this information in the docstrings.
- Apart from particular cases, prefer pydantic BaseModel to dataclass
- Place imports at the top of the file. Only break this rule to prevent heavy imports in a rarely used function (for example, CLI options).

## Type hints
- Use python type hints. When a function that does not yet use hints is substantially edited, take the opportunity to add type hints.
- Do not quote class names in hints. Use `from __future__ import annotations` when necessary.
- Similarly, use pipe `|` instead of `Union`, `dict` instead of `Dict`, etc.

## General comments
- The project uses `uv`. To run a command in the correct environment, use `uv run`
- Running the whole test suite is slow. Do it only after all the specific tests are passing, as a final check.
