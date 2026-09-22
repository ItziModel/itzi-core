# Itzi flood model

## Common commands
- Run a single test: `uv run pytest tests/my_test.py`
- Enforce code formatting: `uvx ruff format .`
- After editing a *.pyx file, recompile with `uv pip install -e .`; if not the old binary will continue to be used.

## Code style
- Since the arguments and return types are already documented by the type annotation, there's no need to duplicate this information in the docstrings.
- Use pydantic BaseModel when validating data (for example, user input).
- Avoid functions with more than 5 arguments. Use a data structure when necessary.
- Place imports at the top of the file. Only break this rule to prevent heavy imports in a rarely used function (for example, CLI options).

## Type annotation
- Use python type hints. When a function that does not yet use annotations is substantially edited, take the opportunity to add typing information.
- Do not quote class names in hints. Use `from __future__ import annotations` when necessary.
- Similarly, use pipe `|` instead of `Union`, `dict` instead of `Dict`, etc.

## General comments
- The project uses `uv`. To run a command in the correct environment, use `uv run`
- Running the whole test suite is slow. Do it only after all the specific tests are passing, as a final check.
