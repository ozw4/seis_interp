# Claude Code instructions

Follow `AGENTS.md` and `docs/repository_layout.md` as the authoritative repository rules.

Keep changes narrowly scoped to the active study. Prefer small functions, explicit names, and tests over broad frameworks. Do not commit SEG-Y files, generated runs, credentials, or machine-specific paths.

Validation commands:

```bash
ruff check .
ruff format --check .
pytest
python -m seis_interp.cli doctor
```
