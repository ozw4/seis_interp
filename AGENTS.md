# Repository instructions

## Implementation rules

- Inspect the relevant current code and tests before editing. Do not discard unrelated changes.
- Keep each change scoped. Do not mix refactoring with behavior changes unless the task requires both.
- Write small functions and classes with one clear responsibility.
- Prefer straightforward code. Avoid speculative abstractions, factories, registries, compatibility layers, and premature optimization.
- Put reusable implementation in `src/seis_interp/`. Follow `docs/repository_layout.md` when choosing locations.
- Keep `pipelines/` limited to orchestration. Pipeline modules must not import private implementation details from other pipeline modules; move shared behavior to a focused `data/`, `processing/`, `training/`, or `evaluation/` module.
- Use responsibility-specific module names. Do not create `utils.py`, `common.py`, or `misc.py`.
- Preserve existing interfaces and observable behavior unless the task explicitly changes them, including CLI arguments, file formats, checkpoint payloads, ordering, random-number sequences, and error contracts.
- Do not add dependencies, `schema_version`, backward-compatibility machinery, or future-use directories unless explicitly required.
- Add or update focused tests for changed behavior.

## Required checks

- Run tests that directly cover the changed behavior and its immediate dependents.
- Do not run the full test suite unless the user explicitly requests it or the change has repository-wide impact.
- Report the exact test commands and results at handoff. Full-suite regression testing is handled by CI.

```bash
ruff check .
ruff format --check .
pytest <relevant test files>
```
