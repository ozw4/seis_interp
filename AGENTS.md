# Repository instructions

## Objective

Build a focused proof of concept for coordinate-based multidimensional seismic interpolation. The first validation dataset is SEG C3 Narrow-Azimuth. The work is paper-inspired; exact reproduction of every published experiment is not required.

## Required context

Before changing a study, read:

1. `docs/repository_layout.md`
2. the relevant `studies/<study>/README.md`
3. the study `config.yaml`
4. the study `inputs.yaml`
5. current code and tests related to the task

Read the relevant `decisions.md` only when:

- changing a study condition
- changing an accepted evaluation rule
- investigating why a current condition was adopted
- reviewing study history or rationale at the user's request

Treat repository artifacts according to these roles:

- `README.md`: current research contract in prose
- `config.yaml`: executable conditions
- `inputs.yaml`: executable input contract
- current code and tests: implementation behavior
- `decisions.md`: historical rationale, not implementation specification
- `runs/`: immutable execution facts

Do not infer current implementation requirements from historical decisions. Current code, tests, resolved configuration, and input manifests take precedence.

## Engineering rules

- Write small functions with one responsibility.
- Avoid abstractions that are not required by the current POC.
- Implement a working, testable path before optimizing performance.
- Put reusable logic in `src/seis_interp/`.
- Put research conditions and decisions in `studies/`.
- Keep pipelines thin; computational logic belongs in focused modules.
- Do not create `utils.py`, `misc.py`, `common.py`, or root-level analysis scripts.
- Do not add empty future-use directories.

## Data and reproducibility

- Never commit real SEG-Y files, large arrays, model checkpoints, credentials, or absolute host paths.
- Treat SEG C3 NA as external data and record source, version, and checksum in its manifest.
- Split train, validation, and test sets at trace level; do not leak time samples from the same trace across splits.
- Save resolved configuration, input locks, seed, Git SHA, metrics, and environment metadata for each run.
- Do not hand-edit generated run outputs.

## Quality gates

Run before proposing changes:

```bash
ruff check .
ruff format --check .
pytest
```

Use `python -m seis_interp.cli doctor` to inspect Python, PyTorch/CUDA, seismic dependencies, Codex, Claude Code, and the configured data root.
