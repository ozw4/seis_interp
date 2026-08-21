# Runs

This directory stores machine-generated execution records. Do not edit run outputs by hand and do not commit large run directories.

The expected layout is:

```text
runs/<study-id>/<YYYYMMDDThhmmssZ_gitsha>/
├── run.json
├── config.resolved.yaml
├── inputs.lock.yaml
├── metrics.json
├── logs/
├── artifacts/
├── figures/
└── tables/
```

Accepted figures, tables, or models are promoted separately to `results/` when that directory is needed.
