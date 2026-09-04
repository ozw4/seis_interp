# Runs

This directory stores machine-generated execution records. Do not edit run outputs by hand and do not commit large run directories.

The expected layout is:

```text
runs/<study-id>/<YYYYMMDDThhmmssZ_gitsha[_condition]>/
├── config.resolved.yaml
├── inputs.lock.json
├── metrics.json
├── run.json
└── artifacts/
    └── best.pt
```

A runner that trains several conditions in one invocation writes one such directory per condition plus a sibling summary JSON sharing the timestamp and Git SHA prefix; the training-fit diagnostics record metrics only and write no `artifacts/`.

Formal study run directories are immutable. Scratch workspaces labeled in [`studies/README.md`](../studies/README.md) instead maintain an overwriteable current output under their own `runs/` subdirectory.

Accepted figures, tables, or models are promoted separately to `results/` when that directory is needed.
