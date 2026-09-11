# Runs

This directory stores machine-generated execution records. Do not edit run outputs by hand and do not commit large run directories.

The five random-80 PoC methods use:

```text
runs/<study-id>/<run-id>/
├── config.resolved.yaml
├── inputs.lock.json
├── metadata.json
├── metrics.json
├── prediction.npy
└── final.pt          # neural methods only
```

POCS and DRR have no checkpoint. NeRSI, CCNet5D, and relational trace graph save the final
fixed-step model. The full physical prediction preserves observations exactly.
Metrics contain only the common evaluator result. Metadata records common identity,
normalization, objective, training/reconstruction, coverage, timing, resource usage, and nested
method details. Input comparisons must include the full selection and nested volume hashes.

Other training pipelines use `run.json` and method-specific files under `artifacts/`.
Model-selection training saves `best.pt`; per-volume SIREN saves `final.pt` and `prediction.npy`.

A runner that trains several conditions in one invocation writes one such directory per condition plus a sibling summary JSON sharing the timestamp and Git SHA prefix; the training-fit diagnostics record metrics only and write no `artifacts/`.

Formal study run directories are immutable. Scratch workspaces labeled in [`studies/README.md`](../studies/README.md) instead maintain an overwriteable current output under their own `runs/` subdirectory.

Shared run pipelines capture `git_commit` and `git_worktree_dirty` in run metadata (`metadata.json.method_details` for PoC) at run start. The dirty flag includes staged, unstaged, and untracked files, excluding ignored files. Dirty runs are allowed for development; promotion to formal results requires `git_worktree_dirty` to be `false`.

Accepted figures, tables, or models are promoted separately to `results/` when that directory is needed.
