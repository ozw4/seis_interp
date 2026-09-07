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
    ├── best.pt
    ├── final.pt
    └── prediction.npy
```

The artifact entries are method-dependent alternatives, not three files required in every run.
Model-selection training pipelines write `best.pt`; fixed-step per-volume SIREN writes `final.pt`
and `prediction.npy`; POCS and DRR interpolation write `prediction.npy` without a checkpoint.
Supervised CCNet5D training writes `patch_plan.json`, `best.pt`, and `final.pt` under `artifacts/`;
frozen CCNet5D inference writes only `prediction.npy` and binds the source checkpoint and its
training provenance in the input lock. Training/internal-selection cost and frozen inference
cost belong to separate runs.
Each run contains only the artifacts produced by its method.

A runner that trains several conditions in one invocation writes one such directory per condition plus a sibling summary JSON sharing the timestamp and Git SHA prefix; the training-fit diagnostics record metrics only and write no `artifacts/`.

Formal study run directories are immutable. Scratch workspaces labeled in [`studies/README.md`](../studies/README.md) instead maintain an overwriteable current output under their own `runs/` subdirectory.

Shared run pipelines capture `git_commit` and `git_worktree_dirty` in `run.json` at run start. The dirty flag includes staged, unstaged, and untracked files, excluding ignored files. Dirty runs are allowed for development; promotion to formal results requires `git_worktree_dirty` to be `false`.

Accepted figures, tables, or models are promoted separately to `results/` when that directory is needed.
