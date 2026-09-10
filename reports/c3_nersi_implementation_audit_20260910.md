# C3 NeRSI implementation audit — 2026-09-10

## Scope and status

This audit covers the repository NeRSI reimplementation, its verified-volume pipeline and
CLI, fixed-suite preflight/harness integration, Study 034 plan, checkpoint restoration,
saved-prediction re-scoring, and optional first-results comparison. It is a repository
contract audit, not a claim of complete reproduction of Gao et al.

No unresolved correctness or target-leakage blocker was found after the fixes recorded
below. The implementation remains a per-volume model: a checkpoint represents exactly one
verified benchmark case and volume and is not a reusable pretrained model.

The worktree was already dirty and contained unrelated CCNet and study work. Those changes
were preserved. The implementation audit itself created no commit; commits were organized
only after a separate explicit follow-up request.

## Static leakage audit

| Boundary | Finding |
|---|---|
| Input loading | The native pipeline enters through `load_c3_volume_run_inputs`; training receives the zero-filled `ObservedC3Volume`, not a target-amplitude buffer. The frozen-suite preflight uses the corresponding leakage-safe public volume loader. |
| Normalization | `build_c3_volume_nersi_data` accumulates global RMS in float64 from `values[:, observed_trace_mask]` only. True observed zeros remain in the count; target traces do not. |
| Loss | The trainer broadcasts the trace mask across time and uses boolean selection before subtraction. Masked NaNs therefore do not enter the loss or graph. A focused test covers this case. |
| Profile coverage | Profiles with no observed receiver-y trace are excluded from stable training candidates but remain in the coordinate array and full prediction domain. |
| Training lifecycle | Adam performs exactly the configured update count. There is no evaluator, early stopping, best-checkpoint selection, retry, target statistic, or target-derived normalization in the training path. |
| Evaluation boundary | `predict_c3_volume_nersi` completes the full physical prediction and hard observed reinsertion before `evaluate_c3_volume_prediction` is called. Call-order instrumentation covers this boundary. Metrics do not feed back into training. |
| Study scope | Study 034 resolves the one frozen validation case. Its plan declares `test_execution: false`; neither its native run nor preflight resolves a test truth path. |
| Checkpoint | The payload contains model/preprocessing/training-final state plus case ID, volume ID, benchmark-case SHA-256, and volume-file SHA-256 values. It contains no optimizer, RNG state, evaluator result, target metric, best-selection record, or target waveform. |

The real dry-run resolved suite SHA-256
`f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee`, case
`c3_benchmark_validation_random_trace_80_seed142`, shape `[384, 9, 32, 8, 32]`,
14,729 observed traces, and 58,999 evaluation-target traces. The target evaluation domain
therefore remains all 58,999 traces × 384 samples = 22,655,616 samples. The case/project
mask seed is 142; the independently declared initialization/profile-sampling seed is
20260908.

## Determinism, restoration, and independent scoring

The deterministic CPU audit runs the complete native pipeline twice in separate output
directories on a `(8, 2, 3, 2, 8)` fixture for two Adam steps. With the same configuration
and seed, it requires byte-identical resolved configuration and input lock, exact training
history/final loss and numeric metrics, `array_equal` predictions, identical constructor
configuration, and exact state-dict tensors. Volatile UTC, Git, elapsed-time, RSS, and other
resource metadata are intentionally excluded.

The restore audit loads `final.pt` with `weights_only=True` on CPU, reconstructs the model
from the complete constructor configuration, and loads its state strictly. Before reuse,
the public binding validator compares the checkpoint's case/volume/hashes, coordinate and
profile orders, spatial/profile shapes, and observed-only amplitude scale with current
verified inputs. Re-prediction uses the current validated adapter data, restores amplitude
once, applies the same hard reinsertion, and matches `prediction.npy` at `rtol=1e-6`,
`atol=1e-6`. Tests reject a different case/volume binding and different preprocessing.

The independent-scoring audit reloads `prediction.npy` and the verified input contract,
checks full shape, floating dtype, finiteness, hard observed consistency, and unchanged
target count, then independently recomputes target-only physical S/N, RMSE, reference
energy, and error energy. The result matches the native metrics. Run metadata records both
checkpoint and prediction SHA-256 values; the optional comparison collector requires and
rechecks both for NeRSI.

## Optional comparison behavior

NeRSI is optional in `c3_first_results`; the five Study 028 methods remain required exactly
as before. If no NeRSI path is supplied, no NeRSI row or new comparison-context column is
emitted, preserving the legacy five-method output structure. If explicitly supplied, the
collector validates method identity, case and volume IDs, input hashes, final checkpoint
role/budget, prediction coverage, target count/domain, and independently re-scores the
saved prediction.

The optional comparison context distinguishes per-volume training, additional supervised
training data, and target-volume optimization. NeRSI total time is its observed-only
per-volume training plus prediction. GNN pretraining and frozen prediction remain separate;
the report does not claim equal training information or compute. Paper-public S/N values
remain study context and are not placed in the repository-result metric column.

## Findings fixed during audit

1. **High, resolved — wrong-volume checkpoint reuse:** the initial checkpoint had only
   shape/preprocessing metadata, so an equal-shaped checkpoint could be paired silently
   with another case. Required case/volume/hash binding and current-data validation were
   added, with cross-volume regression coverage.
2. **Medium, resolved — legacy summary stability:** comparison-context fields were initially
   emitted for every five-method-only summary. They are now emitted only when an explicit
   NeRSI run is included.
3. **Medium, resolved — artifact/cross-record integrity:** NeRSI run metadata now fixes the
   final checkpoint and prediction SHA-256 values. The collector requires those digests and
   validates the NeRSI metadata/metrics case and volume IDs.
4. **Low, resolved — paper/config metadata:** the paper-alignment record now states the C3
   paper contract `K=40` separately from the configured Fourier component count used by a
   synthetic test or native run.
5. **Low, resolved — GPU preflight parity:** NeRSI preflight now uses the same model-seeding
   and cuDNN numerical-mode initializer as the full pipeline, while retaining RNG isolation
   and disposable state.

## Quality gates

Executed from `/workspace` on 2026-09-10 UTC:

```bash
pytest -q \
  tests/unit/test_nersi.py \
  tests/unit/test_c3_volume_nersi_data.py \
  tests/unit/test_fixed_step_nersi.py \
  tests/unit/test_c3_volume_nersi_prediction.py \
  tests/unit/test_nersi_checkpoints.py \
  tests/unit/test_nersi_pipeline_config.py \
  tests/unit/test_interpolate_commands.py \
  tests/unit/test_c3_first_results.py \
  tests/unit/test_c3_first_results_neural_preflight.py \
  tests/unit/test_c3_nersi_study.py \
  tests/integration/test_interpolate_nersi.py \
  tests/integration/test_c3_first_results_bridge.py \
  tests/integration/test_c3_first_results_neural_preflight.py \
  tests/integration/test_c3_first_results_checks.py
```

Result: **321 passed in 61.79 s**.

```bash
ruff check .
```

Result: **pass — `All checks passed!`**.

```bash
ruff format --check .
```

Result: **pass — 514 files already formatted**.

```bash
git diff --check
```

Result: **pass**.

The full repository pytest suite was not run, in accordance with the repository instruction
to use focused tests unless a full run is explicitly requested.

## Real-data dry-runs and deferred execution

The real suite manifest and two CUDA devices were present. The preflight plan and all four
candidate plans were resolved read-only. Each returned `status=dry_run` and `writes=false`;
Candidates A–D resolved respectively to `(width, latent, lr)` values
`(256,64,1e-3)`, `(256,64,3e-4)`, `(384,96,1e-3)`, and `(384,96,3e-4)`, all with 5,000
updates.

The disposable GPU preflight and four full validation candidates were **not executed**.
Although data and GPUs are available, the request authorized implementation, not the
Task 10 long-running `--execute` sequence. Consequently there is no measured Study 034
timing, memory, validation score, selected candidate, result artifact, or adoption decision,
and none is inferred here.

When explicitly authorized, run in this order:

```bash
python -m seis_interp.pipelines.c3_first_results \
  --config studies/study_034_c3_nersi_baseline/config_preflight.yaml \
  --inputs studies/study_034_c3_nersi_baseline/inputs.yaml \
  --action nersi --preflight --execute

for candidate in a b c d; do
  python -m seis_interp.pipelines.c3_first_results \
    --config "studies/study_034_c3_nersi_baseline/config_candidate_${candidate}.yaml" \
    --inputs studies/study_034_c3_nersi_baseline/inputs.yaml \
    --action nersi --execute
done
```

Before candidate execution, record the successful preflight hashes, timing, memory, and
linear estimates in `decisions.md` and confirm they fit the declared timeout. After all four
immutable runs, independently re-score each saved prediction before applying the
predeclared validation selection rule. Do not run the test partition.
