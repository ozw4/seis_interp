# Launch rehearsal and frozen supervisors

This is an operator procedure. The implementation session runs CPU tests only and
neither prepares persistent real experiments nor launches or signals GPU jobs.
Run the procedure after the study decisions/configurations and implementation are
frozen on a clean launch commit, with both GPUs available and no probes running.
Keep at least 30 GiB free. The configured pinned-process exclusions remain in
force; both devices must be available for the two-device acceptance checks.

The evaluation seeds **990671, 990672, 990673 are reserved/burned for this
rehearsal**, within the existing partition namespaces. Do not reuse them for
scientific evaluation. The real evaluation seeds remain 7321–7323. After an
accepted rehearsal, the operator records its paths, launch commit and PASS table
with these burned seeds in `decisions.md` when that document is next updated.
This new note avoids changing the concurrently maintained study documents.

`rehearsal.yaml` is accepted only for a study ending `_rehearsal`. Preparation
seals its digest, the rehearsal flag and actual S1 counts. Workers load overrides
and faults only when the seal has `rehearsal: true`; a plan added to a real
experiment cannot activate them. The generated diff changes only study identity,
evaluation seeds, the 300-update/report budget, the FORGE CCNet shard count,
DRR's two iterations and the new rehearsal plan. The separate `--determinism`
variant also disables CCNet's cuDNN benchmark. Grid bytes, memory budgets,
imports, input contracts and the POCS configuration remain frozen.

Report faults run after the report is written. A due checkpoint is saved before
that callback: report@250 with a 100-update checkpoint interval resumes from 200;
a report fault at a checkpoint boundary resumes from that boundary. The
before-prediction fault resumes at 300 and performs prediction only. CUDA ReGSI
restore checks are state exact; its atomic accumulation does not promise a
bitwise identical CUDA trajectory. Primary CCNet runs and their uninterrupted
twins keep the frozen `cudnn_benchmark: true` setting. Compare their run-level
validation summaries: absolute relMSE difference at most 1% of the primary
relMSE, absolute mean trace SNR difference at most 0.05 dB, and Pearson loss-curve
correlation at least 0.99 over all 300 updates. FORGE uses the `clean_target`
validation population; C3 uses its validation summary. Identical constant loss
curves pass; other constant curves fail because correlation is undefined.
Require matching steps and prediction shapes, finite losses/predictions/metrics,
and completed manifests. These comparisons cover the benchmark-on restart
trajectory against its uninterrupted twin; re-logged losses only need to be
finite. No individual loss or prediction element is compared for closeness.
Checkpoint state digests must still match exactly.

Independently, run two fault-free CCNet experiments per dataset with
`cudnn_benchmark: false`. These determinism twins must have byte-identical
`history.csv` and `prediction.npy`; they are compared with each other, separately
from the benchmark-on runs. NeRSI's re-logged losses remain exact. Only rehearsal
histories record every update for CCNet and NeRSI.

The rehearsal watchdog floor is 900 seconds. This allows FORGE a05 startup,
cuDNN autotuning and the first 50-update interval under contention, while still
detecting the injected 1800-second ReGSI stall. The production floor is unchanged.

In each faulted rehearsal, DRR `a01_b01` holds at `pre_method` until every neural
method in that experiment has at least one finalized evaluation-seed prediction.
The `barrier_hold` fault polls manifests every ten seconds and requires an
`awaiting_barrier` or `completed` manifest with `prediction_finalized_at` and its
`prediction.npy`. It records those timestamps and `released_at` in the fault
evidence. Its six-hour timeout fails the attempt; retries re-enter the hold even
when the fault marker already exists. DRR cannot finish S1 while the hold waits,
so the pre-barrier prediction criterion is independent of GPU admission order.
The checker requires a pre-barrier prediction for each sealed neural method and
verifies the hold's release evidence. Fault-free twins and import replay omit it.

## Prepare and replay imports

Run these commands from the clean launch checkout. They create new artifacts;
choose a fresh rehearsal root if it already exists. Preparation/check-only uses
CPU, and check-only writes its temporary experiment outside `runs/`.

```bash
cd /workspace
test -z "$(git status --porcelain)" || exit 1
export RH=/workspace/runs/study_067_forge_c3_common_grid_rehearsal
export LAUNCH_COMMIT=$(git rev-parse HEAD)
export PY=/workspace/.venv/bin/python
export PYTHONPATH=/workspace/src
unset CUDA_VISIBLE_DEVICES

"$PY" scripts/make_study067_rehearsal.py --out "$RH/study_src"
"$PY" scripts/make_study067_rehearsal.py --out "$RH/study_twin" --no-faults
"$PY" scripts/make_study067_rehearsal.py --out "$RH/study_determinism_a" --determinism
"$PY" scripts/make_study067_rehearsal.py --out "$RH/study_determinism_b" --determinism
"$PY" scripts/make_study067_rehearsal.py --out "$RH/study_imports" --imports-only
"$PY" scripts/check_study067_rehearsal.py --plan-only "$RH/study_src"
"$PY" scripts/check_study067_rehearsal.py --plan-only "$RH/study_twin"
"$PY" scripts/check_study067_rehearsal.py --plan-only "$RH/study_determinism_a"
"$PY" scripts/check_study067_rehearsal.py --plan-only "$RH/study_determinism_b"
"$PY" scripts/check_study067_rehearsal.py --plan-only "$RH/study_imports"

P() { CUDA_VISIBLE_DEVICES='' "$PY" "scripts/run_$1.py" prepare --repo /workspace --study "$2" "${@:3}"; }
P c3_region_experiment "$RH/study_src/c3" --check-only > "$RH/c3_prepare_check.json"
P forge_grid_experiment "$RH/study_src/forge" --check-only > "$RH/forge_prepare_check.json"
export C3=$(P c3_region_experiment "$RH/study_src/c3" | tail -1)
export FG=$(P forge_grid_experiment "$RH/study_src/forge" | tail -1)
export IC3=$(P c3_region_experiment "$RH/study_imports/c3" | tail -1)
export IFG=$(P forge_grid_experiment "$RH/study_imports/forge" | tail -1)
```

R0 is a CPU replay of every predicted importable S1 cell. The checker requires
its coverage file; this command calculates coverage from the prepared snapshot,
then invokes only those pilots, using that snapshot and an empty CUDA device
list. It neither prints nor examines held-out scores. It is safe to omit this
optional replay for exploration, but then the full checker reports R0 FAIL and
does not claim full acceptance.

```bash
for pair in "c3:$IC3" "forge:$IFG"; do
  dataset=${pair%%:*}; experiment=${pair#*:}
  CUDA_VISIBLE_DEVICES='' PYTHONPATH="$experiment/source/src" "$PY" - "$dataset" "$experiment" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
from seis_interp.data.experiment_import_coverage import experiment_import_coverage
kind, root = sys.argv[1], Path(sys.argv[2])
coverage = experiment_import_coverage(Path('/workspace'), root, kind)
with (root / 'import_coverage.json').open('x') as stream:
    json.dump(coverage, stream, indent=2)
runner = 'c3_region' if kind == 'c3' else 'forge_grid'
for method, values in coverage['methods'].items():
    for candidate in values['s1_sources']:
        subprocess.run([sys.executable, str(root / 'source/scripts' / f'run_{runner}_experiment.py'),
                        'pilot', '--repo', '/workspace', '--experiment', str(root),
                        '--method', method, '--candidate', candidate], check=True, env=os.environ)
PY
 done
```

## Detached launch and evidence sampling

The launcher uses the **frozen script and frozen PYTHONPATH for the supervisor**,
`nohup`, a new session (the `setsid` equivalent), `/dev/null` input and an append
log. Workers also use the frozen CLI. These commands apply to real prepared
experiments too; launching a live `scripts/run_*.py run` directly is insufficient
for a frozen supervisor. The launcher's `--dry-run` prints its exact command and
creates nothing.

```bash
L() { "$PY" scripts/launch_grid_experiment.py "$1" --repo /workspace --experiment "$2" "${@:3}"; }
L forge "$FG" --dry-run
L c3 "$C3" --dry-run
setsid nohup "$PY" scripts/sample_study067_rehearsal.py --repo /workspace \
  --c3 "$C3" --forge "$FG" --out "$RH/gpu_samples.jsonl" \
  > "$RH/sampler.out" 2>&1 </dev/null &
# Admit only the two large shards first. Starting the full FORGE supervisor here
# would let ReGSI reserve a GPU before a05, defeating the launch precondition.
mkdir -p "$FG/workers"
W() {
  local method=$1 shard=$2 worker=$3
  local niceness=()
  if [[ "$method" == pocs_grid || "$method" == drr_grid ]]; then niceness=(nice -n 10); fi
  setsid nohup env PYTHONPATH="$FG/source/src" OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${niceness[@]}" "$PY" "$FG/source/scripts/run_forge_grid_experiment.py" worker \
    --repo /workspace --experiment "$FG" --method "$method" --shard "$shard" \
    >> "$FG/workers/$worker.log" 2>&1 </dev/null &
}
W ccnet5d_grid 0/2 ccnet5d_grid
W ccnet5d_grid 1/2 ccnet5d_grid_shard_1_of_2
```

Within five minutes, require both live FORGE CCNet shards to have admissions on
different GPUs. On success, start the remaining FORGE workers and wait for their
state files, then adopt all workers into the supervisor and launch C3. This avoids
charging a restart to a missing worker on its first launch. Failure exits this
shell block before either supervisor launch; inspect the logs and resource
availability before recovery.

```bash
if "$PY" - <<'PY'
import json, os, time
from pathlib import Path
from seis_interp.training.worker_processes import process_is_alive
root = Path(os.environ['FG']) / 'workers'
deadline = time.monotonic() + 300
while True:
    paths = [root / 'ccnet5d_grid.json', root / 'ccnet5d_grid_shard_1_of_2.json']
    try:
        states = [json.loads(p.read_text()) if p.exists() else {} for p in paths]
    except (OSError, ValueError):
        states = []
    devices = {s.get('last_admission', {}).get('device') for s in states}
    live = len(states) == 2 and all(
        s.get('status') == 'running'
        and process_is_alive(s.get('pid', -1), s.get('process_start_time'))
        and s.get('last_admission', {}).get('run') in ('a05_b02', 'a05_b05')
        for s in states)
    if live and devices == {'cuda:0', 'cuda:1'}:
        break
    if time.monotonic() >= deadline:
        raise SystemExit('both FORGE CCNet shards were not admitted on separate GPUs within 5 min')
    time.sleep(5)
PY
then
  for method in regsi_real nersi_profile pocs_grid drr_grid; do W "$method" 0/1 "$method"; done
  if "$PY" - <<'PY'
import json, os, time
from pathlib import Path
from seis_interp.training.worker_processes import process_is_alive
root = Path(os.environ['FG']) / 'workers'
deadline = time.monotonic() + 60
while True:
    ready = True
    for method in ('regsi_real', 'nersi_profile', 'pocs_grid', 'drr_grid'):
        try:
            state = json.loads((root / f'{method}.json').read_text())
        except (OSError, ValueError):
            ready = False
            continue
        if state.get('status') in ('failed', 'retry_pending'):
            raise SystemExit(f'{method} failed during startup; inspect its log')
        ready &= state.get('status') == 'completed' or process_is_alive(
            state.get('pid', -1), state.get('process_start_time'))
    if ready:
        break
    if time.monotonic() >= deadline:
        raise SystemExit('FORGE worker startup precondition was not met')
    time.sleep(1)
PY
  then
    L forge "$FG" --only-dead-workers && L c3 "$C3"
  else
    exit 1
  fi
else
  exit 1
fi
```

The sampler records JSONL GPU totals/use, live reservations before and after the
GPU query (`slots_before` and `slots_after`, identified by `slot_file`),
worker identities, `/proc` supervisor PYTHONPATH/cmdline, nice values, CPU ticks,
barrier hashes and summary hashes. It stops itself when both supervisors report
completion. Keep its output; a plain `nvidia-smi` text log cannot prove process
or barrier criteria. A failed or malformed `nvidia-smi` query records `gpu_error`
and an empty GPU list for that sample; process/barrier sampling continues and the
next iteration retries the query. Acceptance requires successful GPU samples.

Memory accounting uses only these experiments' slots (identified through worker
and seed-child identities and run manifests in the container PID namespace).
The second slot read waits 1.5 seconds after the GPU query, at least 1.5 times the
one-second admission monitor poll. Each active slot publishes instantaneous
`current_reserved_now_bytes` and
`peak_reserved_bytes` on every monitor poll. `current_reserved_bytes` retains the
30-second minimum exclusively for admission/regrowth decisions. Per device,
owned memory uses each slot's maximum instantaneous reservation across those two
reads, plus 1 GiB per owning process for CUDA context/library allocations outside
PyTorch's allocator. Multiple slots
from one process count its allocator/context once, using its largest instantaneous
reservation. `external = max(0, total_used - owned)`;
`sum(own required_bytes) + external` must fit within total minus the separate
2 GiB device context allowance. Each slot's requirement counts once across both
reads. A capacity excess fails only when observed on the same device in at least
two consecutive samples; a healthy or missing GPU sample breaks the sequence.
This tolerates sub-poll ramp/release artefacts while retaining the steady-state
capacity limit. Other studies' slots are external. A queued or retained retry
reservation also records instantaneous/peak evidence, including
cached memory that has not yet been released. The checker requires instantaneous
evidence and never substitutes the 30-second minimum or the peak.
`nvidia-smi` host process PIDs are never matched against container PIDs.

## Operator supervisor-loss drill

Wait for `$C3/workers/s1_barrier.json`. Only the operator performs the deliberate
supervisor loss below. The command verifies the kernel start identity and frozen
experiment CLI before targeting that supervisor; it does not target workers.

```bash
while ! test -f "$C3/workers/s1_barrier.json"; do sleep 5; done
"$PY" - <<'PY'
import json, os, signal
from pathlib import Path
root = Path(os.environ['C3'])
state = json.loads((root / 'workers/supervisor.json').read_text())
pid = state['pid']
stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
assert int(stat[19]) == state['process_start_time'] and stat[0] not in ('Z', 'X')
cmdline = Path(f'/proc/{pid}/cmdline').read_text().replace('\0', ' ')
assert str(root) in cmdline and str(root / 'source/scripts/run_c3_region_experiment.py') in cmdline
os.kill(pid, signal.SIGKILL)
PY
"$PY" scripts/check_study067_rehearsal.py --snapshot "$C3" --label before_refusal
if PYTHONPATH="$C3/source/src" "$PY" "$C3/source/scripts/run_c3_region_experiment.py" \
  run --repo /workspace --experiment "$C3" > "$C3/refused_supervisor.out" 2>&1; then
  echo 'FAIL: duplicate live launch was accepted'; exit 1
fi
"$PY" scripts/check_study067_rehearsal.py --snapshot "$C3" --label after_refusal
```

Verify the refusal message names a duplicate live worker. The before/after hashes
cover worker PID/start identity, method, shard and source root, plus complete
supervisor, retry-budget, selection and barrier files. Normal changes to worker
phase, status, progress counters and seed-child lists do not change these hashes.
Capture both snapshots after supervisor loss and before recovery, so natural
supervisor restarts cannot race the check. After NeRSI's post-scoring fault leaves
it dead/retry pending, recover immediately with the new mode:

```bash
L c3 "$C3" --only-dead-workers
```

This mode adopts live workers by PID/start identity, skips completed workers and
starts only missing/dead retryable workers. It retains retry budgets and S1
completion. It never signals a worker. Normal `run` retains its duplicate-live
refusal contract. A fresh supervisor lock excludes concurrent supervisors.

After both experiments complete, capture completed summary bytes, relaunch C3 in
that mode once more, wait for it to complete, and capture again:

```bash
"$PY" scripts/check_study067_rehearsal.py --snapshot "$C3" --label before_relaunch
export RELAUNCH_PID=$(L c3 "$C3" --only-dead-workers | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["pid"])')
"$PY" - <<'PY'
import json, os, time
from pathlib import Path
root, pid = Path(os.environ['C3']), int(os.environ['RELAUNCH_PID'])
while True:
    state = json.loads((root / 'workers/supervisor.json').read_text())
    if state.get('pid') == pid and state['status'] in ('completed', 'failed'):
        assert state['status'] == 'completed', state
        break
    time.sleep(1)
PY
"$PY" scripts/check_study067_rehearsal.py --snapshot "$C3" --label after_relaunch
```

## Twins and acceptance

```bash
export TC3=$(P c3_region_experiment "$RH/study_twin/c3" | tail -1)
export TFG=$(P forge_grid_experiment "$RH/study_twin/forge" | tail -1)
PYTHONPATH="$TC3/source/src" "$PY" "$TC3/source/scripts/run_c3_region_experiment.py" \
  pilot --repo /workspace --experiment "$TC3" --method ccnet5d_grid --candidate a05_b02
PYTHONPATH="$TFG/source/src" "$PY" "$TFG/source/scripts/run_forge_grid_experiment.py" \
  pilot --repo /workspace --experiment "$TFG" --method ccnet5d_grid --candidate a05_b05
export DC3A=$(P c3_region_experiment "$RH/study_determinism_a/c3" | tail -1)
export DC3B=$(P c3_region_experiment "$RH/study_determinism_b/c3" | tail -1)
export DFGA=$(P forge_grid_experiment "$RH/study_determinism_a/forge" | tail -1)
export DFGB=$(P forge_grid_experiment "$RH/study_determinism_b/forge" | tail -1)
for experiment in "$DC3A" "$DC3B"; do
  PYTHONPATH="$experiment/source/src" "$PY" "$experiment/source/scripts/run_c3_region_experiment.py" \
    pilot --repo /workspace --experiment "$experiment" --method ccnet5d_grid --candidate a05_b02
done
for experiment in "$DFGA" "$DFGB"; do
  PYTHONPATH="$experiment/source/src" "$PY" "$experiment/source/scripts/run_forge_grid_experiment.py" \
    pilot --repo /workspace --experiment "$experiment" --method ccnet5d_grid --candidate a05_b05
done
"$PY" scripts/check_study067_rehearsal.py --c3 "$C3" --forge "$FG" \
  --gpu-log "$RH/gpu_samples.jsonl" --twin "$TC3" --twin "$TFG" \
  --determinism-twin "$DC3A" --determinism-twin "$DC3B" \
  --determinism-twin "$DFGA" --determinism-twin "$DFGB" \
  --imports "$IC3" --imports "$IFG" --launch-commit "$LAUNCH_COMMIT" \
  > "$RH/acceptance.txt"
cat "$RH/acceptance.txt"
```

Every acceptance area prints PASS/FAIL with evidence paths; missing evidence is a
failure. Training-loss comparisons are performed internally. Held-out metric
values and prediction contents are never displayed. Target-read counts are
reported per method/seed, including two reads after a post-scoring crash.
The OOM cleanup criterion compares the retried attempt's final thread count with
the failed attempt's **starting** count (the pre-OOM baseline), and requires the
same worker PID and a positive retry peak no greater than 1.10 times the original
peak. This is a leak/cleanup gate: a lower peak passes. In v1, C3 CCNet
`S1/a05_b02` peaked at 17.39 GiB before the injected OOM and 15.36 GiB on its
same-PID retry from step 200 (11.7% lower). With `cudnn_benchmark: true`, that
retry reuses autotuned algorithms and can skip their original workspace peak.
`--plan-only STUDY` is the CPU dry-run check and explicitly does not evaluate GPU
acceptance. A rehearsal passing on synthetic CPU fixtures is not a GPU PASS.
