#!/bin/bash
#SBATCH -A c39
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -J baseline-candidate
#SBATCH -o baseline_candidate_%j.out
#SBATCH -e baseline_candidate_%j.err
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=default
set -eu
TAB=$HOME/pasc/chris/baseline-audit-candidate
RIK=$HOME/pasc/chris/ri-kernels-shard
ENV=$HOME/pasc/chris/poly-interp-ref/tabascal/.pixi/envs/cuda12-dev
PYTHON=$ENV/bin/python
BENCH=$HOME/pasc/chris/benchmark-tabascal/gp-fine-interp-bench
OUT=$HOME/pasc/chris/poly-interp-hyb
export PATH=$ENV/bin:$PATH PYTHONPATH=$RIK:$TAB PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$SCRATCH/gp-fine-interp-cache/tmp
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$OUT/runs"
cd "$TAB"
# The C++ sources are unchanged by the seam/placement fixes. Reuse the installed
# binaries, record their identities, and require both analytic and quadrature
# GPU entry points before running either axis against this exact build.
$PYTHON - <<'PY'
import hashlib
import jax
from ri_kernels.jax_api import eval_with_indices, analytic_eval_with_indices
from ri_kernels.jax_api.rfi_analytic_vis_op import _TAB_LIB_ANALYTIC_GPU
from ri_kernels.jax_api.rfi_interp_vis_op import _TAB_LIB_INTERP_GPU
assert len(jax.devices()) == 4
for lib in (_TAB_LIB_ANALYTIC_GPU, _TAB_LIB_INTERP_GPU):
    assert lib is not None
    print('KERNELS_OK', lib._name, hashlib.sha256(open(lib._name, 'rb').read()).hexdigest())
PY
# conftest defaults CUDA_VISIBLE_DEVICES to 0 when it is unset. Set all four
# explicitly, and reject a vacuous green run in which every case skipped.
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m pytest -q --no-cov \
  --junitxml="$OUT/gpu_equivalence_${SLURM_JOB_ID}.xml" \
  tests/components/test_baseline_shard_equivalence.py tests/components/test_ast_baseline_sharding.py
$PYTHON - "$OUT/gpu_equivalence_${SLURM_JOB_ID}.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
suites = ET.parse(sys.argv[1]).getroot().iter('testsuite')
counts = {key: 0 for key in ('tests', 'skipped', 'failures', 'errors')}
for suite in suites:
    for key in counts:
        counts[key] += int(suite.get(key, 0))
assert counts['tests'] >= 21 and not any(counts[k] for k in ('skipped', 'failures', 'errors')), counts
print('GPU_EQUIVALENCE_VERIFIED', counts)
PY
simdir=$(readlink -f "$BENCH/sims/8ch/512A")
ms=$(find "$simdir" -maxdepth 1 -name '*.ms' -print -quit)
for budget in 8192 1024; do
  run="$OUT/runs/candidate_probe_${budget}_${SLURM_JOB_ID}"
  mkdir -p "$run"
  cp "$OUT/tab_hybrid_allanalytic.yaml" "$run/tab.yaml"
  echo "PROBE scratch_mb=$budget run=$run"
  TABASCAL_SHARD_AXIS=baseline RI_KERNELS_INTERP_SCRATCH_MB=$budget timeout 900 \
    $PYTHON -u benchmarks/baseline_sharding/probe.py "$run/tab.yaml" "$ms" "$run" \
    --orbit-dir "$BENCH/tles" > "$run/log.txt" 2>&1 || { tail -40 "$run/log.txt"; exit 1; }
  grep -E '"event": "(compiled_memory|step|memory|done)"' "$run/probe.jsonl"
done
for case in source:8192 baseline:8192 baseline:1024; do
  axis=${case%:*}; budget=${case#*:}
  run="$OUT/runs/candidate_cli_${axis}_${budget}_${SLURM_JOB_ID}"
  mkdir -p "$run"
  sed 's/^  max_iter:.*/  max_iter: 102/' "$OUT/tab_hybrid_allanalytic.yaml" > "$run/tab.yaml"
  echo "BENCHMARK axis=$axis scratch_mb=$budget run=$run"
  TABASCAL_SHARD_AXIS=$axis RI_KERNELS_INTERP_SCRATCH_MB=$budget timeout 1200 \
    $PYTHON -u "$TAB/tabascal/scripts/run_tabascal.py" run -c "$run/tab.yaml" -ms "$ms" \
    -od "$run" -t --no-log --skip-ms-write -np "$BENCH/starlink_selected_ids.txt" \
    --extra-orbit-dir "$BENCH/tles" > "$run/log.txt" 2>&1 || { tail -40 "$run/log.txt"; exit 1; }
  grep -A 8 'Memory usage:' "$run/log.txt"
  grep -E 'run_custom_svi|tabascal_subtraction|Reduced Chi' "$run/log.txt"
  $PYTHON - "$run/log.txt" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
values = re.findall(r'Reduced Chi\^2 @ opt params : ([0-9.eE+-]+)', text)
assert values and float(values[-1]) == 83.42597198486328, values
print('REFERENCE_CHI2_EXACT', values[-1])
PY
done
echo JOB_DONE
