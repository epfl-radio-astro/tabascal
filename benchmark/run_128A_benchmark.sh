#!/bin/bash
# Driver for the issue #107 RFI-signal comparison on a CSCS GH200 node.
#
#   sbatch -A <account> run_128A_benchmark.sh [max_iter]
#
# Stages: select satellites -> generate the simulation -> run each rfi_signal
# variant. Each stage is skipped if its output already exists, so the script can be
# resubmitted after a timeout without redoing the expensive simulation.
#SBATCH --job-name=tab107bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=normal

set -eu

MAX_ITER="${1:-20}"
REPO="${REPO:-$HOME/pasc/chris/issue-103/benchmark-repo}"
WORK="${WORK:-$SCRATCH/issue-107-benchmark}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
ENVNAME="${ENVNAME:-cuda12-dev}"

# shellcheck source=/dev/null
[ -f "$HOME/pasc/chris/issue-103/env.sh" ] && . "$HOME/pasc/chris/issue-103/env.sh"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$WORK"
cd "$WORK"

# Ancillary files tabsim resolves relative to the working directory.
cp -n "$REPO/ci/reframe/data/SKA-Low-512.itrf.txt" . 2>/dev/null || true
cp -n "$REPO/benchmark/sim_128A_zenith.yaml" . 2>/dev/null || true
cp -n "$REPO/benchmark/tab_128A.yaml" . 2>/dev/null || true

echo "########## stage 1: select the 32 highest passes"
if [ ! -f selected_norad_ids.txt ]; then
    $PIXI run --manifest-path "$REPO/pyproject.toml" -e "$ENVNAME" python3 \
        "$REPO/benchmark/select_top_satellites.py" \
        --sim-config sim_128A_zenith.yaml --names starlink -n 32 \
        -o selected_norad_ids.txt
else
    echo "selected_norad_ids.txt exists, skipping"
fi
echo "selected: $(tr '\n' ' ' < selected_norad_ids.txt)"

echo
echo "########## stage 1b: calibrate power_scale to ~1000 Jy on-axis"
# --write-back so the generated simulation cannot disagree with the calibration.
$PIXI run --manifest-path "$REPO/pyproject.toml" -e "$ENVNAME" python3 \
    "$REPO/benchmark/calibrate_power_scale.py" \
    --sim-config sim_128A_zenith.yaml --ids selected_norad_ids.txt \
    --target-jy 1000 --write-back

echo
echo "########## stage 2: generate the simulation"
if [ ! -f sim_dir.txt ]; then
    $PIXI run --manifest-path "$REPO/pyproject.toml" -e "$ENVNAME" python3 \
        -c "import tabsim,os;print(os.path.join(os.path.dirname(tabsim.__file__),'scripts','sim_vis.py'))" \
        > .simvis_path
    $PIXI run --manifest-path "$REPO/pyproject.toml" -e "$ENVNAME" python3 \
        "$(cat .simvis_path)" -c sim_128A_zenith.yaml
    find "$WORK/data" -maxdepth 1 -name 'pnt_src*' -type d | head -1 > sim_dir.txt
else
    echo "sim_dir.txt exists, skipping"
fi
echo "sim_dir: $(cat sim_dir.txt)"

echo
echo "########## stage 3: run the rfi_signal variants (max_iter=$MAX_ITER)"
$PIXI run --manifest-path "$REPO/pyproject.toml" -e "$ENVNAME" python3 \
    "$REPO/benchmark/run_comparison.py" \
    --tab-config tab_128A.yaml \
    --sim-dir "$(cat sim_dir.txt)" \
    --workdir "results_iter${MAX_ITER}" \
    --tle-dir "$REPO/tabascal/data/tles" \
    --max-iter "$MAX_ITER" \
    --repo-root "$REPO"
