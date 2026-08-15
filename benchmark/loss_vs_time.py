"""Compare RFI-signal components by loss against wall-clock time (issue #107).

The CSCS perf check regressed when #103 swapped the RFI-signal component from
the real-space ``ComplexRFI`` to the Fourier ``FourierGPRFI`` (renamed
``ComplexRFIVarAnt``). Optimiser runtime is the whole of that regression, so
the question is not whether the Fourier component costs more per iteration --
it does -- but whether it reaches a given loss sooner in wall-clock terms.

Loss-per-iteration cannot answer that: it hides the cost of an iteration, and
so flatters whichever model takes the more expensive step. This script records
loss against elapsed time for each component and plots them on a shared axis.

The runs are strictly sequential. Running them concurrently would have them
compete for the same cores and make the timings meaningless, which is the one
thing this script exists to measure.

Usage
-----
    python benchmark/loss_vs_time.py --workdir out/ --max-iter 1000
    python benchmark/loss_vs_time.py --workdir out/ --plot-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]

# Component names as on `main`, which still has both models. After #103 these
# are RealRFIVarAnt / ComplexRFIVarAnt / ComplexRFIConstAnt.
VARIANTS = {
    "matrix": ("rfi_signal:ComplexRFI", "matrix GP (ComplexRFI)"),
    "fourier": ("rfi_signal:FourierGPRFI", "Fourier, scan+remat (FourierGPRFI)"),
}

# Held fixed across variants: only rfi_signal changes. This is the FFI vis
# kernel, which is where the regression is largest -- it removed the RFI-vis
# cost, leaving the signal component as the dominant term.
FIXED = [
    "trajectory:FixedOrbit",
    None,  # rfi_signal slot
    "rfi_vis:RiemannVisTimeFreqCalculationFFI",
    "ast_vis:FourierTimeFreqGPAst",
    "gains:UnitaryGains",
]


def components_for(spec: str) -> str:
    parts = [spec if p is None else p for p in FIXED]
    return ",".join(parts)


def prepare(workdir: Path, spec: str, precision: str, max_iter: int) -> Path:
    wd = workdir / f"wd_{spec.split(':')[1]}"
    subprocess.run(
        [
            sys.executable,
            str(SRC_ROOT / "ci" / "reframe" / "prepare_data.py"),
            "--components", components_for(spec),
            "--workdir", str(wd),
            "--src-root", str(SRC_ROOT),
            "--precision", precision,
            "--max-iter", str(max_iter),
        ],
        check=True,
    )
    return wd


def run(wd: Path, trace_path: Path) -> str:
    sim_dir = (wd / "sim_dir.txt").read_text().strip()
    env = {**os.environ, "TAB_LOSS_TRACE": str(trace_path)}
    proc = subprocess.run(
        [
            sys.executable,
            str(SRC_ROOT / "tabascal" / "scripts" / "run_tabascal.py"),
            "run",
            "-c", str(wd / "tab_target.yaml"),
            "-s", sim_dir,
            "--extra-orbit-dir", str(SRC_ROOT / "tabascal" / "data" / "tles"),
            "-t",
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError(f"run failed for {wd}")
    return proc.stdout


def parse_chi2(stdout: str) -> dict:
    out = {}
    for point in ("init", "opt"):
        m = re.search(rf"Reduced Chi\^2 @ {point} params : ([\d.eE+-]+)", stdout)
        if m:
            out[point] = float(m.group(1))
    return out


def plot(workdir: Path, meta: dict, precision: str, max_iter: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colours = {"matrix": "#B45309", "fourier": "#1D4ED8"}

    for key, (_, label) in VARIANTS.items():
        d = np.load(workdir / f"trace_{key}.npz")
        loss, t = d["loss"], d["time_s"]
        axes[0].plot(np.arange(1, len(loss) + 1), loss, color=colours[key], label=label)
        axes[1].plot(t, loss, color=colours[key], label=label)

    axes[0].set_xlabel("Iteration")
    axes[0].set_title("By iteration — hides cost per step")
    axes[1].set_xlabel("Wall-clock time (s)")
    axes[1].set_title("By wall-clock time — the honest axis")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Loss (neg. log posterior / obs size)")

    n_iter = 2 * max_iter
    fig.suptitle(
        f"RFI-signal component convergence — {meta['workload']}, "
        f"{precision} precision, {n_iter} iterations, CPU",
        fontsize=12,
    )
    fig.tight_layout()
    out = workdir / "loss_vs_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    meta_path = args.workdir / "meta.json"

    if args.plot_only:
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"workload": "96 ant / 90 times / 1 chan / 32 sat", "runs": {}}
        for key, (spec, label) in VARIANTS.items():
            print(f"=== {label} ===", flush=True)
            wd = prepare(args.workdir, spec, args.precision, args.max_iter)
            stdout = run(wd, args.workdir / f"trace_{key}.npz")
            (args.workdir / f"stdout_{key}.txt").write_text(stdout)
            meta["runs"][key] = {"label": label, "chi2": parse_chi2(stdout)}
            print(meta["runs"][key], flush=True)
        meta_path.write_text(json.dumps(meta, indent=2))

    out = plot(args.workdir, meta, args.precision, args.max_iter)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
