"""Can the matrix GP represent the RFI if given a shorter corr_time? (issue #107)

On the perf workload the matrix GP starts with an RFI-visibility error 2.2x
worse than the Fourier component (RMSE/noise 1.645 vs 0.733), *at
initialisation*. Since the config initialises from truth, that is a statement
about what each basis can represent, not about optimiser progress: the matrix
GP places inducing points at ``corr_time`` spacing, so if that spacing is too
coarse for the RFI fringe the basis cannot express the signal however long it
is optimised.

This sweeps ``rfi.corr_time`` for the matrix GP and records what it buys
(representation error) against what it costs (inducing points -> Cholesky and
per-iteration time). The Fourier component is run once at the baseline for
reference. Short runs: the question is answered at init, and a few iterations
only confirm the starting point is not a fluke.

Usage
-----
    python benchmark/corr_time_sweep.py --workdir out/ --corr-times 24 12 6 3
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]

FIXED = [
    "trajectory:FixedOrbit",
    None,
    "rfi_vis:RiemannVisTimeFreqCalculationFFI",
    "ast_vis:FourierTimeFreqGPAst",
    "gains:UnitaryGains",
]


def components_for(spec: str) -> list:
    return [spec if p is None else p for p in FIXED]


def parse(stdout: str) -> dict:
    out = {}
    for point in ("init", "opt"):
        if m := re.search(rf"Reduced Chi\^2 @ {point} params : ([\d.eE+-]+)", stdout):
            out[f"chi2_{point}"] = float(m.group(1))

    # Truth metric rows, e.g.
    #   RFI Vis  | RMSE  4.778e-01 Jy  /noise 1.645e+00  /signal 1.373e-01
    for point in ("init", "opt"):
        start = stdout.find(f"Truth metrics @ {point} params:")
        if start == -1:
            continue
        block = stdout[start : start + 800]
        for label, key in (("Ast. Vis", "ast"), ("RFI Vis", "rfi")):
            m = re.search(
                rf"{re.escape(label)}\s*\|\s*RMSE\s+[\d.eE+-]+ Jy\s+/noise ([\d.eE+-]+)",
                block,
            )
            if m:
                out[f"{key}_nrmse_{point}"] = float(m.group(1))

    # Both rfi_signal:FourierGPRFI and ast_vis:FourierTimeFreqGPAst print
    # "(n_k_fq, n_k_tm)", so a single regex cannot say which is which. Capture
    # all of them; only the RFI one varies with rfi.corr_time, which is what
    # identifies it once the sweep is done.
    out["k_grids"] = [
        (int(a), int(b))
        for a, b in re.findall(r"\(n_k_fq, n_k_tm\): \((\d+), (\d+)\)", stdout)
    ]
    if m := re.search(r"^\s{2}run_opt\s+\d+\s+[\d.]+\s+\S+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)\s+([numkM]?)s\s",
                      stdout, re.M):
        scale = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3}[m.group(2)]
        out["opt_s"] = float(m.group(1)) * scale
    return out


def run_case(workdir: Path, base_cfg: Path, sim_dir: str, spec: str,
             corr_time: float, max_iter: int, tag: str) -> dict:
    wd = workdir / tag
    wd.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(base_cfg.read_text())
    cfg["model"]["components"] = components_for(spec)
    cfg["rfi"]["corr_time"] = corr_time
    cfg["opt"]["max_iter"] = max_iter
    (wd / "tab_target.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [sys.executable, str(SRC_ROOT / "tabascal" / "scripts" / "run_tabascal.py"),
         "run", "-c", str(wd / "tab_target.yaml"), "-s", sim_dir,
         "--extra-orbit-dir", str(SRC_ROOT / "tabascal" / "data" / "tles"), "-t"],
        capture_output=True, text=True, check=False,
    )
    (wd / "stdout.txt").write_text(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise RuntimeError(f"run failed: {tag}")

    res = parse(proc.stdout)
    res.update(component=spec, corr_time=corr_time)
    if spec.endswith("ComplexRFI"):
        # Inducing times are exactly what get_times selects at this spacing.
        from tabascal.gp import get_times
        import xarray as xr
        res["basis"] = int(len(get_times(_times(sim_dir), corr_time)))
    return res


def _times(sim_dir: str):
    import xarray as xr
    import jax.numpy as jnp
    xds = xr.open_zarr(Path(sim_dir) / f"{Path(sim_dir).name}.zarr")
    return jnp.asarray(xds.time.data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--corr-times", type=float, nargs="+", default=[24, 12, 6, 3])
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--components", default="both",
                   choices=["both", "matrix", "fourier"])
    args = p.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Stage the perf workload once; every case reuses it.
    stage = args.workdir / "stage"
    subprocess.run(
        [sys.executable, str(SRC_ROOT / "ci" / "reframe" / "prepare_data.py"),
         "--components", ",".join(components_for("rfi_signal:ComplexRFI")),
         "--workdir", str(stage), "--src-root", str(SRC_ROOT),
         "--precision", args.precision, "--max-iter", str(args.max_iter)],
        check=True,
    )
    base_cfg = stage / "tab_target.yaml"
    sim_dir = (stage / "sim_dir.txt").read_text().strip()
    baseline_corr = yaml.safe_load(base_cfg.read_text())["rfi"]["corr_time"]

    specs = {"matrix": "rfi_signal:ComplexRFI", "fourier": "rfi_signal:FourierGPRFI"}
    if args.components != "both":
        specs = {args.components: specs[args.components]}

    results = []
    for name, spec in specs.items():
        for ct in args.corr_times:
            print(f"=== {name}, corr_time={ct:g} s ===", flush=True)
            results.append(run_case(args.workdir, base_cfg, sim_dir, spec, ct,
                                    args.max_iter, f"{name}_ct{ct:g}"))
            print(results[-1], flush=True)

    (args.workdir / "results.json").write_text(json.dumps(results, indent=2))

    hdr = f"{'component':14s} {'corr_t':>7s} {'basis':>7s} {'k_grids':>16s} {'chi2_init':>10s} {'rfi_nrmse_init':>15s} {'ast_nrmse_init':>15s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        name = r["component"].split(":")[1]
        print(f"{name:14s} {r['corr_time']:7.1f} {str(r.get('basis','-')):>7} "
              f"{str(r.get('k_grids','-')):>16s} "
              f"{r.get('chi2_init',float('nan')):10.4f} "
              f"{r.get('rfi_nrmse_init',float('nan')):15.4f} "
              f"{r.get('ast_nrmse_init',float('nan')):15.5f}")


if __name__ == "__main__":
    main()
