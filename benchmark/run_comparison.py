"""Run the RFI-signal component comparison for issue #107.

Runs the same simulation through several ``rfi_signal`` components, holding every
other component fixed, and collects the three things the decision rests on: peak
device memory, runtime, and reconstruction accuracy.

``RealRFI`` is deliberately absent: it emits a real-valued ``rfi_A`` and every FFI
RFI-vis kernel requires a complex amplitude, so it cannot be run against the same
``rfi_vis`` component as the others (see issue #107). Running it against the
non-FFI kernel instead would change two variables at once.

Usage
-----
    python run_comparison.py --tab-config tab_64A.yaml --sim-dir <path> \
        --workdir results/
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

# rfi_signal components under test, with the rest of the model held fixed.
# Names are as on `main` (this branch predates the issue #103 renames).
VARIANTS = {
    "ComplexRFI (matrix GP, time only)": "rfi_signal:ComplexRFI",
    "ComplexRFITimeFreq (matrix GP, time+freq)": "rfi_signal:ComplexRFITimeFreq",
    "FourierGPRFI (Fourier, per-ant)": "rfi_signal:FourierGPRFI",
    "FourierGPRFIScan (Fourier, per-ant, scan+remat)": "rfi_signal:FourierGPRFIScan",
    "FourierGPRFIConstAnt (Fourier, shared ant)": "rfi_signal:FourierGPRFIConstAnt",
}

FIXED_COMPONENTS = [
    "trajectory:FixedOrbit",
    None,  # rfi_signal slot
    "rfi_vis:RiemannVisTimeFreqCalculationFFI",
    "ast_vis:FourierTimeFreqGPAst",
    "gains:UnitaryGains",
]


def parse_run(stdout: str) -> dict:
    """Pull memory, runtime, chi2 and truth metrics out of a tabascal run."""
    out = {}

    # "cuda:0  32.798     102.005" from print_memory_usage; both columns are
    # fixed-point, which distinguishes it from the device table printed earlier.
    if m := re.search(r"^\s*(cuda:\d+|cpu:\d+)\s+([\d.]+)\s+([\d.]+)\s*$", stdout, re.M):
        out["peak_GB"] = float(m.group(2))

    if m := re.search(
        r"^tabascal_subtraction\s+\d+\s+[\d.]+\s+\S+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)\s+s\s",
        stdout, re.M,
    ):
        out["total_s"] = float(m.group(1))
    if m := re.search(
        r"^\s{2}run_opt\s+\d+\s+[\d.]+\s+\S+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)\s+s\s",
        stdout, re.M,
    ):
        out["opt_s"] = float(m.group(1))

    for point in ("init", "opt"):
        if m := re.search(
            rf"Reduced Chi\^2 @ {point} params : ([\d.eE+-]+)", stdout
        ):
            out[f"chi2_{point}"] = float(m.group(1))

    # Truth metrics block: two rows per quantity, the label only on the first.
    if (start := stdout.find("Truth metrics @ opt params:")) != -1:
        current = None
        labels = {"Ast. Vis": "ast", "RFI Vis": "rfi", "Gains": "gains"}
        for line in stdout[start:].splitlines()[1:]:
            label, sep, rest = line.partition("|")
            if not sep:
                break
            if lbl := label.strip():
                current = labels.get(lbl)
                if current is None:
                    break
            rest = rest.strip()
            if rest.startswith("RMSE") and current:
                if m := re.search(r"/noise\s+([-\d.eE+]+)", rest):
                    out[f"{current}_NRMSE_noise"] = float(m.group(1))
                if m := re.search(r"RMSE\s+([-\d.eE+]+)", rest):
                    out[f"{current}_RMSE"] = float(m.group(1))
            elif rest.startswith("bias") and current:
                if m := re.search(r"\[\s*([-\d.eE+]+)\s*sigma", rest):
                    out[f"{current}_bias_sigma"] = float(m.group(1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-config", required=True)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--workdir", default="results")
    parser.add_argument("--tle-dir", default=None, help="--extra-orbit-dir for tabascal")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--precision", choices=["single", "double"], default=None,
        help="Override model.precision. Single halves every RFI array, which is the "
             "main lever on the memory ceiling these components run into.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Class names to run (e.g. FourierGPRFIScan). Defaults to all. Memory is "
             "deterministic for a fixed simulation, so re-running variants already "
             "measured on the same sim just burns GPU time.",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve()
    base = yaml.safe_load(Path(args.tab_config).read_text())
    ids_path = Path(base["satellites"]["norad_ids_path"]).resolve()

    variants = VARIANTS
    if args.variants:
        variants = {
            k: v for k, v in VARIANTS.items() if v.split(":")[1] in args.variants
        }
        unknown = set(args.variants) - {v.split(":")[1] for v in VARIANTS.values()}
        if unknown:
            raise SystemExit(f"Unknown variant(s): {sorted(unknown)}")

    results = {}
    for label, component in variants.items():
        slug = component.split(":")[1]
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)

        config = dict(base)
        components = [c if c else component for c in FIXED_COMPONENTS]
        config["model"] = {**base["model"], "components": components}
        config["satellites"] = {**base["satellites"], "norad_ids_path": str(ids_path)}
        if args.max_iter is not None:
            config["opt"] = {**base["opt"], "max_iter": args.max_iter}
        if args.precision is not None:
            config["model"] = {**config["model"], "precision": args.precision}

        run_dir = workdir / slug
        run_dir.mkdir(exist_ok=True)
        config_path = run_dir / "tab.yaml"
        config_path.write_text(yaml.safe_dump(config))

        cmd = [
            sys.executable,
            str(repo_root / "tabascal" / "scripts" / "run_tabascal.py"),
            "run", "-c", str(config_path), "-s", args.sim_dir, "-t",
        ]
        if args.tle_dir:
            cmd += ["--extra-orbit-dir", args.tle_dir]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir)
        (run_dir / "stdout.txt").write_text(proc.stdout)
        (run_dir / "stderr.txt").write_text(proc.stderr)

        if proc.returncode != 0:
            print(f"FAILED (rc={proc.returncode})")
            print(proc.stderr[-2000:])
            results[label] = {"failed": True, "returncode": proc.returncode}
            continue

        parsed = parse_run(proc.stdout)
        results[label] = parsed
        print(json.dumps(parsed, indent=2))

    (workdir / "results.json").write_text(json.dumps(results, indent=2))

    # Summary table
    cols = [
        ("peak_GB", "peak GB"), ("total_s", "total s"), ("opt_s", "opt s"),
        ("chi2_init", "chi2 init"), ("chi2_opt", "chi2 opt"),
        ("ast_NRMSE_noise", "ast NRMSE"),
        ("rfi_NRMSE_noise", "rfi NRMSE"), ("rfi_bias_sigma", "rfi bias"),
    ]
    print(f"\n\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    print(f"{'component':<44}" + "".join(f"{h:>12}" for _, h in cols))
    for label, res in results.items():
        if res.get("failed"):
            print(f"{label:<44}{'FAILED':>12}")
            continue
        row = "".join(
            f"{res[k]:>12.4g}" if k in res else f"{'-':>12}" for k, _ in cols
        )
        print(f"{label:<44}{row}")
    print(f"\nResults written to {workdir / 'results.json'}")


if __name__ == "__main__":
    main()
