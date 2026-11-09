"""Prepare test data and config for ReFrame tabascal performance check."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tabsim
import yaml
from huggingface_hub import snapshot_download


def _load_ci_module(path: Path):
    """Import a CI script by path; ci/ is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The publisher's key, so the data looked up is the data it published.
compute_sha256 = _load_ci_module(Path(__file__).resolve().parents[1] / "rfi_simulations.py").compute_sha256


def simulation_gpu(environ) -> str:
    """The one GPU tabsim may simulate on: the first visible, GPU 0 if none is named.

    tabsim refuses more than one visible GPU and the multi-GPU check exposes them
    all. An empty CUDA_VISIBLE_DEVICES hides every GPU and is passed on unchanged.
    """
    return environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def main():
    parser = argparse.ArgumentParser(
        description="Download/generate test data and prepare config for ReFrame check."
    )
    parser.add_argument(
        "--components",
        required=True,
        help="Comma-separated component specs for model.components",
    )
    parser.add_argument(
        "--rfi-backend",
        default=None,
        help="rfi.backend of the rfi_vis component: jax or ffi (default: the component's)",
    )
    parser.add_argument(
        "--workdir", required=True, help="Working directory for output files"
    )
    parser.add_argument(
        "--precision",
        default="double",
        choices=["single", "double"],
        help="Model floating point precision (default: double)",
    )
    parser.add_argument(
        "--src-root",
        default=".",
        help="Root of the tabascal source tree (default: .)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Value for opt.max_iter in the generated config (default: 1000)",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    src_root = Path(args.src_root).resolve()
    data_dir = src_root / "ci" / "reframe" / "data"
    sim_config = data_dir / "sim_target_96A.yaml"
    tab_template = data_dir / "tab_target.yaml"

    # Step 1: Download or generate simulation data
    branch = f"tabsim_v{tabsim.__version__}"
    input_hash = compute_sha256(sim_config)

    try:
        local_dir = Path(
            snapshot_download(
                repo_id="epfl-radio-astro/rfi-simulations",
                repo_type="dataset",
                revision=branch,
            )
        )
    except Exception as e:
        logging.warning(f"HuggingFace download failed: {e}. Generating with tabsim.")
        local_dir = None

    # Step 2: Check for cached data or generate locally
    input_dir = local_dir / input_hash if local_dir is not None else None
    if input_dir is None or not input_dir.is_dir():
        if local_dir is not None:
            logging.warning(
                f"Hash directory {input_hash} not found in HF snapshot. "
                "Generating data with tabsim."
            )
        # Per tabsim release, like the snapshot it stands in for.
        local_dir = workdir / "generated_data" / branch
        local_dir.mkdir(parents=True, exist_ok=True)
        input_dir = local_dir / input_hash
        if input_dir.is_dir():
            logging.info(
                f"Simulation output {input_dir} already exists. Skipping generation."
            )
        else:
            # Copy ancillary files referenced by relative paths in sim_config
            # into the working directory so sim_vis.py can find them.
            for ancillary in data_dir.glob("*"):
                if ancillary.is_file() and ancillary.suffix != ".yaml":
                    shutil.copy2(ancillary, local_dir / ancillary.name)
            tabsim_script = Path(tabsim.__file__).parent / "scripts" / "sim_vis.py"
            subprocess.run(
                [
                    sys.executable,
                    str(tabsim_script),
                    "-c",
                    str(sim_config),
                    "-sp",
                    str(input_dir),
                    "--gpu-id",
                    simulation_gpu(os.environ),
                ],
                cwd=str(local_dir),
                check=True,
            )

    sim_dir = next((d for d in input_dir.glob("pnt_src*") if d.is_dir()), None)
    if sim_dir is None:
        raise RuntimeError(f"No pnt_src* directory found in {input_dir}")

    # The run writes its results into the MS, so give it a copy under workdir
    # rather than the HuggingFace cache, whose blobs are read-only. copyfile
    # drops the 0444 mode bits; the basename is kept because the run derives
    # its MS/zarr paths from it. -od still receives a tab-sim simulation
    # directory: without an ms_path it is read as one, as the perf job expects.
    # Staged under the tabsim release and the config hash too, since the basename
    # does not change with either, so a reused workdir picks up new data.
    staged = workdir / "sim" / branch / input_hash / sim_dir.name
    if not staged.is_dir():
        shutil.copytree(sim_dir, staged, copy_function=shutil.copyfile)
    sim_dir = staged
    (workdir / "sim_dir.txt").write_text(str(sim_dir))

    # Step 3: Create modified config with requested components
    # Use tabascal's loader so bare scientific-notation floats parse as floats.
    from tabascal.config import yaml_load

    components = args.components.split(",")
    config = yaml_load(tab_template)
    config["model"] = {"components": components, "precision": args.precision}
    config.setdefault("opt", {})["max_iter"] = args.max_iter
    if args.rfi_backend is not None:
        config.setdefault("rfi", {})["backend"] = args.rfi_backend
    config_out = workdir / "tab_target.yaml"
    with open(config_out, "w") as f:
        yaml.dump(config, f)

    print(f"Data prepared. sim_dir={sim_dir}, config={config_out}")



if __name__ == "__main__":
    main()
