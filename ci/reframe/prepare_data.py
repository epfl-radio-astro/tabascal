"""Prepare test data and config for ReFrame tabascal performance check."""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tabsim
import yaml
from huggingface_hub import snapshot_download


def compute_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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

    # Step 0: Create spacetrack_login.yaml from environment variables
    # TLE files are now bundled in tabascal with required code
    # This step can be removed when tabsim also bundles these TLEs
    st_user = os.environ.get("SPACETRACK_LOGIN")
    st_pass = os.environ.get("SPACETRACK_PASSWORD")
    if not st_user or not st_pass:
        raise RuntimeError(
            "SPACETRACK_LOGIN and SPACETRACK_PASSWORD environment variables must be set"
        )
    spacetrack_path = workdir / "spacetrack_login.yaml"
    spacetrack_path.write_text(
        yaml.dump({"username": st_user, "password": st_pass})
    )

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
        local_dir = workdir / "generated_data"
        local_dir.mkdir(exist_ok=True)
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
            shutil.copy2(spacetrack_path, local_dir / spacetrack_path.name)
            tabsim_script = Path(tabsim.__file__).parent / "scripts" / "sim_vis.py"
            subprocess.run(
                [
                    sys.executable,
                    str(tabsim_script),
                    "-c",
                    str(sim_config),
                    "-sp",
                    str(input_dir),
                ],
                cwd=str(local_dir),
                check=True,
            )

    sim_dir = next((d for d in input_dir.glob("pnt_src*") if d.is_dir()), None)
    if sim_dir is None:
        raise RuntimeError(f"No pnt_src* directory found in {input_dir}")

    (workdir / "sim_dir.txt").write_text(str(sim_dir))

    # Step 3: Create modified config with requested components
    # Use tabascal's loader so bare scientific-notation floats parse as floats.
    from tabascal.config import yaml_load

    components = args.components.split(",")
    config = yaml_load(tab_template)
    config["model"] = {"components": components, "precision": args.precision}
    config.setdefault("opt", {})["max_iter"] = args.max_iter
    config_out = workdir / "tab_target.yaml"
    with open(config_out, "w") as f:
        yaml.dump(config, f)

    print(f"Data prepared. sim_dir={sim_dir}, config={config_out}")



if __name__ == "__main__":
    main()
