"""Prepare test data and config for ReFrame tabascal performance check."""

import argparse
import hashlib
import logging
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
        "--src-root",
        default="/tabascal/src",
        help="Root of the tabascal source tree (default: /tabascal/src)",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    src_root = Path(args.src_root)
    data_dir = src_root / "tests" / "data"
    sim_config = data_dir / "sim_target_8A.yaml"
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
        local_dir = workdir / "generated_data"
        local_dir.mkdir(exist_ok=True)
        tabsim_script = Path(tabsim.__file__).parent / "scripts" / "sim_vis.py"
        subprocess.run(
            [
                sys.executable,
                str(tabsim_script),
                "-c",
                str(sim_config),
                "-sp",
                str(local_dir / input_hash),
            ],
            cwd=str(local_dir),
            check=True,
        )

    # Step 2: Locate pnt_src directory
    input_dir = local_dir / input_hash
    sim_dir = next((d for d in input_dir.glob("pnt_src*") if d.is_dir()), None)
    if sim_dir is None:
        raise RuntimeError(f"No pnt_src* directory found in {input_dir}")

    (workdir / "sim_dir.txt").write_text(str(sim_dir))

    # Step 3: Create modified config with requested components
    components = args.components.split(",")
    with open(tab_template) as f:
        config = yaml.safe_load(f)
    config["model"] = {"components": components}
    config_out = workdir / "tab_target.yaml"
    with open(config_out, "w") as f:
        yaml.dump(config, f)

    print(f"Data prepared. sim_dir={sim_dir}, config={config_out}")


if __name__ == "__main__":
    main()
