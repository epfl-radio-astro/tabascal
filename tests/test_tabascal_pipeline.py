"""Test full Tabascal pipeline runs."""

import hashlib
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tabsim
import yaml
from huggingface_hub import snapshot_download


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 hash of a file.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal string representation of the SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@pytest.fixture
def provide_test_data(tmp_path: Path) -> Path:
    """Pytest fixture to download test data from HuggingFace.

    Downloads RFI simulation data from the epfl-radio-astro/rfi-simulations dataset
    using a branch name based on the current tabsim version. Data is cached locally
    by HuggingFace (usually in ~/.cache/huggingface).
    If data cannot be downloaded from HuggingFace, it is generated with tabsim.

    Args:
        tmp_path: Pytest fixture providing temporary directory for test files

    Returns:
        Path to the local directory containing the downloaded test data
    """
    branch = f"tabsim_v{tabsim.__version__}"

    try:
        # Data is cached by HuggingFace, usually in ~/.cache/huggingface
        local_dir = snapshot_download(
            repo_id="epfl-radio-astro/rfi-simulations",
            repo_type="dataset",
            revision=branch,
        )
        return Path(local_dir)
    except Exception as e:
        logging.warning(
            f"Download of pre-generated RFI simulation data failed for tabsim "
            f"version {tabsim.__version__}: {e}. Generating data with tabsim."
        )

        tabsim_script = Path(tabsim.__file__).parent / "scripts" / "sim_vis.py"
        data_dir = Path(__file__).parent / "data"

        for config_file in data_dir.glob("sim_target*.yaml"):
            input_hash = compute_sha256(config_file)
            logging.warning(f"Generating data for {config_file}.")

            result = subprocess.run(
                [
                    sys.executable,
                    str(tabsim_script),
                    "-c",
                    str(config_file),
                    "-sp",
                    str(tmp_path / input_hash),
                ],
                capture_output=True,
                text=True,
                cwd=tmp_path,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Data generation failed for {config_file}: {result.stderr}"
                )

        return tmp_path



def read_and_modify_yaml(
    new_data: dict[str, Any], input_path: Path, output_path: Path
) -> None:
    """Read a YAML file, update it with new data, and write to output path.

    Args:
        new_data: Dictionary of updates to merge into the YAML data
        input_path: Path to the input YAML file to read
        output_path: Path where the modified YAML will be written
    """
    with open(input_path) as f:
        data = yaml.safe_load(f)

    data.update(new_data)

    with open(output_path, "w") as f:
        yaml.dump(data, f)


def test_tabascal_pipeline(provide_test_data: Path, tmp_path: Path) -> None:
    """Test the complete Tabascal pipeline execution.

    This test verifies that the full Tabascal pipeline runs successfully and produces
    expected results. It:
    1. Downloads test simulation data from HuggingFace
    2. Configures the pipeline with specific component modules
    3. Executes the run_tabascal.py script
    4. Validates that the output Reduced Chi^2 value matches the expected result

    Args:
        provide_test_data: Fixture providing path to downloaded test data
        tmp_path: Pytest fixture providing temporary directory for test files
    """
    local_dir = Path(provide_test_data)
    data_dir = Path(__file__).parent / "data"
    input_hash = compute_sha256(data_dir / "sim_target_8A.yaml")

    input_dir = local_dir / input_hash
    config_template = data_dir / "tab_target.yaml"
    config_path = tmp_path / "tab_target.yaml"

    config_mod = {
        "model": {
            "components": [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ]
        }
    }
    read_and_modify_yaml(config_mod, config_template, config_path)

    # Take the first match for "pnt_src*"
    input_src_dir = next((d for d in input_dir.glob("pnt_src*") if d.is_dir()), None)
    assert input_src_dir, f"No pnt_src* directory found in {input_dir}"

    tabascal_script = (
        Path(__file__).parent.parent / "tabascal" / "scripts" / "run_tabascal.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(tabascal_script),
            "-c",
            str(config_path),
            "-s",
            str(input_src_dir),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 0, f"Tabascal failed: {result.stderr}"

    match = re.search(r"Reduced Chi\^2 @ opt params : ([\d.eE+-]+)", result.stdout)
    assert match, f"Could not find Reduced Chi^2 in output: {result.stdout}"

    value = float(match.group(1))
    expected_value = 0.8549507174978265
    assert value == pytest.approx(expected_value, rel=1e-3)
