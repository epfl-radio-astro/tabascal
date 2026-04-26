"""Test full Tabascal pipeline runs."""

import hashlib
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest
import tabsim
import yaml
from huggingface_hub import snapshot_download


def _has_spacetrack_credentials() -> bool:
    try:
        from tabascal.tle import load_spacetrack_credentials
        user, passwd = load_spacetrack_credentials()
        return user is not None and passwd is not None
    except Exception:
        return False

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



@dataclass
class PipelineTestConfig:
    """Configuration for a single pipeline test case.

    When chi2_ref is None the test only checks that chi2 lies in (0, 5).
    Once a reference value is known, set chi2_ref to pin the regression.
    A pre-commit hook (ci/check_chi2_refs.py) enforces that chi2_ref is
    populated before the config is committed.
    """
    sim_file_name: str
    components: list[str]
    config_overrides: dict = field(default_factory=dict)
    chi2_ref: Optional[float] = None


def _run_pipeline(
    provide_test_data: Path,
    tmp_path: Path,
    t_config: PipelineTestConfig,
) -> tuple[int, str, str]:
    local_dir = Path(provide_test_data)
    data_dir = Path(__file__).parent / "data"
    input_hash = compute_sha256(data_dir / t_config.sim_file_name)

    input_dir = local_dir / input_hash
    config_template = data_dir / "tab_target.yaml"
    config_path = tmp_path / "tab_target.yaml"

    config_mod: dict = {"model": {"components": t_config.components}}
    config_mod.update(t_config.config_overrides)
    read_and_modify_yaml(config_mod, config_template, config_path)

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
    return result.returncode, result.stdout, result.stderr


def _assert_chi2(stdout: str, chi2_ref: Optional[float]) -> None:
    match = re.search(r"Reduced Chi\^2 @ opt params : ([\d.eE+-]+)", stdout)
    assert match, f"Could not find Reduced Chi^2 in output: {stdout}"
    value = float(match.group(1))
    if chi2_ref is not None:
        assert value == pytest.approx(chi2_ref, rel=1e-2)
    else:
        print(f"\nReduced Chi^2 @ opt params : {value}")
        assert 0.0 < value < 5.0, f"Chi^2 = {value} is outside the expected (0, 5) range"


# ---------------------------------------------------------------------------
# Trajectory components — downstream fixed to RiemannVisTimeFreqCalculation + UnitaryGains
# ---------------------------------------------------------------------------

trajectory_configs = [
    # FixedOrbit without PhaseCalculationRFI covered by RiemannVsisTimeFreqCalculation
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            chi2_ref=0.8977856043436502,
        ),
        id="FixedOrbit+PhaseCalculationRFI",
    ),
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:SGP4LEONoDragOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            config_overrides={"opt": {"max_iter": 200}},
            chi2_ref=0.8834671325695459, # Run with MEO satellites
        ),
        id="SGP4LEONoDragOrbit+PhaseCalculationRFI",
    ),
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:SGP4LEOOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            config_overrides={"opt": {"max_iter": 200}},
            chi2_ref=0.8834671467969134,  # Run with MEO satellites
        ),
        id="SGP4LEOOrbit+PhaseCalculationRFI",
    ),
]


# ---------------------------------------------------------------------------
# RFI signal components — upstream fixed to FixedOrbit
# ---------------------------------------------------------------------------

rfi_signal_configs = []


# ---------------------------------------------------------------------------
# RFI visibility components — upstream fixed to FixedOrbit
# ---------------------------------------------------------------------------

rfi_vis_configs = [
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            chi2_ref=0.8977856059138833,
        ),
        id="RiemannVisTimeFreqCalculation",
    ),
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculationFFI",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            chi2_ref=0.8977856059138833,
        ),
        id="RiemannVisTimeFreqCalculationFFI",
    ),
]


# ---------------------------------------------------------------------------
# Astronomical sky signal components — upstream fixed to FixedOrbit
# ---------------------------------------------------------------------------

ast_signal_configs = []

# ---------------------------------------------------------------------------
# Astronomical visibility components — upstream fixed to FixedOrbit
# ---------------------------------------------------------------------------

ast_vis_configs = []


# ---------------------------------------------------------------------------
# Gains components — upstream fixed to FixedOrbit
# ---------------------------------------------------------------------------

gains_configs = [
    # UnitaryGains covered by RiemannVisTimeFreqCalculation
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:GPGains",
            ],
            config_overrides={
                "gains": {
                    "amp_mean": 1.0,
                    "amp_std": 1.0,
                    "phase_mean": 0.0,
                    "phase_std": 1.0,
                    "amp_corr_time": None,
                    "phase_corr_time": None,
                    "amp_corr_freq": None,
                    "phase_corr_freq": None,
                    "r_seed": 123,
                },
            },
            chi2_ref=0.8977832575028029,
        ),
        id="GPGains",
    ),
]


# ---------------------------------------------------------------------------
# All pipeline tests — single parametrized function
# ---------------------------------------------------------------------------

all_configs = trajectory_configs + rfi_signal_configs + rfi_vis_configs + ast_signal_configs + ast_vis_configs + gains_configs


@pytest.mark.parametrize("t_config", all_configs)
def test_pipeline(provide_test_data: Path, tmp_path: Path, t_config) -> None:
    """Parametrized integration test covering all component pipeline combinations."""
    returncode, stdout, stderr = _run_pipeline(provide_test_data, tmp_path, t_config)
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_chi2(stdout, t_config.chi2_ref)
