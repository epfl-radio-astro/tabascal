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

    Attributes:
        sim_file_name: Name of the simulation YAML configuration file
        components: List of component module specifications for the pipeline
        chi2_ref: Expected reduced chi-squared value for validation
    """
    sim_file_name: str
    components: list[str]
    chi2_ref: float



test_configs = [
    pytest.param(
        PipelineTestConfig(
        "sim_target_8A.yaml",
        [
            "trajectory:FixedOrbit",
            "rfi_signal:ComplexRFI",
            "rfi_vis:RiemannVisTimeFreqCalculation",
            "ast_vis:FourierTimeFreqGPAst",
            "gains:UnitaryGains",
        ], 0.8549507174978265),
        id="RiemannVisTimeFreqCalculation"
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
        ], 0.8549507174978265),
        id="RiemannVisTimeFreqCalculationFFI"
    )
]

@pytest.mark.parametrize("t_config", test_configs)
def test_tabascal_pipeline(provide_test_data: Path, tmp_path: Path, t_config) -> None:
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
    input_hash = compute_sha256(data_dir / t_config.sim_file_name)

    input_dir = local_dir / input_hash
    config_template = data_dir / "tab_target.yaml"
    config_path = tmp_path / "tab_target.yaml"

    config_mod = {
        "model": {
            "components":  t_config.components       }
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
    expected_value = t_config.chi2_ref
    assert value == pytest.approx(expected_value, rel=1e-2)


# ---------------------------------------------------------------------------
# New-component integration tests
# Uses a separate config type so chi2_ref can be left unset until a reference
# value has been established from a first successful run.
# ---------------------------------------------------------------------------

@dataclass
class NewComponentPipelineTestConfig:
    """Like PipelineTestConfig but with an optional chi2_ref.

    When chi2_ref is None the test only checks that the value lies in (0, 5).
    Once a reference value is known, set chi2_ref to pin the regression.
    """
    sim_file_name: str
    components: list[str]
    config_overrides: dict = field(default_factory=dict)
    chi2_ref: Optional[float] = None


def _run_new_component_pipeline(
    provide_test_data: Path,
    tmp_path: Path,
    t_config: NewComponentPipelineTestConfig,
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


def _assert_new_chi2(stdout: str, chi2_ref: Optional[float]) -> None:
    match = re.search(r"Reduced Chi\^2 @ opt params : ([\d.eE+-]+)", stdout)
    assert match, f"Could not find Reduced Chi^2 in output: {stdout}"
    value = float(match.group(1))
    if chi2_ref is not None:
        assert value == pytest.approx(chi2_ref, rel=1e-2)
    else:
        assert 0.0 < value < 5.0, f"Chi^2 = {value} is outside the expected (0, 5) range"


# GPGains — no extra Space-Track calls needed beyond what TabConfig already does
gpgains_configs = [
    pytest.param(
        NewComponentPipelineTestConfig(
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
                "opt": {"max_iter": 50, "dual_run": False},
            },
            # chi2_ref=...,  # set after first successful run
        ),
        id="GPGains",
    ),
]


@pytest.mark.parametrize("t_config", gpgains_configs)
def test_gpgains_pipeline(provide_test_data: Path, tmp_path: Path, t_config) -> None:
    """Integration test for GPGains replacing UnitaryGains in the standard pipeline.

    Verifies the pipeline runs without error and emits a reasonable chi-squared.
    Update chi2_ref in gpgains_configs once a reference value is established.
    """
    returncode, stdout, stderr = _run_new_component_pipeline(
        provide_test_data, tmp_path, t_config
    )
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_new_chi2(stdout, t_config.chi2_ref)


# FixedOrbit + PhaseCalculationRFI — no Space-Track credentials required.
# FixedOrbit writes rfi_xyz into state; PhaseCalculationRFI recomputes rfi_phase
# from that xyz. This exercises PhaseCalculationRFI at the pipeline level without
# needing SGP4LEONoDragOrbit.
phase_calc_configs = [
    pytest.param(
        NewComponentPipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            config_overrides={"opt": {"max_iter": 50, "dual_run": False}},
        ),
        id="FixedOrbit+PhaseCalculationRFI",
    ),
]


@pytest.mark.parametrize("t_config", phase_calc_configs)
def test_phase_calculation_rfi_pipeline(provide_test_data: Path, tmp_path: Path, t_config) -> None:
    """Integration test for PhaseCalculationRFI in the full pipeline.

    Uses FixedOrbit to supply rfi_xyz (no Space-Track needed) then
    PhaseCalculationRFI to recompute rfi_phase from that xyz, exercising the
    component at the highest level without a Space-Track dependency.
    """
    returncode, stdout, stderr = _run_new_component_pipeline(
        provide_test_data, tmp_path, t_config
    )
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_new_chi2(stdout, t_config.chi2_ref)


# SGP4LEONoDragOrbit + PhaseCalculationRFI — requires Space-Track credentials
sgp4_configs = [
    pytest.param(
        NewComponentPipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:SGP4LEONoDragOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            config_overrides={"opt": {"max_iter": 50, "dual_run": False}},
        ),
        id="SGP4LEONoDragOrbit+PhaseCalculationRFI+UnitaryGains",
    ),
    pytest.param(
        NewComponentPipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:SGP4LEONoDragOrbit",
                "trajectory:PhaseCalculationRFI",
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
                "opt": {"max_iter": 50, "dual_run": False},
            },
        ),
        id="SGP4LEONoDragOrbit+PhaseCalculationRFI+GPGains",
    ),
    pytest.param(
        NewComponentPipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:SGP4LEOOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_vis:FourierTimeFreqGPAst",
                "gains:UnitaryGains",
            ],
            config_overrides={"opt": {"max_iter": 50, "dual_run": False}},
        ),
        id="SGP4LEOOrbit+PhaseCalculationRFI+UnitaryGains",
    ),
]


@pytest.mark.skipif(
    not _has_spacetrack_credentials(),
    reason="Space-Track credentials not configured",
)
@pytest.mark.parametrize("t_config", sgp4_configs)
def test_sgp4_component_pipeline(provide_test_data: Path, tmp_path: Path, t_config) -> None:
    """Integration tests for SGP4LEONoDragOrbit + PhaseCalculationRFI pipelines.

    These require Space-Track credentials because SGP4LEONoDragOrbit fetches
    fresh TLEs from the Space-Track API during component setup (in addition to
    the TLE fetch performed by TabConfig itself).

    Verifies the pipeline runs without error and emits a reasonable chi-squared.
    """
    returncode, stdout, stderr = _run_new_component_pipeline(
        provide_test_data, tmp_path, t_config
    )
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_new_chi2(stdout, t_config.chi2_ref)
