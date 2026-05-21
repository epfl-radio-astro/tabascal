"""Test full Tabascal pipeline runs."""

import hashlib
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.resources import files as _res_files
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



@dataclass
class PipelineTestConfig:
    """Configuration for a single pipeline test case.
    
    Attributes:
        sim_file_name: Name of the simulation YAML configuration file
        components: List of component module specifications for the pipeline
        chi2_ref: Expected reduced chi-squared value for validation
        config_overrides: Dictionary of overrides to the tabascal config file
    """
    sim_file_name: str
    components: list[str]
    chi2_ref: float
    config_overrides: dict = field(default_factory=dict)


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

    bundled_tle_dir = str(_res_files("tabascal").joinpath("data/tles"))

    result = subprocess.run(
        [
            sys.executable,
            str(tabascal_script),
            "run",
            "-c",
            str(config_path),
            "-s",
            str(input_src_dir),
            "--extra-tle-dir",
            bundled_tle_dir,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _assert_chi2(stdout: str, chi2_ref: float) -> None:
    match = re.search(r"Reduced Chi\^2 @ opt params : ([\d.eE+-]+)", stdout)
    assert match, f"Could not find Reduced Chi^2 in output: {stdout}"
    value = float(match.group(1))
    assert value == pytest.approx(chi2_ref, rel=1e-2)

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
            chi2_ref=0.8834671325695459, # Run with MEO satellites
            config_overrides={"opt": {"max_iter": 200}},
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
            chi2_ref=0.8834671467969134,  # Run with MEO satellites
            config_overrides={"opt": {"max_iter": 200}},
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

ast_signal_configs = [
    # Learnable point-source sky with a Laplace sparsity prior, init at the
    # catalogue truth. Exercises PointSky end to end (params optimised in SVI).
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_signal:PointSky",
                "ast_vis:PointSourceVisCalculation",
                "gains:UnitaryGains",
            ],
            config_overrides={"ast": {"signals": {"PointSky": {
                "init": {"type": "from_catalogue", "fmt": "zarr"},
                "start": "truth",
                "prior": {"laplace_width": 1.0}}}}},
            chi2_ref=3.2505896974263226,
        ),
        id="PointSky+PointSourceVisCalculation",
    ),
    # Fixed dense image: rasterise the data catalogue onto the config grid and
    # degrid with the wgridder. Exercises FixedImageSky + ImageVisCalculation
    # (and the shared image grid/plan) end to end.
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_signal:FixedImageSky",
                "ast_vis:ImageVisCalculation",
                "gains:UnitaryGains",
            ],
            config_overrides={"ast": {
                "grid": {"fov_deg": 5.0, "n_pix": 256, "epsilon": 1e-6},
                "signals": {"FixedImageSky": {}}}},
            chi2_ref=8.108350608931893,
        ),
        id="FixedImageSky+ImageVisCalculation",
    ),
    # Learnable dense image (log-normal GRF) -> wgridder visibilities. Exercises
    # ImageSky + ImageVisCalculation end to end (params optimised in SVI).
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_signal:ImageSky",
                "ast_vis:ImageVisCalculation",
                "gains:UnitaryGains",
            ],
            config_overrides={"ast": {
                "grid": {"fov_deg": 5.0, "n_pix": 256, "epsilon": 1e-6},
                "signals": {"ImageSky": {
                    # 'zeros' init (== prior mean): a 'from_ms' dirty-image init
                    # now works but imprints unsubtracted RFI from the data into
                    # the sky, so it suits only RFI-subtracted / astronomy data.
                    "init": {"type": "zeros"},
                    "prior": {"mean": {"type": "zeros"},
                              "pow_spec": {"p0": 1.0, "k0_freq": 1.0, "k0_lm": 300.0,
                                           "gamma_freq": 2.0, "gamma_lm": 2.0,
                                           "cutoff": 1e-6, "mu": -2.0}}}}}},
            chi2_ref=1.050073696043625,
        ),
        id="ImageSky+ImageVisCalculation",
    ),
]

# ---------------------------------------------------------------------------
# Astronomical visibility components — upstream fixed to FixedOrbit
# ---------------------------------------------------------------------------

ast_vis_configs = [
    # Fixed point-source sky (catalogue) -> direct-DFT visibilities. The
    # astronomy is fixed to truth (no free params), so this exercises the full
    # pipeline through FixedPointSky + PointSourceVisCalculation and validates
    # the uvw/visibility convention end to end. The reduced chi^2 is elevated
    # (vs the learnable FourierTimeFreqGPAst case) because the fixed model uses
    # the catalogue's intrinsic fluxes while the data carries the primary-beam
    # attenuation; it is still a stable regression guard (a convention/uvw error
    # would inflate it far more).
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFI",
                "rfi_vis:RiemannVisTimeFreqCalculation",
                "ast_signal:FixedPointSky",
                "ast_vis:PointSourceVisCalculation",
                "gains:UnitaryGains",
            ],
            chi2_ref=8.310332651546352,
        ),
        id="FixedPointSky+PointSourceVisCalculation",
    ),
]


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
def test_pipeline(provide_test_data: Path, tmp_path: Path, t_config: PipelineTestConfig) -> None:
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
        t_config: Tabascal pipeline test config 
    """
    returncode, stdout, stderr = _run_pipeline(provide_test_data, tmp_path, t_config)
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_chi2(stdout, t_config.chi2_ref)
