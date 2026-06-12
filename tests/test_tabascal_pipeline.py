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
    # Use tabascal's loader so bare scientific-notation floats (e.g. 1e6, 209e3)
    # parse as floats rather than strings; the stock yaml.safe_load does not.
    from tabascal.config import yaml_load

    data = yaml_load(input_path)

    data.update(new_data)

    with open(output_path, "w") as f:
        yaml.dump(data, f)



@dataclass
class PipelineTestConfig:
    """Configuration for a single pipeline test case.

    Attributes:
        sim_file_name: Name of the simulation YAML configuration file
        components: List of component module specifications for the pipeline
        chi2_ref: Expected reduced chi-squared per precision, keyed by the
            precision string ("double"/"single"). Double-precision values are
            consistent across architectures and tested with 1% relative
            tolerance. Single-precision values are given as ``(lo, hi)`` bounds
            because fp32 convergence rate differs across architectures: ARM
            converges in ~100 iterations (chi2 ~0.92), x86 needs ~2000
            iterations to reach a similar value (chi2 ~1.13 at 100 iters), and
            GPU overshoots to ~0.91 at 100 iters. All three land in (0.9, 1.2).
            ``requires_double`` cases only need the "double" entry.
        requires_double: True if any component only runs in double precision; the
            case is skipped under single precision (``--x64 false``).
        config_overrides: Dictionary of overrides to the tabascal config file
        metrics_ref: Optional truth-based RMSE references, keyed by precision then by
            quantity (``ast``/``rfi``/``gains``) then by metric name (``RMSE``,
            ``NRMSE(signal)``, ``NRMSE(noise)``, ``NRMSE(peak)``). Each value follows the
            same scalar-or-``(lo, hi)`` tolerance convention as ``chi2_ref`` and is
            asserted at the opt point. Only the listed metrics are checked; omit the
            field (or a precision) to skip the RMSE assertion for that case.
    """
    sim_file_name: str
    components: list[str]
    chi2_ref: dict[str, float | tuple[float, float]]
    requires_double: bool = False
    config_overrides: dict = field(default_factory=dict)
    metrics_ref: dict[str, dict[str, dict[str, float | tuple[float, float]]]] = field(
        default_factory=dict
    )


def _run_pipeline(
    provide_test_data: Path,
    tmp_path: Path,
    t_config: PipelineTestConfig,
    precision: str,
) -> tuple[int, str, str]:
    local_dir = Path(provide_test_data)
    data_dir = Path(__file__).parent / "data"
    input_hash = compute_sha256(data_dir / t_config.sim_file_name)

    input_dir = local_dir / input_hash
    config_template = data_dir / "tab_target.yaml"
    config_path = tmp_path / "tab_target.yaml"

    config_mod: dict = {
        "model": {"components": t_config.components, "precision": precision}
    }
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


def _check_value(value: float, ref: float | tuple[float, float], label: str, rel: float = 1e-2) -> None:
    """Compare a captured value against a reference.

    A scalar ``ref`` is matched with relative tolerance ``rel``; a ``(lo, hi)`` tuple is
    treated as inclusive bounds (used for single precision, whose convergence varies
    across architectures). Shared by the chi^2 and truth-metric assertions.
    """
    if isinstance(ref, tuple):
        lo, hi = ref
        assert lo <= value <= hi, f"{label} {value:.6g} not in [{lo}, {hi}]"
    else:
        assert value == pytest.approx(ref, rel=rel), f"{label} {value:.6g} != {ref} (rel={rel})"


def _assert_chi2(stdout: str, chi2_ref: float | tuple[float, float]) -> None:
    match = re.search(r"Reduced Chi\^2 @ opt params : ([\d.eE+-]+)", stdout)
    assert match, f"Could not find Reduced Chi^2 in output: {stdout}"
    _check_value(float(match.group(1)), chi2_ref, "chi2")


# Truth-metric quantity labels as printed by tabascal.tab_tools.print_truth_metrics,
# mapped to the short keys used in PipelineTestConfig.metrics_ref.
_TRUTH_QUANTITY_KEYS = {"Ast. Vis": "ast", "RFI Vis": "rfi", "Gains": "gains"}


def _parse_truth_metrics(stdout: str, point: str) -> dict[str, dict[str, float]]:
    """Capture the ``Truth metrics @ <point> params:`` block printed by a run.

    Returns ``{quantity: {metric: value}}`` where ``quantity`` is one of ``ast``/``rfi``/
    ``gains`` and ``metric`` is the printed name (``RMSE``, ``NRMSE(signal)``,
    ``NRMSE(noise)``, ``NRMSE(peak)``). An absent block (no truth available) yields ``{}``.
    """
    lines = stdout.splitlines()
    header = f"Truth metrics @ {point} params:"
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration:
        return {}

    metrics: dict[str, dict[str, float]] = {}
    for ln in lines[start + 1:]:
        label, sep, rest = ln.partition("|")
        if not sep:
            break  # block ends at the first non-metric line
        key = _TRUTH_QUANTITY_KEYS.get(label.strip())
        if key is None:
            break
        metrics[key] = {
            name: float(val)
            for name, val in re.findall(r"(RMSE|NRMSE\([a-z]+\)):\s*([-\d.eE+]+)", rest)
        }
    return metrics


def _assert_truth_metrics(
    stdout: str, point: str, metrics_ref: dict[str, dict[str, float | tuple[float, float]]]
) -> None:
    """Assert captured truth metrics against references (mirrors :func:`_assert_chi2`).

    ``metrics_ref`` is ``{quantity: {metric: scalar | (lo, hi)}}``; each value uses the
    same tolerance convention as the chi^2 reference. Only the quantities/metrics listed
    are checked, so a case can assert just the metrics it cares about.
    """
    parsed = _parse_truth_metrics(stdout, point)
    for quantity, wanted in metrics_ref.items():
        assert quantity in parsed, (
            f"No '{quantity}' truth metrics @ {point} params found in output:\n{stdout}"
        )
        for metric, ref in wanted.items():
            assert metric in parsed[quantity], (
                f"Metric '{metric}' missing for '{quantity}' @ {point} params: {parsed[quantity]}"
            )
            _check_value(parsed[quantity][metric], ref, f"{quantity} {metric}")


def test_truth_metric_capture_roundtrips_printed_output(capsys):
    """The capture parses exactly what ``print_truth_metrics`` emits.

    Round-trips the real reporter through the parser so the regex stays in sync with the
    printed format (and a future format change fails here rather than silently parsing to
    nothing). No network / sim data required.
    """
    from types import SimpleNamespace

    import jax.numpy as jnp

    from tabascal.tab_tools import print_truth_metrics

    n_bl, n_freq, n_time, n_ant = 3, 2, 4, 5
    tab_config = SimpleNamespace(noise=2.0, flags=jnp.zeros((n_bl, n_freq, n_time), dtype=bool))

    true_ast = jnp.ones((n_bl, n_freq, n_time), dtype=complex)
    truth = {
        "vis_ast": true_ast,                  # RMSE 1, signal 1, peak 1
        "vis_rfi": jnp.nan * true_ast,        # unavailable -> dynamically skipped
        "gains": jnp.full((n_ant, n_freq, n_time), 2.0 + 0j),  # RMSE 2, signal 2, peak 2
    }
    pred = {
        "vis_ast": jnp.zeros((1, n_bl, n_freq, n_time), dtype=complex),
        "vis_rfi": jnp.zeros((1, n_bl, n_freq, n_time), dtype=complex),
        "gains": jnp.zeros((1, n_ant, n_freq, n_time), dtype=complex),
    }

    print_truth_metrics(pred, truth, tab_config, "opt")
    stdout = capsys.readouterr().out

    parsed = _parse_truth_metrics(stdout, "opt")
    assert set(parsed) == {"ast", "gains"}  # rfi truth is NaN -> not printed/captured
    assert parsed["ast"]["RMSE"] == pytest.approx(1.0)
    assert parsed["ast"]["NRMSE(signal)"] == pytest.approx(1.0)
    assert parsed["ast"]["NRMSE(noise)"] == pytest.approx(0.5)   # noise = 2
    assert parsed["ast"]["NRMSE(peak)"] == pytest.approx(1.0)
    assert parsed["gains"]["RMSE"] == pytest.approx(2.0)
    assert "NRMSE(noise)" not in parsed["gains"]  # noise-norm not printed for gains

    # Assertion helper: scalar (rel tol) and (lo, hi) bounds, plus the failure paths.
    _assert_truth_metrics(stdout, "opt", {"ast": {"RMSE": 1.0, "NRMSE(noise)": (0.4, 0.6)}})
    with pytest.raises(AssertionError):
        _assert_truth_metrics(stdout, "opt", {"ast": {"RMSE": 2.0}})
    with pytest.raises(AssertionError, match="No 'rfi'"):
        _assert_truth_metrics(stdout, "opt", {"rfi": {"RMSE": 1.0}})

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
            chi2_ref={"double": 0.8977856043436502},
            requires_double=True,
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
            chi2_ref={"double": 0.8834671325695459}, # Run with MEO satellites
            requires_double=True,
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
            chi2_ref={"double": 0.8834671467969134},  # Run with MEO satellites
            requires_double=True,
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
            # single: ARM~0.916 (100 iters), x86~1.128 (100 iters), GPU~0.910 (100 iters)
            chi2_ref={"double": 0.8977856059138833, "single": (0.9, 1.2)},
            # Truth-based RMSE at the opt point (true ast/rfi/gains vis present in the
            # sim). Captured from local ARM runs: double bounds ~+/-5% (double is
            # architecture-stable); single bounds are wider to absorb the documented
            # cross-architecture spread (single chi2 ranges ARM~0.92 / x86~1.13 /
            # GPU~0.91). This sim has identity gains and UnitaryGains predicts identity,
            # so gains RMSE is ~0 exactly. Tighten once canonical CI values are recorded.
            metrics_ref={
                "double": {
                    "ast": {"RMSE": (0.18, 0.20), "NRMSE(signal)": (0.104, 0.115)},
                    "rfi": {"RMSE": (0.24, 0.26), "NRMSE(signal)": (0.022, 0.024)},
                    "gains": {"RMSE": (0.0, 1e-6)},
                },
                "single": {
                    "ast": {"RMSE": (0.17, 0.26), "NRMSE(signal)": (0.10, 0.15)},
                    "rfi": {"RMSE": (0.24, 0.36), "NRMSE(signal)": (0.021, 0.032)},
                    "gains": {"RMSE": (0.0, 1e-6)},
                },
            },
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
            # single: ARM~0.921 (100 iters), x86~1.128 (100 iters), GPU~0.910 (100 iters)
            chi2_ref={"double": 0.8977856059138833, "single": (0.9, 1.2)},
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
            # single: ARM~0.916 (100 iters), x86~1.128 (100 iters), GPU~0.910 (100 iters)
            chi2_ref={"double": 0.8977832575028029, "single": (0.9, 1.2)},
            # Truth-based RMSE at the opt point (true ast/rfi/gains vis present in the
            # sim). Captured from local ARM runs: double bounds ~+/-5%; single bounds
            # widened for the documented cross-architecture spread. GPGains fits the gains
            # and recovers them to ~4e-4; the gains upper bound leaves headroom for the
            # single-precision / cross-architecture fit residual. Tighten once canonical
            # CI values are recorded.
            metrics_ref={
                "double": {
                    "ast": {"RMSE": (0.18, 0.20), "NRMSE(signal)": (0.104, 0.115)},
                    "rfi": {"RMSE": (0.24, 0.26), "NRMSE(signal)": (0.022, 0.024)},
                    "gains": {"RMSE": (0.0, 1e-3)},
                },
                "single": {
                    "ast": {"RMSE": (0.17, 0.26), "NRMSE(signal)": (0.10, 0.15)},
                    "rfi": {"RMSE": (0.24, 0.36), "NRMSE(signal)": (0.021, 0.032)},
                    "gains": {"RMSE": (0.0, 3e-3)},
                },
            },
        ),
        id="GPGains",
    ),
]


# ---------------------------------------------------------------------------
# All pipeline tests — single parametrized function
# ---------------------------------------------------------------------------

all_configs = trajectory_configs + rfi_signal_configs + rfi_vis_configs + ast_signal_configs + ast_vis_configs + gains_configs


@pytest.mark.parametrize("t_config", all_configs)
def test_pipeline(
    provide_test_data: Path,
    tmp_path: Path,
    t_config: PipelineTestConfig,
    precision: str,
) -> None:
    """Test the complete Tabascal pipeline execution.

    This test verifies that the full Tabascal pipeline runs successfully and produces
    expected results. It:
    1. Downloads test simulation data from HuggingFace
    2. Configures the pipeline with specific component modules at the session
       precision (driven by the ``--x64`` flag)
    3. Executes the run_tabascal.py script
    4. Validates that the output Reduced Chi^2 value matches the expected result
       for that precision

    Args:
        provide_test_data: Fixture providing path to downloaded test data
        tmp_path: Pytest fixture providing temporary directory for test files
        t_config: Tabascal pipeline test config
        precision: Session precision ("double"/"single") from the --x64 flag
    """
    if t_config.requires_double and precision != "double":
        pytest.skip(
            "uses a component that requires double precision; not run under --x64 false"
        )

    chi2_ref = t_config.chi2_ref[precision]
    assert chi2_ref is not None, (
        f"No {precision}-precision chi^2 reference recorded for this case"
    )

    returncode, stdout, stderr = _run_pipeline(provide_test_data, tmp_path, t_config, precision)
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_chi2(stdout, chi2_ref)

    metrics_ref = t_config.metrics_ref.get(precision)
    if metrics_ref:
        _assert_truth_metrics(stdout, "opt", metrics_ref)
