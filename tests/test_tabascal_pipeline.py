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
        metrics_ref: Optional truth-based error references, keyed by precision then by
            quantity (``ast``/``rfi``/``gains``) then by metric name. Available metrics are
            the RMSE family (``RMSE``, ``NRMSE(noise)``, ``NRMSE(signal)``), the
            mean-error / bias family (``|ME|``, ``NME(noise)``, ``NME(signal)``) and
            ``bias_significance`` (the bias in units of sigma). Each value follows the same
            scalar-or-``(lo, hi)`` tolerance convention as ``chi2_ref`` and is asserted at the
            opt point. Only the listed metrics are checked; omit the field (or a precision) to
            skip the assertion for that case.
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

    The reporter prints two rows per quantity (``RMSE`` and ``bias``), each an absolute value
    plus ``/noise`` and ``/signal`` ratios; the ``bias`` row also carries a significance in
    sigma. They are flattened to ``{quantity: {metric: value}}`` where ``quantity`` is one of
    ``ast``/``rfi``/``gains`` and ``metric`` is one of the RMSE family (``RMSE``,
    ``NRMSE(noise)``, ``NRMSE(signal)``), the mean-error/bias family (``|ME|``, ``NME(noise)``,
    ``NME(signal)``) or ``bias_significance`` (the bias in units of sigma). An absent block
    (no truth available) yields ``{}``.
    """
    lines = stdout.splitlines()
    header = f"Truth metrics @ {point} params:"
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration:
        return {}

    # (row keyword) -> (absolute key, /noise key, /signal key)
    row_keys = {
        "RMSE": ("RMSE", "NRMSE(noise)", "NRMSE(signal)"),
        "bias": ("|ME|", "NME(noise)", "NME(signal)"),
    }
    metrics: dict[str, dict[str, float]] = {}
    current: str | None = None
    for ln in lines[start + 1:]:
        # The quantity label only appears on its first ("RMSE") row; the "bias" row has a
        # blank label, so an empty label means "still the previous quantity".
        label, sep, rest = ln.partition("|")
        if not sep:
            break  # block ends at the first non-metric line
        lbl = label.strip()
        if lbl:
            current = _TRUTH_QUANTITY_KEYS.get(lbl)
            if current is None:
                break
        rest = rest.strip()
        row = rest.split(None, 1)[0] if rest else ""
        if row not in row_keys or current is None:
            break
        abs_k, noise_k, sig_k = row_keys[row]
        d = metrics.setdefault(current, {})
        d[abs_k] = float(re.search(r"[-\d.eE+]+", rest[len(row):]).group())
        if (m := re.search(r"/noise\s+([-\d.eE+]+)", rest)):
            d[noise_k] = float(m.group(1))
        if (m := re.search(r"/signal\s+([-\d.eE+]+)", rest)):
            d[sig_k] = float(m.group(1))
        if (m := re.search(r"\[\s*([-\d.eE+]+)\s*sigma", rest)):
            d["bias_significance"] = float(m.group(1))  # printed in units of sigma
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
        "vis_ast": true_ast,                  # RMSE 1, signal 1
        "vis_rfi": jnp.nan * true_ast,        # unavailable -> dynamically skipped
        "gains": jnp.full((n_ant, n_freq, n_time), 2.0 + 0j),  # RMSE 2, signal 2
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
    # ast: pred 0 vs true 1 -> RMSE 1, |ME| 1 (constant error -> coherent), signal 1, noise 2.
    # A constant residual is fully coherent: N_eff = 1, so sigma = |ME| * sqrt(2) / RMSE ~ 1.4.
    assert parsed["ast"]["RMSE"] == pytest.approx(1.0)
    assert parsed["ast"]["NRMSE(noise)"] == pytest.approx(0.5)   # noise = 2
    assert parsed["ast"]["NRMSE(signal)"] == pytest.approx(1.0)
    assert parsed["ast"]["|ME|"] == pytest.approx(1.0)
    assert parsed["ast"]["NME(noise)"] == pytest.approx(0.5)
    assert parsed["ast"]["NME(signal)"] == pytest.approx(1.0)
    assert parsed["ast"]["bias_significance"] == pytest.approx(1.4)   # printed to 1 dp
    assert parsed["gains"]["RMSE"] == pytest.approx(2.0)
    assert parsed["gains"]["|ME|"] == pytest.approx(2.0)
    assert parsed["gains"]["bias_significance"] == pytest.approx(1.4)
    assert "NRMSE(noise)" not in parsed["gains"]  # noise-norm not printed for gains
    assert "NME(noise)" not in parsed["gains"]

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
            # requires_double (phase trajectory needs fp64): only the double chi2 is asserted;
            # truth metrics are printed but not asserted. Measured opt-point values, double
            # precision (gains identity -> RMSE 0). Fill in x86/GPU after running there:
            #   arch | chi2  | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   ARM  | 0.904 |     0.293        0.7      |     0.388        0.2
            #   x86  |  TBD  |      TBD         TBD      |      TBD         TBD
            #   GPU  |  TBD  |      TBD         TBD      |      TBD         TBD
            # (recorded chi2_ref below is the CI/x86 value; ARM reproduces ~0.7% high, within 1% tol.)
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
            # requires_double; only double chi2 asserted (MEO satellites, opt max_iter 200).
            # Measured opt-point values, double precision (gains identity -> RMSE 0). Fill in
            # x86/GPU after running there:
            #   arch | chi2  | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   ARM  | 0.883 |     0.343        0.7      |     0.452        0.4
            #   x86  |  TBD  |      TBD         TBD      |      TBD         TBD
            #   GPU  |  TBD  |      TBD         TBD      |      TBD         TBD
            chi2_ref={"double": 0.8834671325695459},
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
            # requires_double; only double chi2 asserted (MEO satellites, opt max_iter 200).
            # Measured opt-point values, double precision (gains identity -> RMSE 0; matches
            # SGP4LEONoDragOrbit -- same orbit to fp precision). Fill in x86/GPU after running:
            #   arch | chi2  | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   ARM  | 0.883 |     0.343        0.7      |     0.452        0.4
            #   x86  |  TBD  |      TBD         TBD      |      TBD         TBD
            #   GPU  |  TBD  |      TBD         TBD      |      TBD         TBD
            chi2_ref={"double": 0.8834671467969134},
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
            # Truth-based metrics at the opt point. ast/rfi assert NRMSE(noise) -- the residual
            # against the thermal-noise floor, the science-meaningful yardstick (< 1 means
            # sub-noise) and the most architecture-stable normalisation -- plus
            # bias_significance, the coherent mean error in units of sigma (correlation-deflated
            # by N_eff; see print_truth_metrics). The significance bound is a "no significant
            # coherent bias" guard, not a tight value: the bias is ~1 sigma here (N_eff ~ 50),
            # so the upper bound only trips on gross RFI->ast leakage.
            #
            # Measured opt-point values (UnitaryGains -> identity gains, so gains RMSE ~0); chi2
            # is the ARM-measured opt value (chi2_ref above is the asserted CI/x86 reference).
            # Fill in the TBD rows after running on those arches:
            #   precision/arch | ast NRMSE(noise)  ast sig | rfi NRMSE(noise)  rfi sig | chi2
            #   double  ARM    |      0.293         0.7     |      0.389        0.2     | 0.904
            #   double  x86    |       TBD          TBD     |       TBD         TBD     |  TBD
            #   double  GPU    |       TBD          TBD     |       TBD         TBD     |  TBD
            #   single  ARM    |      0.294         1.0     |      0.407        1.3     | 0.916
            #   single  x86    |       TBD          TBD     |      ~0.78        TBD     | 1.128
            #   single  GPU    |       TBD          TBD     |       TBD         TBD     | 0.910
            # (single x86 rfi ~0.78 is inferred from the CI RMSE that motivated the wide single
            # bounds; its slower fp32 convergence inflates the rfi residual. double is
            # architecture-stable to ~+/-5%. Tighten once canonical CI values are recorded.)
            metrics_ref={
                "double": {
                    "ast": {"NRMSE(noise)": (0.27, 0.31), "bias_significance": (0.0, 2.0)},
                    "rfi": {"NRMSE(noise)": (0.36, 0.41), "bias_significance": (0.0, 2.0)},
                    "gains": {"RMSE": (0.0, 1e-6)},
                },
                "single": {
                    "ast": {"NRMSE(noise)": (0.20, 0.50), "bias_significance": (0.0, 4.0)},
                    "rfi": {"NRMSE(noise)": (0.30, 0.90), "bias_significance": (0.0, 4.0)},
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
            # Only chi2 is asserted -- the FFI kernel is the unit under test; truth metrics
            # match the non-FFI RiemannVisTimeFreqCalculation case above. Measured opt-point
            # values (local ARM via the real harness; gains identity -> RMSE 0):
            #   precision/arch | chi2  | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   double  ARM    | 0.904 |     0.293        0.7      |     0.389        0.2
            #   double  x86    |  TBD  |      TBD         TBD      |      TBD         TBD
            #   double  GPU    |  TBD  |      TBD         TBD      |      TBD         TBD
            #   single  ARM    | 0.921 |     0.294        1.0      |     0.407        1.3
            #   single  x86    | 1.128 |      TBD         TBD      |      TBD         TBD
            #   single  GPU    | 0.910 |      TBD         TBD      |      TBD         TBD
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
            # Truth-based metrics at the opt point; same scheme as RiemannVisTimeFreqCalculation
            # (ast/rfi assert NRMSE(noise) against the noise floor + bias_significance as a
            # "no significant coherent bias" guard). Here GPGains fits the gains and recovers
            # them, so gains keeps an RMSE bound with headroom for the fp32 fit residual.
            #
            # Measured opt-point values; chi2 is the ARM-measured opt value (chi2_ref above is
            # the asserted CI/x86 reference). Fill in the TBD rows after running on those arches:
            #   precision/arch | ast NRMSE(noise)  ast sig | rfi NRMSE(noise)  rfi sig | gains RMSE | chi2
            #   double  ARM    |      0.293         0.7     |      0.389        0.2     |  3.9e-4    | 0.904
            #   double  x86    |       TBD          TBD     |       TBD         TBD     |   TBD      |  TBD
            #   double  GPU    |       TBD          TBD     |       TBD         TBD     |   TBD      |  TBD
            #   single  ARM    |      0.294         0.9     |      0.407        1.3     |  4.3e-4    | 0.916
            #   single  x86    |       TBD          TBD     |      ~0.78        TBD     |   TBD      | 1.128
            #   single  GPU    |       TBD          TBD     |       TBD         TBD     |   TBD      | 0.910
            # (single x86 rfi ~0.78 inferred from the CI RMSE that motivated the wide single
            # bounds. double is architecture-stable to ~+/-5%. Tighten once canonical CI values
            # are recorded.)
            metrics_ref={
                "double": {
                    "ast": {"NRMSE(noise)": (0.27, 0.31), "bias_significance": (0.0, 2.0)},
                    "rfi": {"NRMSE(noise)": (0.36, 0.41), "bias_significance": (0.0, 2.0)},
                    "gains": {"RMSE": (0.0, 1e-3)},
                },
                "single": {
                    "ast": {"NRMSE(noise)": (0.20, 0.50), "bias_significance": (0.0, 4.0)},
                    "rfi": {"NRMSE(noise)": (0.30, 0.90), "bias_significance": (0.0, 4.0)},
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
