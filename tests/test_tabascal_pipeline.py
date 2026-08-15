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
        chi2_ref: Expected reduced chi-squared, tested with 1% relative tolerance.
            One value covers both precisions. The pre-#103 real-space model needed
            separate per-precision references because its fp32 convergence was
            architecture-dependent (ARM reached chi2 ~0.92 in ~100 iterations while
            x86 was still at ~1.13 there), which forced wide ``(lo, hi)`` bounds for
            single. The current Fourier model converges to the same point in fp32 as
            in fp64 -- measured agreement is 2.4e-5 relative, ~400x inside the 1%
            tolerance, and ARM/x86/GPU agree with each other to ~1e-7 in both
            precisions -- so the split bought nothing and was removed.
        requires_double: True if any component only runs in double precision; the
            case is skipped under single precision (``--x64 false``).
        config_overrides: Dictionary of overrides to the tabascal config file
        metrics_ref: Optional truth-based error references, keyed by quantity
            (``ast``/``rfi``/``gains``) then by metric name. Like ``chi2_ref`` these are
            precision-independent: every case measured identical values in fp32 and fp64
            to the printed precision. Available metrics are
            the RMSE family (``RMSE``, ``NRMSE(noise)``, ``NRMSE(signal)``), the
            mean-error / bias family (``|ME|``, ``NME(noise)``, ``NME(signal)``) and
            ``bias_significance`` (the bias in units of sigma). Each value follows the same
            scalar-or-``(lo, hi)`` tolerance convention as ``chi2_ref`` and is asserted at the
            opt point. Only the listed metrics are checked; omit the field (or a precision) to
            skip the assertion for that case.
    """
    sim_file_name: str
    components: list[str]
    chi2_ref: float | tuple[float, float]
    requires_double: bool = False
    config_overrides: dict = field(default_factory=dict)
    metrics_ref: dict[str, dict[str, float | tuple[float, float]]] = field(
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
            "--extra-orbit-dir",
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
# Trajectory components — downstream fixed to RiemannVis + UnitaryGains
# ---------------------------------------------------------------------------

trajectory_configs = [
    # FixedOrbit without PhaseCalculationRFI covered by RiemannVis
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFIVarAnt",
                "rfi_vis:RiemannVis",
                "ast_vis:GPVisAst",
                "gains:UnitaryGains",
            ],
            # requires_double (phase trajectory needs fp64), so this runs in fp64 only.
            # Measured opt-point values, double precision (gains identity -> RMSE 0):
            #   arch | chi2               | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   ARM  | 0.8768424518784641 |     0.3061       0.6      |     0.4318       0.1
            #   x86  | 0.8768424504584391 |     0.3061       0.6      |     0.4318       0.1
            #   GPU  | 0.8768424565006032 |     0.3061       0.6      |     0.4318       0.1
            # (re-recorded for issue #103, which replaced the deleted real-space
            # rfi_signal:ComplexRFI with the Fourier ComplexRFIVarAnt. ARM = Apple silicon
            # CPU, x86 = x86_64 CPU, GPU = GH200 on CSCS Daint. A consumer x86 GPU was also
            # checked and tracks the GH200 column to <1e-7; it has no row of its own only to
            # keep these tables to three. Every platform agrees to <1e-7 relative, far inside
            # the 1% tolerance, so the asserted value being the ARM one is immaterial.)
            chi2_ref=0.8768424518784641,
            requires_double=True,
            metrics_ref={
                "ast": {"NRMSE(noise)": (0.28, 0.33), "bias_significance": (0.0, 2.0)},
                "rfi": {"NRMSE(noise)": (0.40, 0.46), "bias_significance": (0.0, 2.0)},
                "gains": {"RMSE": (0.0, 1e-6)},
            },
        ),
        id="FixedOrbit+PhaseCalculationRFI",
    ),
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:NoDragOrbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFIVarAnt",
                "rfi_vis:RiemannVis",
                "ast_vis:GPVisAst",
                "gains:UnitaryGains",
            ],
            # requires_double, so this runs in fp64 only. max_iter 200: these MEO cases are
            # genuinely not converged at the standard 100 -- re-measured there, chi2 is 5.5%
            # higher (0.9017) and both NRMSEs are materially worse (ast 0.4130, rfi 0.6027).
            # Measured opt-point values, double precision (gains identity -> RMSE 0); see the
            # note on the FixedOrbit case for what ARM/x86/GPU are:
            #   arch | chi2               | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   ARM  | 0.8546158788536706 |     0.3648       1.0      |     0.5199       1.2
            #   x86  | 0.8546162445425675 |     0.3648       1.0      |     0.5199       1.2
            #   GPU  | 0.8546157757462419 |     0.3648       1.0      |     0.5199       1.2
            # (widest spread of any case, and still only 4e-7 relative.)
            chi2_ref=0.8546158788536706,
            requires_double=True,
            config_overrides={"opt": {"max_iter": 200}},
            metrics_ref={
                "ast": {"NRMSE(noise)": (0.34, 0.39), "bias_significance": (0.0, 2.0)},
                "rfi": {"NRMSE(noise)": (0.48, 0.56), "bias_significance": (0.0, 2.0)},
                "gains": {"RMSE": (0.0, 1e-6)},
            },
        ),
        id="NoDragOrbit+PhaseCalculationRFI",
    ),
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:Orbit",
                "trajectory:PhaseCalculationRFI",
                "rfi_signal:ComplexRFIVarAnt",
                "rfi_vis:RiemannVis",
                "ast_vis:GPVisAst",
                "gains:UnitaryGains",
            ],
            # requires_double, so this runs in fp64 only. max_iter 200: these MEO cases are
            # genuinely not converged at the standard 100 -- re-measured there, chi2 is 5.5%
            # higher (0.9017) and both NRMSEs are materially worse (ast 0.4130, rfi 0.6027).
            # Measured opt-point values, double precision (gains identity -> RMSE 0; matches
            # NoDragOrbit -- same orbit to fp precision):
            #   arch | chi2               | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   ARM  | 0.8546160082354400 |     0.3648       1.0      |     0.5199       1.2
            #   x86  | 0.8546160234159469 |     0.3648       1.0      |     0.5199       1.2
            #   GPU  | 0.8546160141310710 |     0.3648       1.0      |     0.5199       1.2
            chi2_ref=0.85461600823544,
            requires_double=True,
            config_overrides={"opt": {"max_iter": 200}},
            metrics_ref={
                "ast": {"NRMSE(noise)": (0.34, 0.39), "bias_significance": (0.0, 2.0)},
                "rfi": {"NRMSE(noise)": (0.48, 0.56), "bias_significance": (0.0, 2.0)},
                "gains": {"RMSE": (0.0, 1e-6)},
            },
        ),
        id="Orbit+PhaseCalculationRFI",
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
                "rfi_signal:ComplexRFIVarAnt",
                "rfi_vis:RiemannVis",
                "ast_vis:GPVisAst",
                "gains:UnitaryGains",
            ],
            chi2_ref=0.8767134197875864,
            # Truth-based metrics at the opt point. ast/rfi assert NRMSE(noise) -- the residual
            # against the thermal-noise floor, the science-meaningful yardstick (< 1 means
            # sub-noise) and the most architecture-stable normalisation -- plus
            # bias_significance, the coherent mean error in units of sigma (correlation-deflated
            # by N_eff; see print_truth_metrics). The significance bound is a "no significant
            # coherent bias" guard, not a tight value: the bias is ~1 sigma here (N_eff ~ 50),
            # so the upper bound only trips on gross RFI->ast leakage.
            #
            # Measured opt-point values (UnitaryGains -> identity gains, so gains RMSE ~0),
            # re-recorded for issue #103 (rfi_signal:ComplexRFI deleted -> ComplexRFIVarAnt)
            # on all three platforms in both precisions:
            #   precision/arch | ast NRMSE(noise)  ast sig | rfi NRMSE(noise)  rfi sig | chi2
            #   double  ARM    |      0.3062        0.6     |      0.4320       0.1     | 0.8767134198
            #   double  x86    |      0.3062        0.6     |      0.4320       0.1     | 0.8767134197
            #   double  GPU    |      0.3062        0.6     |      0.4320       0.1     | 0.8767134177
            #   single  ARM    |      0.3063        0.6     |      0.4320       0.1     | 0.8767347336
            #   single  x86    |      0.3063        0.6     |      0.4320       0.1     | 0.8767347336
            #   single  GPU    |      0.3063        0.6     |      0.4320       0.1     | 0.8767347336
            # fp32 and fp64 agree to 2.4e-5 on chi2 and to the printed precision on the
            # metrics, on every platform tested -- the old real-space component did not --
            # so a single set of references covers both and there is no per-precision split.
            metrics_ref={
                "ast": {"NRMSE(noise)": (0.28, 0.33), "bias_significance": (0.0, 2.0)},
                "rfi": {"NRMSE(noise)": (0.40, 0.46), "bias_significance": (0.0, 2.0)},
                "gains": {"RMSE": (0.0, 1e-6)},
            },
        ),
        id="RiemannVis",
    ),
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFIVarAnt",
                "rfi_vis:RiemannVisFFI",
                "ast_vis:GPVisAst",
                "gains:UnitaryGains",
            ],
            # Only chi2 is asserted -- the FFI kernel is the unit under test; truth metrics
            # match the non-FFI RiemannVis case above. Measured opt-point
            # values (gains identity -> RMSE 0):
            #   precision/arch | chi2         | ast NRMSE(noise) ast sig | rfi NRMSE(noise) rfi sig
            #   double  ARM    | 0.8767134198 |     0.3062       0.6      |     0.4320       0.1
            #   double  x86    | 0.8767134197 |     0.3062       0.6      |     0.4320       0.1
            #   double  GPU    | 0.8767134177 |     0.3062       0.6      |     0.4320       0.1
            #   single  ARM    | 0.8767345548 |     0.3063       0.6      |     0.4320       0.1
            #   single  x86    | 0.8767346144 |     0.3063       0.6      |     0.4320       0.1
            #   single  GPU    | 0.8767346740 |     0.3063       0.6      |     0.4320       0.1
            chi2_ref=0.8767134197875865,
        ),
        id="RiemannVisFFI",
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
    # UnitaryGains covered by RiemannVis
    pytest.param(
        PipelineTestConfig(
            "sim_target_8A.yaml",
            [
                "trajectory:FixedOrbit",
                "rfi_signal:ComplexRFIVarAnt",
                "rfi_vis:RiemannVis",
                "ast_vis:GPVisAst",
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
            chi2_ref=0.8767019546502159,
            # Truth-based metrics at the opt point; same scheme as RiemannVis
            # (ast/rfi assert NRMSE(noise) against the noise floor + bias_significance as a
            # "no significant coherent bias" guard). Here GPGains fits the gains and recovers
            # them, so gains keeps an RMSE bound with headroom for the fp32 fit residual.
            #
            # Measured opt-point values, re-recorded for issue #103 on all three platforms in
            # both precisions. Unlike the UnitaryGains cases, GPGains actually fits the gains,
            # so gains RMSE is a real fitted residual here rather than an exact zero:
            #   precision/arch | ast NRMSE(noise)  ast sig | rfi NRMSE(noise)  rfi sig | gains RMSE | chi2
            #   double  ARM    |      0.3062        0.6     |      0.4323       0.1     |  5.358e-4  | 0.8767019547
            #   double  x86    |      0.3062        0.6     |      0.4323       0.1     |  5.358e-4  | 0.8767019546
            #   double  GPU    |      0.3062        0.6     |      0.4323       0.1     |  5.358e-4  | 0.8767019526
            #   single  ARM    |      0.3062        0.6     |      0.4323       0.1     |  5.364e-4  | 0.8767232299
            #   single  x86    |      0.3062        0.6     |      0.4323       0.1     |  5.363e-4  | 0.8767232299
            #   single  GPU    |      0.3062        0.6     |      0.4323       0.1     |  5.364e-4  | 0.8767232299
            # The gains RMSE bound keeps ~2x headroom over the measured fp32 residual, which
            # is why one bound covers both precisions (fp64 5.358e-4, fp32 5.364e-4).
            metrics_ref={
                "ast": {"NRMSE(noise)": (0.28, 0.33), "bias_significance": (0.0, 2.0)},
                "rfi": {"NRMSE(noise)": (0.40, 0.46), "bias_significance": (0.0, 2.0)},
                "gains": {"RMSE": (0.0, 1e-3)},
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

    returncode, stdout, stderr = _run_pipeline(provide_test_data, tmp_path, t_config, precision)
    assert returncode == 0, f"Tabascal failed: {stderr}"
    _assert_chi2(stdout, t_config.chi2_ref)

    if t_config.metrics_ref:
        _assert_truth_metrics(stdout, "opt", t_config.metrics_ref)


# ---------------------------------------------------------------------------
# RFI-axis device sharding (multi-GPU / multi-process)
# ---------------------------------------------------------------------------
#
# tabascal shards the MAP solve over the RFI-source axis whenever more than one
# device is visible (see tabascal/distributed.py). These tests fake extra devices
# on CPU (XLA_FLAGS) / spawn two coordinated processes (jax.distributed) and
# require the sharded solve to reproduce the single-device solve. The test sim has
# 3 satellites, so a 2-device mesh also exercises the dark-dummy padding (3 -> 4).

def _sharded_components(rfi_vis: str) -> list[str]:
    return [
        "trajectory:FixedOrbit",
        "trajectory:PhaseCalculationRFI",
        "rfi_signal:ComplexRFIVarAnt",
        f"rfi_vis:{rfi_vis}",
        "ast_vis:GPVisAst",
        "gains:UnitaryGains",
    ]


# Same case as FixedOrbit+PhaseCalculationRFI above (double precision), so it shares that
# case's re-recorded reference. Verified on ARM CPU, x86 CPU and a GH200 node (the sharded
# child is pinned to CPU by this test either way; the reference run uses whatever is there).
_SHARDED_CHI2_REF = 0.8768424518784641


def _prepare_sharded_run(
    provide_test_data: Path,
    work_dir: Path,
    rfi_vis: str = "RiemannVis",
) -> tuple[list[str], Path]:
    """Copy the 8A/3-satellite sim into ``work_dir`` and build the run command.

    The copy keeps the sim directory basename (the zarr/MS paths inside the run are
    derived from it) and keeps these runs from writing results into the shared
    HuggingFace cache -- the multi-process test in particular has two processes
    using one sim directory.
    """
    import shutil

    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).parent / "data"
    input_hash = compute_sha256(data_dir / "sim_target_8A.yaml")
    input_dir = Path(provide_test_data) / input_hash
    src = next((d for d in input_dir.glob("pnt_src*") if d.is_dir()), None)
    assert src, f"No pnt_src* directory found in {input_dir}"
    sim_dir = work_dir / src.name
    shutil.copytree(src, sim_dir)

    config_path = work_dir / "tab_target.yaml"
    read_and_modify_yaml(
        {"model": {"components": _sharded_components(rfi_vis), "precision": "double"}},
        data_dir / "tab_target.yaml",
        config_path,
    )

    script = Path(__file__).parent.parent / "tabascal" / "scripts" / "run_tabascal.py"
    tle_dir = str(_res_files("tabascal").joinpath("data/tles"))
    cmd = [
        sys.executable, str(script), "run",
        "-c", str(config_path),
        "-s", str(sim_dir),
        "--extra-orbit-dir", tle_dir,
        "-nl",
    ]
    return cmd, sim_dir


def _extract_chi2(stdout: str, point: str) -> float:
    match = re.search(rf"Reduced Chi\^2 @ {point} params : ([\d.eE+-]+)", stdout)
    assert match, f"Could not find 'Reduced Chi^2 @ {point}' in output:\n{stdout}"
    return float(match.group(1))


@pytest.mark.parametrize(
    "rfi_vis",
    [
        # The plain variant runs entirely through GSPMD+shard_map on pure JAX ops;
        # the FFI variant additionally exercises the custom C kernel inside
        # shard_map (the reason for check_vma=False in psum_over_rfi).
        "RiemannVis",
        "RiemannVisFFI",
    ],
)
def test_pipeline_sharded_equivalence(
    provide_test_data: Path, tmp_path: Path, precision: str, rfi_vis: str
) -> None:
    """Sharded run (2 fake CPU devices, padding 3 sats -> 4) == single-device run.

    Exact in double precision up to summation reduction order, hence the requires-
    double skip and the tight (but not bitwise) tolerance on the optimized chi^2.

    Both legs are pinned to CPU, and the device count of each is set explicitly, so
    the comparison does not depend on what hardware the test happens to run on. That
    matters in both directions: without pinning, the reference leg picks up every
    visible accelerator and shards itself (breaking the "single device" assertions on
    a multi-GPU node), and on a single-GPU machine it runs on the GPU while the
    sharded leg runs on the CPU -- an accelerator-vs-CPU comparison that misses the
    1e-8 init tolerance by ~1e-8 purely through differing float reduction order. The
    property under test is that sharding does not change the answer, which is a
    property of the sharding logic rather than of any device; single-device
    accelerator execution is covered by ``test_pipeline`` above.
    """
    if precision != "double":
        pytest.skip("equivalence is asserted exactly; components require double")

    import os

    def _cpu_env(n_devices: int) -> dict:
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={n_devices}"
        ).strip()
        # Pinning to CPU is only half of it: a stale CUDA_VISIBLE_DEVICES in the
        # parent environment would still be inherited by the child.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        return env

    ref_cmd, _ = _prepare_sharded_run(provide_test_data, tmp_path / "ref", rfi_vis)
    ref = subprocess.run(
        ref_cmd, capture_output=True, text=True, cwd=tmp_path / "ref", env=_cpu_env(1)
    )
    assert ref.returncode == 0, f"single-device run failed: {ref.stderr}"

    shard_cmd, _ = _prepare_sharded_run(provide_test_data, tmp_path / "shard", rfi_vis)
    shard = subprocess.run(
        shard_cmd, capture_output=True, text=True, cwd=tmp_path / "shard", env=_cpu_env(2)
    )
    assert shard.returncode == 0, f"sharded run failed: {shard.stderr}"

    assert "Padded 3 RFI sources to 4" in shard.stdout
    assert "Sharding RFI sources over 2 devices:" in shard.stdout
    assert "Sharding" not in ref.stdout
    assert "Running on single device:" in ref.stdout

    assert _extract_chi2(shard.stdout, "opt") == pytest.approx(
        _extract_chi2(ref.stdout, "opt"), rel=1e-6
    )
    assert _extract_chi2(shard.stdout, "init") == pytest.approx(
        _extract_chi2(ref.stdout, "init"), rel=1e-8
    )
    _assert_chi2(shard.stdout, _SHARDED_CHI2_REF)


def test_pipeline_multiprocess(
    provide_test_data: Path, tmp_path: Path, precision: str
) -> None:
    """Two coordinated processes (jax.distributed, 1 CPU device each) solve once.

    Verifies the full multi-process path: coordinator bring-up through the
    MASTER_ADDR/RANK env route in init_distributed, RFI padding/sharding across
    processes, worker stdout silence, and rank-0-only result writing.
    """
    if precision != "double":
        pytest.skip("uses components that require double precision")

    import os
    import socket

    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]

    cmd, sim_dir = _prepare_sharded_run(provide_test_data, tmp_path)
    # Drop any launcher variables we inherited: ``init_distributed`` consults SLURM /
    # OpenMPI / PMI before the torchrun-style ``WORLD_SIZE``/``RANK`` pair, so running
    # this test inside an allocation (SLURM_NTASKS=1, SLURM_PROCID=0) would shadow the
    # coordinates set below -- the two children would then each no-op past
    # ``jax.distributed.initialize`` and run as independent single-process jobs.
    base_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("SLURM_", "OMPI_", "PMI_"))
    }
    # One CPU device per process (no --xla_force_host_platform_device_count): the two
    # processes must contribute one device each to a 2-device global mesh.
    base_env.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "JAX_PLATFORMS": "cpu",
    })

    procs = [
        subprocess.Popen(
            cmd,
            env={**base_env, "RANK": str(rank)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
        )
        for rank in (0, 1)
    ]
    outs = [p.communicate(timeout=600) for p in procs]
    for rank, (p, (out, err)) in enumerate(zip(procs, outs)):
        assert p.returncode == 0, f"rank {rank} failed:\nstdout:\n{out}\nstderr:\n{err}"

    rank0_out = outs[0][0]
    assert "Sharding RFI sources over 2 devices:" in rank0_out
    _assert_chi2(rank0_out, _SHARDED_CHI2_REF)

    # Workers must be silent and must not write results; rank 0 wrote them once.
    # The CPU collectives backend (gloo) prints a connection banner from C++ straight to
    # fd 1 on every rank during jax.distributed.initialize -- before any tabascal code
    # runs, and out of reach of suppress_worker_stdout, which only rebinds sys.stdout.
    # It is runtime chatter, not tabascal output, so drop it before asserting silence.
    worker_out = "".join(
        line for line in outs[1][0].splitlines(keepends=True)
        if not line.startswith("[Gloo]")
    )
    assert worker_out == "", f"rank 1 was not silent:\n{worker_out}"
    assert (sim_dir / "results" / "map_pred_Custom.zarr").is_dir()
