"""Tests for tabascal.validation — config schema checking.

Validation happens before the MS read and TLE fetch, and reports *every* problem
in one go. The tests below are split into the layers described in the module
docstring of ``tabascal/validation.py``: the static schema (unknown keys, types,
enums, ranges) and the cross-field checks, which
:func:`tabascal.config.load_config` runs, then the per-component
``required_config`` / ``config_choices`` declarations, which the run defers to
``build_model`` so the component imports stay behind ``set_precision``.

The last tests in the file are the important ones in the other direction: every
config shipped in the repository must still load, and loading one must not drag
in the component stack. Those are the guards against the schema drifting into
being stricter than the code actually is, and against the deferred layer
creeping back into load time.
"""

import subprocess
import sys

import pytest

from tabascal.config import load_config, yaml_load
from tabascal.validation import ConfigError, validate_config


# A component list that exercises all five component modules. Used wherever a
# test needs the per-component checks to actually run.
COMPONENTS = [
    "trajectory:FixedOrbit",
    "rfi_signal:ComplexRFI",
    "rfi_vis:RiemannVisTimeFreqCalculationFFI",
    "ast_vis:FourierTimeFreqGPAst",
    "gains:UnitaryGains",
]


def write_config(tmp_path, body: str):
    """Write a config file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return str(p)


def base_config(**sections):
    """A minimal config dict that passes validation, with overrides applied."""
    config = {
        "model": {"components": list(COMPONENTS), "precision": "single"},
        "data": {"corr": "xx", "data_col": "DATA", "flags": False},
        "plots": {"truth": False, "prior": False, "prior_samples": 100},
        "opt": {"epsilon": 1e-2, "max_iter": 500, "guide": "map"},
        "ast": {
            "init": "sample",
            "mean": 0,
            "freq_pad_factor": 2,
            "time_pad_factor": 2,
            "pow_spec": {"p0": 3e3, "k0_freq": 1, "gammas": [5, 5], "cutoff": 1e-6},
        },
        "rfi": {
            "init": "sample",
            "freq_int_samples": 1,
            "min_time_bins": 1,
            "max_time_bins": 30,
        },
    }
    for name, overrides in sections.items():
        config.setdefault(name, {}).update(overrides)
    return config


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def test_unknown_key_is_rejected_with_a_suggestion():
    """A typo must not merge silently into a run using the default."""
    config = base_config()
    config["opt"]["max_itr"] = 10

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    message = str(excinfo.value)
    assert "opt.max_itr" in message
    assert "unknown key" in message
    assert "'max_iter'" in message


def test_unknown_key_without_a_close_match_still_errors():
    config = base_config()
    config["rfi"]["totally_made_up"] = 1

    with pytest.raises(ConfigError, match=r"rfi\.totally_made_up.*unknown key"):
        validate_config(config)


def test_unknown_top_level_section_is_rejected():
    config = base_config()
    config["plot"] = {"init": True}  # missing 's'

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    assert "'plots'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Types, enums, ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section, key, value, expected",
    [
        ("model", "precision", "float32", "is not one of"),
        ("data", "corr", "rr", "is not one of"),
        ("opt", "guide", "svi", "is not one of"),
        ("ast", "init", "bogus", "is not one of"),
        ("opt", "max_iter", "many", "expected an integer"),
        ("data", "flags", "yes_please", "expected a boolean"),
        ("opt", "epsilon", -1e-3, "expected a number > 0"),
        ("opt", "max_iter", -1, "expected a number >= 0"),
        ("rfi", "freq_int_samples", 0, "expected a number >= 1"),
    ],
)
def test_invalid_values_are_rejected(section, key, value, expected):
    config = base_config()
    config[section][key] = value

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    message = str(excinfo.value)
    assert f"{section}.{key}" in message
    assert expected in message


def test_booleans_are_rejected_where_a_number_is_required():
    """isinstance(True, int) is True in Python, so this needs an explicit guard.

    The per-component validators this replaced accepted ``epsilon: true``.
    """
    config = base_config()
    config["opt"]["epsilon"] = True

    with pytest.raises(ConfigError, match=r"opt\.epsilon.*expected a number"):
        validate_config(config)


def test_list_element_types_are_checked():
    config = base_config()
    config.setdefault("satellites", {})["norad_ids"] = [123, "abc"]

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    assert "satellites.norad_ids[1]" in str(excinfo.value)
    assert "expected an integer" in str(excinfo.value)


def test_null_is_accepted_for_optional_keys():
    """`null` is how the base config spells 'derive this from the data'."""
    config = base_config()
    config["data"]["noise"] = None
    config["ast"]["pow_spec"]["fov_deg"] = None

    validate_config(config)  # does not raise


@pytest.mark.parametrize(
    "key", ["freq_pad_factor", "time_pad_factor", "freq_int_samples"]
)
def test_null_is_rejected_for_the_keys_read_on_every_run(key):
    """These are read by fix_padding / _set_freqs_times, not by a component.

    No ``required_config`` covers them, and ``fix_padding`` no longer has a
    try/except, so a null would surface as a bare TypeError while TabConfig is
    being built.
    """
    config = base_config(rfi={"freq_pad_factor": 2, "time_pad_factor": 2})
    config["rfi"][key] = None

    with pytest.raises(ConfigError, match=rf"rfi\.{key}.*cannot be null"):
        validate_config(config)


def test_a_section_that_is_not_a_mapping_is_rejected():
    """`plots:` with nothing under it wipes the merged defaults to None."""
    config = base_config()
    config["plots"] = None

    with pytest.raises(ConfigError, match=r"plots.*expected a mapping"):
        validate_config(config)


# ---------------------------------------------------------------------------
# Every problem is reported at once
# ---------------------------------------------------------------------------


def test_all_problems_are_reported_in_a_single_error():
    """The whole design turns on this: no fix-one-rerun-repeat loop."""
    config = base_config()
    config["opt"]["max_itr"] = 10
    config["model"]["precision"] = "float32"
    config["opt"]["epsilon"] = -1e-3
    config["data"]["corr"] = "rr"

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    message = str(excinfo.value)
    for expected in ("opt.max_itr", "model.precision", "opt.epsilon", "data.corr"):
        assert expected in message


def test_error_message_names_the_config_file():
    config = base_config()
    config["opt"]["max_itr"] = 10

    with pytest.raises(ConfigError, match="my_run.yaml"):
        validate_config(config, "my_run.yaml")


# ---------------------------------------------------------------------------
# Cross-field checks
# ---------------------------------------------------------------------------


def test_min_time_bins_must_not_exceed_max_time_bins():
    config = base_config()
    config["rfi"]["min_time_bins"] = 40
    config["rfi"]["max_time_bins"] = 30

    with pytest.raises(ConfigError, match=r"rfi\.min_time_bins.*must be <="):
        validate_config(config)


def test_prior_plots_require_prior_samples():
    config = base_config()
    config["plots"]["prior"] = True
    config["plots"]["prior_samples"] = None

    with pytest.raises(ConfigError, match=r"plots\.prior_samples"):
        validate_config(config)


# ---------------------------------------------------------------------------
# Per-component requirements
#
# `validate_config` runs this layer; `load_config` deliberately does not (see
# the module docstring). The run calls it from `build_model` instead.
# ---------------------------------------------------------------------------


def test_missing_component_required_key_names_the_component():
    """The check that stops a KeyError deep inside FourierTimeFreqGPAst.setup."""
    config = base_config()
    config["ast"]["pow_spec"]["k0_freq"] = None

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    message = str(excinfo.value)
    assert "ast.pow_spec.k0_freq" in message
    assert "FourierTimeFreqGPAst" in message


def test_component_specific_enum_is_enforced():
    """`est` is a valid ast.init for the schema, but not for this component."""
    config = base_config()
    config["ast"]["init"] = "est"

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    assert "not supported by FourierTimeFreqGPAst" in str(excinfo.value)


def test_freq_int_samples_required_by_the_rfi_vis_components():
    config = base_config()
    config["rfi"]["freq_int_samples"] = None

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    assert "rfi.freq_int_samples" in str(excinfo.value)


def test_empty_component_list_is_rejected():
    config = base_config()
    config["model"]["components"] = []

    with pytest.raises(ConfigError, match=r"model\.components.*at least one"):
        validate_config(config)


def test_unimportable_component_is_reported():
    config = base_config()
    config["model"]["components"] = ["rfi_vis:NoSuchComponent"]

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)

    assert "model.components" in str(excinfo.value)
    assert "NoSuchComponent" in str(excinfo.value)


@pytest.mark.parametrize("value", [0, 1, "ones", "zeros"])
def test_numeric_init_aliases_are_accepted(value):
    """ComplexRFI._compute_init_params takes 0/1 as well as 'zeros'/'ones'."""
    config = base_config(rfi={"init": value})

    validate_config(config)  # does not raise


def test_est_init_is_accepted_by_the_fourier_gp_component():
    """FourierGPRFI reads rfi.est for `init: est`; the enum must not reject it."""
    config = base_config(
        model={"components": ["rfi_signal:FourierGPRFI"]},
        rfi={"init": "est", "mean": "zeros", "est": "rfi_est.zarr",
             "freq_pad_factor": 1, "time_pad_factor": 1},
    )

    validate_config(config)  # does not raise


def test_a_boolean_does_not_satisfy_a_numeric_choice():
    """`True == 1` in Python, so `init: true` would otherwise pass the 1 in the
    choice list and then trip the component's own `else: raise`."""
    config = base_config(rfi={"init": True})

    with pytest.raises(ConfigError, match=r"rfi\.init"):
        validate_config(config)


# ---------------------------------------------------------------------------
# Recognised-but-unread keys
# ---------------------------------------------------------------------------


def test_inert_keys_are_accepted_not_rejected():
    """`rfi.pow_spec` etc. appear in every shipped config but have no reader.

    Rejecting them as "unknown" would break every example in the repository, so
    they are in the schema and merely reported.
    """
    config = base_config()
    config["rfi"]["pow_spec"] = {"p0": 1e3, "k0s": [1e0, 1e-2], "gammas": [5, 5]}
    config.setdefault("satellites", {})["ric_std"] = 1e2

    inert = validate_config(config)  # does not raise

    assert "rfi.pow_spec.p0" in inert
    assert "satellites.ric_std" in inert


def test_inert_keys_are_still_type_checked():
    config = base_config()
    config["rfi"]["pow_spec"] = {"p0": "loads"}

    with pytest.raises(ConfigError, match=r"rfi\.pow_spec\.p0"):
        validate_config(config)


# ---------------------------------------------------------------------------
# File-level loading errors
# ---------------------------------------------------------------------------


def test_missing_file_reports_the_path(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        yaml_load(str(tmp_path / "nope.yaml"))


def test_yaml_syntax_error_reports_the_line(tmp_path):
    path = write_config(tmp_path, "data:\n  data_col: DATA\n   corr: xx\n")

    with pytest.raises(ConfigError) as excinfo:
        yaml_load(path)

    message = str(excinfo.value)
    assert "invalid YAML" in message
    assert "line 3" in message


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="empty"):
        yaml_load(write_config(tmp_path, ""))


def test_non_mapping_document_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="top-level YAML mapping"):
        yaml_load(write_config(tmp_path, "- one\n- two\n"))


# ---------------------------------------------------------------------------
# The shipped configs must keep working
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "examples/tab_target.yaml",
        "tests/data/tab_target.yaml",
        "ci/reframe/data/tab_target.yaml",
    ],
)
def test_shipped_configs_still_validate(path):
    """Guards against the schema becoming stricter than the code.

    These files are the ones users copy, CI runs and the performance benchmark
    drives, so a schema that rejects any of them is wrong by definition.
    """
    load_config(path)


def test_defaults_alone_fail_only_on_the_genuinely_required_keys(tmp_path):
    """A config that names components but sets nothing else.

    The packaged defaults cover everything except ``ast.pow_spec.p0``, which has
    no sensible static default -- so exactly one thing should be reported. This
    is the `tabascal validate-config` path: load (static layer), then the full
    check including the components.
    """
    path = write_config(
        tmp_path,
        "model:\n  components:\n" + "".join(f"    - {c}\n" for c in COMPONENTS),
    )
    config = load_config(path)  # the static layer alone has nothing to report

    with pytest.raises(ConfigError) as excinfo:
        validate_config(config, path)

    message = str(excinfo.value)
    assert "ast.pow_spec.p0" in message
    assert message.count(" : ") == 1


# ---------------------------------------------------------------------------
# load_config stays cheap
# ---------------------------------------------------------------------------


def test_load_config_does_not_import_the_component_stack(tmp_path):
    """The component layer must stay out of load time.

    Resolving ``model.components`` imports ri_kernels and every component
    module. Doing that inside ``load_config`` puts it *before*
    ``set_precision``, which is what decides ``jax_enable_x64`` relative to the
    imports that flip it -- and makes the lightweight CLI paths pay for the
    whole stack. Checked in a subprocess because the rest of this suite has
    already imported those modules.
    """
    path = write_config(
        tmp_path,
        "model:\n  components:\n" + "".join(f"    - {c}\n" for c in COMPONENTS),
    )
    code = (
        "import sys\n"
        "from tabascal.config import load_config\n"
        f"load_config({path!r})\n"
        "leaked = [m for m in ('ri_kernels', 'tabascal.components.rfi_vis',\n"
        "                      'tabascal.components.rfi_signal') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
