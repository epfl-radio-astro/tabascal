"""Tests for tabascal.config YAML loading and schema defaulting.

The config files use bare scientific notation (``1e6``, ``209e3``, ``3e3``) which
PyYAML's stock SafeLoader parses as *strings*. ``tabascal.config.yaml_load`` uses
a private loader subclass that parses them as floats, without polluting the global
``yaml.SafeLoader`` for the rest of the process.

Top-level sections left empty (``data:`` or ``opt: null``) take their Pydantic
schema defaults, while unknown sections and legacy top-level sections
(``ast:``, ``rfi:``, ``gains:``, ``satellites:``) are rejected by
``load_config``.
"""

import pytest
import yaml

from tabascal.config import load_config, yaml_load
from tabascal.schema import ConfigValidationError, RunConfig

_MINIMAL_CONFIG = "model:\n  components: []\n"


def loaded(tmp_path, text) -> RunConfig:
    path = tmp_path / "user.yaml"
    path.write_text(text)
    return load_config(str(path))


def test_yaml_load_parses_bare_scientific_notation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 3e3\nb: 1e6\nc: 209e3\nd: 1e-2\ne: 1.227e9\nf: 1e0\n")

    cfg = yaml_load(p)

    assert cfg == {
        "a": 3000.0,
        "b": 1e6,
        "c": 209000.0,
        "d": 0.01,
        "e": 1.227e9,
        "f": 1.0,
    }
    assert all(isinstance(v, float) for v in cfg.values())


def test_yaml_load_handles_nested_and_list_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  components:\n    - rfi_signal.ComplexRFIVarAnt:\n        corr_freq_hz: 1e6\n        gammas: [3e0, 1e-2]\n")

    cfg = yaml_load(p)

    entry = cfg["model"]["components"][0]["rfi_signal.ComplexRFIVarAnt"]
    assert entry["corr_freq_hz"] == 1e6
    assert entry["gammas"] == [3.0, 0.01]


def test_yaml_load_does_not_pollute_global_safeloader():
    """Importing/using tabascal.config must not reprogram the global SafeLoader."""
    import tabascal.config  # noqa: F401 - ensure the module's import side effects ran

    assert isinstance(yaml.safe_load("3e3"), str)


class TestEmptySectionsAndDefaults:
    """Bare section headers and null component bodies resolve to schema defaults."""

    @pytest.mark.parametrize("section", ["data", "plots", "opt", "satellites"])
    def test_a_bare_top_level_section_keeps_its_defaults(self, tmp_path, section):
        defaults = getattr(loaded(tmp_path, _MINIMAL_CONFIG), section)
        config = loaded(tmp_path, f"{section}:\n" + _MINIMAL_CONFIG)
        assert getattr(config, section) == defaults

    def test_a_bare_component_mapping_takes_its_defaults(self, tmp_path):
        config = loaded(
            tmp_path,
            "model:\n  components:\n    - ast_vis.GPVisAst:\n",
        )
        (spec,) = config.model.components
        assert spec.ref == "ast_vis.GPVisAst"
        assert spec.config.std_jy == "data"
        assert spec.config.freq_pad_factor == 2.0

    def test_a_scalar_set_to_null_overrides_its_default(self, tmp_path):
        config = loaded(
            tmp_path,
            "model:\n  components:\n"
            "    - trajectory.FixedOrbit:\n"
            "        min_elevation_deg: null\n"
            "satellites:\n"
            "  norad_ids: [25544]\n"
            "  remote_max_age_days: null\n",
        )
        traj = config.model.components[0].config
        assert traj.min_elevation_deg is None
        assert config.satellites.remote_max_age_days is None

    @pytest.mark.parametrize("section", ["not_a_section", "rfi", "ast", "gains"])
    def test_an_unknown_or_legacy_section_is_rejected(self, tmp_path, section):
        with pytest.raises(ConfigValidationError, match=section):
            loaded(tmp_path, f"{section}:\n" + _MINIMAL_CONFIG)
