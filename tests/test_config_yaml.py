"""Tests for tabascal.config YAML loading and merging.

The config files use bare scientific notation (``1e6``, ``209e3``, ``3e3``) which
PyYAML's stock SafeLoader parses as *strings*. ``tabascal.config.yaml_load`` uses
a private loader subclass that parses them as floats, without polluting the global
``yaml.SafeLoader`` for the rest of the process.

The user's file is then merged onto the base config by ``deep_update``, which is
where a section header written with nothing under it is decided: yaml gives it to
the merge as ``None``, and the merge has to read that as "no override" rather
than as the new value of the section — while leaving a scalar ``null``, which is
a value the user meant, overriding as before.
"""

import pytest
import yaml

from tabascal.config import deep_update, load_config, yaml_load


#: The smallest user config: it names the model's component list and nothing
#: else, so every other value in the merged config comes from the base.
_MINIMAL_CONFIG = "model:\n  components: []\n"


def merged(tmp_path, text):
    """The merged config of a user file holding ``text``."""

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
    p.write_text("rfi:\n  k0s: [1e0, 1e-2]\n  p0: 3e3\n")

    cfg = yaml_load(p)

    assert cfg["rfi"]["k0s"] == [1.0, 0.01]
    assert cfg["rfi"]["p0"] == 3000.0


def test_yaml_load_does_not_pollute_global_safeloader():
    """Importing/using tabascal.config must not reprogram the global SafeLoader.

    The float resolver is scoped to a private subclass, so a bare
    ``yaml.safe_load`` still parses scientific notation as a string. This locks in
    the fix and guards against a regression back to the global mutation.
    """
    import tabascal.config  # noqa: F401 - ensure the module's import side effects ran

    assert isinstance(yaml.safe_load("3e3"), str)


class TestEmptySectionMerge:
    """A bare section header is no override, not an empty section.

    ``rfi:`` with nothing under it parses as ``{"rfi": None}``. Merged
    literally, that ``None`` replaces the whole base section and every default
    under it disappears, which the run then meets as ``'NoneType' object is not
    subscriptable`` deep inside ``TabConfig`` — an error about a section the
    user did write, listing nothing they got wrong.
    """

    def test_a_bare_section_header_keeps_every_default_under_it(self, tmp_path):
        """The repro. The section is left exactly as the base config ships it."""

        defaults = merged(tmp_path, _MINIMAL_CONFIG)["rfi"]

        config = merged(tmp_path, "rfi:\n" + _MINIMAL_CONFIG)

        assert config["rfi"] == defaults

    def test_the_read_that_used_to_die_on_a_bare_section(self, tmp_path):
        """``config["rfi"]["n_int_freq"]``, the first thing ``TabConfig`` asks
        the section for. The wipe originally surfaced on ``n_int_time``,
        since removed (#195)."""

        config = merged(tmp_path, "rfi:\n" + _MINIMAL_CONFIG)

        assert config["rfi"]["n_int_freq"] == 1

    def test_a_bare_nested_section_keeps_its_defaults(self, tmp_path):
        """The same rule one level down: ``ast.pow_spec`` written bare."""

        defaults = merged(tmp_path, _MINIMAL_CONFIG)["ast"]

        config = merged(tmp_path, "ast:\n  pow_spec:\n" + _MINIMAL_CONFIG)

        assert config["ast"] == defaults
        assert config["ast"]["pow_spec"]["p0"] == pytest.approx(3e3)

    def test_a_value_under_a_bare_section_still_overrides(self, tmp_path):
        """Writing the header is only inert while nothing is under it."""

        defaults = merged(tmp_path, _MINIMAL_CONFIG)["rfi"]

        rfi = merged(tmp_path, "rfi:\n  corr_time: 42\n" + _MINIMAL_CONFIG)["rfi"]

        assert rfi["corr_time"] == 42
        assert rfi == {**defaults, "corr_time": 42}

    @pytest.mark.parametrize(
        "section, key",
        [
            ("rfi", "min_elevation"),
            ("gains", "amp_mean"),
            ("satellites", "remote_max_age_days"),
        ],
    )
    def test_a_scalar_set_to_null_still_overrides(self, tmp_path, section, key):
        """``null`` on a scalar is a value the user meant, and each of these
        documents what it means: no elevation mask, no ceiling on the age of an
        orbital record. Only a section header may be inert."""

        assert merged(tmp_path, _MINIMAL_CONFIG)[section][key] is not None

        config = merged(tmp_path, f"{section}:\n  {key}: null\n" + _MINIMAL_CONFIG)

        assert config[section][key] is None

    def test_a_null_the_base_has_no_key_for_is_stored(self, tmp_path):
        """An unknown key is carried into the merged config whatever its value.

        Nothing validates the key set, so ``None`` under a name the base does
        not have has no base value to keep and is stored like any other unknown
        key. Pinned because it is the boundary of the rule above.
        """

        config = merged(tmp_path, "not_a_section:\n" + _MINIMAL_CONFIG)

        assert "not_a_section" in config
        assert config["not_a_section"] is None


class TestDeepUpdate:
    """The three cases the merge distinguishes, without a config file."""

    def test_none_over_a_mapping_keeps_the_mapping(self):
        assert deep_update({"a": {"b": 1}}, {"a": None}) == {"a": {"b": 1}}

    def test_none_over_a_scalar_replaces_it(self):
        assert deep_update({"a": 1}, {"a": None}) == {"a": None}

    def test_none_over_a_missing_key_is_stored(self):
        assert deep_update({}, {"a": None}) == {"a": None}

    def test_a_mapping_merges_key_by_key(self):
        assert deep_update({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {
            "a": {"b": 1, "c": 3}
        }
