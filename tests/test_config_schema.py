"""Validation of the config YAML against the parameters components declare.

The config file used to be deep-merged into a packaged base file with nothing
checked, so a typo merged cleanly and ran with a default, and a missing key
surfaced as a bare ``KeyError`` deep inside a component's ``setup`` -- after the
Measurement Set read and the TLE preflight. These tests pin the replacement: one
error listing every problem, raised before anything expensive happens.
"""

import pytest
import yaml

from tabascal.config import TabConfig, load_config
from tabascal.config_schema import (
    FROM_DATA,
    REQUIRED,
    ConfigError,
    Param,
    collect_params,
    component_param_owners,
    validate_config,
)
from tabascal.imports import import_components


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAMS = {
    "model.components": Param(types=(list,), item=(str,), default=REQUIRED),
    "sec.flag": Param(types=(bool,), default=False),
    "sec.count": Param(types=(int,), default=3, ge=1),
    "sec.rate": Param(types=(int, float), default=1.0, gt=0),
    "sec.mode": Param(choices=("fast", "slow"), default="fast"),
    "sec.mean": Param(types=(int, float), choices=("data",), default=0),
    "sec.derived": Param(types=(int, float), default=FROM_DATA, gt=0),
    "sec.limit": Param(types=(int, float), default=7, ge=0, null_ok=True),
    "sec.nested.value": Param(types=(str,), default="x"),
}


def validate(config, **kwargs):
    kwargs.setdefault("elsewhere", {})
    return validate_config(config, PARAMS, **kwargs)


def problems(config, **kwargs):
    with pytest.raises(ConfigError) as exc:
        validate(config, **kwargs)
    return str(exc.value)


MINIMAL = {"model": {"components": ["a:B"]}}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_absent_keys_get_their_declared_default(self):
        """The result is complete, so downstream config[...][...] never raises."""
        resolved = validate(MINIMAL)

        assert resolved["sec"] == {
            "flag": False,
            "count": 3,
            "rate": 1.0,
            "mode": "fast",
            "mean": 0,
            "derived": None,
            "limit": 7,
            "nested": {"value": "x"},
        }

    def test_an_explicit_null_takes_the_default(self):
        """`key:` with nothing after it means "unset", not "None"."""
        resolved = validate({**MINIMAL, "sec": {"count": None, "mode": None}})

        assert resolved["sec"]["count"] == 3
        assert resolved["sec"]["mode"] == "fast"

    def test_a_from_data_parameter_resolves_to_none(self):
        """Its default needs the measurement set, so the component fills it in."""
        assert validate(MINIMAL)["sec"]["derived"] is None

    def test_null_ok_keeps_an_explicit_null(self):
        """For these, null is a value ("no limit"), not an omission."""
        resolved = validate({**MINIMAL, "sec": {"limit": None}})

        assert resolved["sec"]["limit"] is None
        # ... and omitting it still gives the default.
        assert validate(MINIMAL)["sec"]["limit"] == 7

    def test_configured_values_are_kept(self):
        resolved = validate({**MINIMAL, "sec": {"count": 9, "nested": {"value": "y"}}})

        assert resolved["sec"]["count"] == 9
        assert resolved["sec"]["nested"]["value"] == "y"

    def test_an_empty_section_is_not_an_error(self):
        """A section header with nothing under it parses as None."""
        assert validate({**MINIMAL, "sec": None})["sec"]["count"] == 3


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

class TestRejections:

    def test_a_missing_required_parameter_is_named(self):
        assert "model.components: required, but not set" in problems({})

    def test_a_wrong_type_is_reported_with_the_value(self):
        report = problems({**MINIMAL, "sec": {"count": "many"}})

        assert "sec.count: expected an integer >= 1, got 'many'" in report

    def test_a_bool_never_satisfies_a_numeric_parameter(self):
        """isinstance(True, int) is True in Python; a config is not so forgiving."""
        assert "sec.count" in problems({**MINIMAL, "sec": {"count": True}})

    def test_an_out_of_range_value_is_reported(self):
        assert "sec.rate: must be > 0" in problems({**MINIMAL, "sec": {"rate": 0}})

    def test_a_value_outside_an_enum_is_reported(self):
        report = problems({**MINIMAL, "sec": {"mode": "medium"}})

        assert "sec.mode: expected one of 'fast', 'slow', got 'medium'" in report

    def test_an_enum_is_not_satisfied_by_any_string(self):
        """A parameter that is a number *or* a keyword still rejects other strings."""
        assert "sec.mean" in problems({**MINIMAL, "sec": {"mean": "truth"}})
        assert validate({**MINIMAL, "sec": {"mean": "data"}})["sec"]["mean"] == "data"
        assert validate({**MINIMAL, "sec": {"mean": 2.5}})["sec"]["mean"] == 2.5

    def test_a_bad_list_element_is_reported(self):
        report = problems({"model": {"components": ["a:B", 7]}})

        assert "model.components" in report and "7" in report

    def test_a_section_that_is_not_a_mapping_is_reported(self):
        assert "sec: expected a mapping" in problems({**MINIMAL, "sec": "nope"})

    def test_a_config_that_is_not_a_mapping_is_rejected(self):
        with pytest.raises(ConfigError, match="mapping of sections"):
            validate(["not", "a", "mapping"])

    def test_every_problem_is_reported_at_once(self):
        """Four mistakes cost one run to find, not four."""
        report = problems({"sec": {"count": "many", "rate": -1, "mode": "medium"}})

        assert "4 problems found" in report
        for expected in ("model.components", "sec.count", "sec.rate", "sec.mode"):
            assert expected in report

    def test_one_problem_is_not_reported_in_the_plural(self):
        assert "1 problem found" in problems({})

    def test_the_source_file_is_named(self):
        assert "tab_target.yaml" in problems({}, source="tab_target.yaml")


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------

class TestUnknownKeys:

    def test_an_unknown_key_is_rejected(self):
        assert "sec.extra: unknown key" in problems({**MINIMAL, "sec": {"extra": 1}})

    def test_an_unknown_section_is_rejected(self):
        assert "other: unknown key" in problems({**MINIMAL, "other": {"a": 1}})

    def test_a_typo_suggests_the_key_that_was_meant(self):
        assert "did you mean 'sec.count'?" in problems({**MINIMAL, "sec": {"cont": 3}})

    def test_a_key_in_the_wrong_section_is_still_found(self):
        """Matched on the leaf name too, so a misplaced key is not just "unknown"."""
        report = problems({**MINIMAL, "sec": {"nested": {"count": 3}}})

        assert "sec.nested.count: unknown key" in report
        assert "did you mean 'sec.count'?" in report

    def test_a_key_declared_by_an_unselected_component_says_so(self):
        report = problems(
            {**MINIMAL, "ast": {"init": "prior"}},
            elsewhere={"ast.init": "ast_vis:GPVisAst"},
        )

        assert "ast: unknown key" in report or "ast.init" in report
        assert "GPVisAst" in report

    def test_strict_false_leaves_undeclared_keys_alone(self):
        """The bootstrap pass validates two keys and must not touch the rest."""
        config = {
            "model": {"components": ["a:B"], "name": "kept"},
            "whatever": {"a": 1},
        }

        resolved = validate_config(
            config,
            {"model.components": PARAMS["model.components"]},
            strict=False,
        )

        assert resolved["whatever"] == {"a": 1}
        assert resolved["model"]["components"] == ["a:B"]
        # A sibling of a key that *was* resolved survives too.
        assert resolved["model"]["name"] == "kept"

    def test_strict_false_keeps_deeply_nested_siblings(self):
        """The overlay is recursive, so nothing below a resolved key is dropped."""
        config = {"a": {"b": {"kept": 1}}}

        resolved = validate_config(
            config, {"a.b.checked": Param(types=(int,), default=7)}, strict=False
        )

        assert resolved["a"]["b"] == {"kept": 1, "checked": 7}


# ---------------------------------------------------------------------------
# Collecting declarations
# ---------------------------------------------------------------------------

class TestCollectParams:

    def test_a_subclass_extends_its_bases_declarations(self):
        class Base:
            config_params = {"a.x": Param(types=(int,), default=1)}

        class Derived(Base):
            config_params = {"a.y": Param(types=(int,), default=2)}

        assert set(collect_params(Derived)) == {"a.x", "a.y"}

    def test_a_subclass_may_override_a_base_declaration(self):
        class Base:
            config_params = {"a.x": Param(types=(int,), default=1)}

        class Derived(Base):
            config_params = {"a.x": Param(types=(int,), default=5)}

        assert collect_params(Derived)["a.x"].default == 5

    def test_two_owners_may_declare_the_same_parameter_identically(self):
        """Every rfi_vis component reads rfi.freq_int_samples; that is fine."""
        shared = Param(types=(int,), default=1)

        class One:
            config_params = {"a.x": shared}

        class Two:
            config_params = {"a.x": Param(types=(int,), default=1)}

        assert collect_params(One, Two)["a.x"] == shared

    def test_conflicting_declarations_are_a_developer_error(self):
        class One:
            config_params = {"a.x": Param(types=(int,), default=1)}

        class Two:
            config_params = {"a.x": Param(types=(str,), default="1")}

        with pytest.raises(ConfigError, match="Conflicting declarations"):
            collect_params(One, Two)

    def test_a_path_cannot_be_both_a_section_and_a_parameter(self):
        params = {"a.x": Param(types=(int,)), "a.x.y": Param(types=(int,))}

        with pytest.raises(ConfigError, match="section"):
            validate_config({}, params)

    def test_every_declared_component_parameter_is_discoverable(self):
        owners = component_param_owners()

        assert owners["ast.pow_spec.p0"] == "ast_vis:GPVisAst"
        assert owners["gains.amp_std"].startswith("gains:")
        assert owners["rfi.corr_time"].startswith("rfi_signal:")


# ---------------------------------------------------------------------------
# The shipped configs
# ---------------------------------------------------------------------------

class TestShippedConfigs:
    """The configs in the repo must validate against their own component list.

    This is what keeps a declaration and the files that exercise it from drifting
    apart the way the base config and its readers did.
    """

    @pytest.mark.parametrize(
        "path",
        ["tests/data/tab_target.yaml", "examples/tab_target.yaml", "ci/reframe/data/tab_target.yaml"],
    )
    def test_shipped_config_is_valid(self, path, request):
        config_path = request.config.rootpath / path
        config = load_config(str(config_path))
        classes = import_components(config["model"]["components"])

        resolved = validate_config(
            config,
            collect_params(TabConfig, *classes),
            source=path,
            elsewhere=component_param_owners(),
        )

        # Defaults are filled in, so the run's direct indexing is safe.
        assert resolved["data"]["data_col"] == "DATA"
        assert resolved["rfi"]["min_elevation"] == 0

    def test_a_resolved_config_round_trips_through_yaml(self):
        """The run dumps the resolved config beside its results; keep it dumpable."""
        resolved = validate(MINIMAL)

        assert yaml.safe_load(yaml.dump(resolved)) == resolved


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def write(self, tmp_path, text):
        path = tmp_path / "c.yaml"
        path.write_text(text)
        return str(path)

    def test_the_bootstrap_keys_are_checked(self, tmp_path):
        path = self.write(tmp_path, "model:\n  precision: quadruple\n  components: [a:B]\n")

        with pytest.raises(ConfigError, match="model.precision"):
            load_config(path)

    def test_missing_components_is_caught_before_anything_else(self, tmp_path):
        path = self.write(tmp_path, "data:\n  data_col: DATA\n")

        with pytest.raises(ConfigError, match="model.components"):
            load_config(path)

    def test_the_rest_of_the_file_is_left_untouched(self, tmp_path):
        """Only the components can say what the other sections should contain."""
        path = self.write(tmp_path, "model:\n  components: [a:B]\nnonsense:\n  q: 1\n")

        config = load_config(path)

        assert config["nonsense"] == {"q": 1}
        assert config["model"]["precision"] == "single"

    def test_a_missing_file_is_a_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="could not be read"):
            load_config(str(tmp_path / "absent.yaml"))

    def test_a_syntax_error_keeps_its_original_message(self, tmp_path):
        """The IOError this replaced hid the parser's line and column."""
        path = self.write(tmp_path, "model:\n  components: [a:B\n")

        with pytest.raises(ConfigError, match="could not be parsed"):
            load_config(path)
