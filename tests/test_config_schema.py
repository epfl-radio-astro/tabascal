"""Tests for the Pydantic v2 compartmentalised configuration schema (tabascal.schema)."""

from pathlib import Path

import pytest
import yaml

from tabascal.config import load_config, yaml_load
from tabascal.run_dir import absolutise_inputs, input_paths
from tabascal.schema import (
    ConfigValidationError,
    EmptyComponentConfig,
    ModelConfig,
    RunConfig,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _minimal_raw(components=None, **sections):
    raw = {"model": {"components": components if components is not None else []}}
    raw.update(sections)
    return raw


# ---------------------------------------------------------------------------
# Shipped configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relpath",
    [
        "examples/tab_target.yaml",
        "tests/data/tab_target.yaml",
        "ci/reframe/data/tab_target_analytic.yaml",
        "ci/reframe/data/tab_target_riemann.yaml",
    ],
)
def test_shipped_configs_validate_cleanly(relpath):
    cfg = load_config(str(REPO_ROOT / relpath))
    assert isinstance(cfg, RunConfig)
    assert len(cfg.model.components) >= 1


# ---------------------------------------------------------------------------
# Top-level structure and legacy rejection
# ---------------------------------------------------------------------------


class TestTopLevelSchema:

    def test_minimal_config_populates_all_defaults(self):
        cfg = validate_config(_minimal_raw())
        assert cfg.model.precision == "single"
        assert cfg.model.components == []
        assert cfg.data.data_col == "DATA"
        assert cfg.data.corr == "xx"
        assert cfg.data.flags is False
        assert cfg.plots.prior_samples == 100
        assert cfg.inference.opt is True
        assert cfg.inference.effective_seed == 1
        assert cfg.opt.epsilon == 1e-2
        assert cfg.opt.max_iter == 500
        assert cfg.opt.dual_run is True
        assert cfg.opt.init_jitter is None

    def test_null_top_level_sections_take_defaults(self):
        cfg = validate_config(
            {
                "model": {"components": []},
                "data": None,
                "plots": None,
                "inference": None,
                "opt": None,
            }
        )
        assert cfg.data.corr == "xx"
        assert cfg.opt.max_iter == 500

    @pytest.mark.parametrize("legacy_section", ["ast", "rfi", "gains", "satellites"])
    def test_legacy_top_level_sections_are_rejected(self, legacy_section):
        with pytest.raises(ConfigValidationError, match=legacy_section):
            validate_config(_minimal_raw(**{legacy_section: {}}))

    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="bogus_section"):
            validate_config(_minimal_raw(bogus_section=123))

    def test_unknown_key_in_top_level_section_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="not_a_data_key"):
            validate_config(_minimal_raw(data={"not_a_data_key": 1}))

    def test_missing_model_section_is_rejected(self):
        with pytest.raises(ConfigValidationError):
            validate_config({})

    def test_non_mapping_root_is_rejected(self):
        with pytest.raises(ConfigValidationError):
            validate_config(["not", "a", "dict"])


# ---------------------------------------------------------------------------
# Component entry syntax
# ---------------------------------------------------------------------------


class TestComponentEntrySyntax:

    @pytest.mark.parametrize(
        "entry",
        [
            "gains.UnitaryGains",
            "gains:UnitaryGains",
            {"gains.UnitaryGains": None},
            {"gains.UnitaryGains": {}},
        ],
        ids=["dot-bare", "colon-bare", "null-mapping", "empty-mapping"],
    )
    def test_bare_and_empty_component_entries_parse(self, entry):
        cfg = validate_config(_minimal_raw([entry]))
        (spec,) = cfg.model.components
        assert spec.ref == "gains.UnitaryGains"
        assert isinstance(spec.config, EmptyComponentConfig)

    def test_multi_key_component_mapping_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="single-key"):
            validate_config(
                _minimal_raw([{"gains.UnitaryGains": {}, "ast_vis.GPVisAst": {}}])
            )

    def test_unknown_component_class_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="DoesNotExist"):
            validate_config(_minimal_raw(["gains.DoesNotExist"]))

    def test_unknown_field_on_component_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="gamma"):
            validate_config(
                _minimal_raw([{"ast_vis.GPVisAst": {"gamma": [3, 3]}}])
            )

    def test_non_empty_config_on_unitary_gains_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="amp_std"):
            validate_config(
                _minimal_raw([{"gains.UnitaryGains": {"amp_std": 10}}])
            )


# ---------------------------------------------------------------------------
# Per-component schemas & strict scalar checks
# ---------------------------------------------------------------------------


class TestComponentSchemas:

    def test_fixed_orbit_defaults_and_validation(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                    "rfi_signal.ComplexRFIVarAnt",
                    "rfi_vis.AnalyticVis",
                ]
            )
        )
        traj = cfg.model.components[0].config
        assert traj.norad_ids == [25544]
        assert traj.min_elevation == 0.0
        assert traj.path_order == 3
        assert traj.path_nodes is None
        assert traj.effective_path_nodes == 13
        assert traj.remote_max_age_days == 3.0
        assert traj.cache_reuse_max_age_days == 1.0

    @pytest.mark.parametrize(
        "bad_fields",
        [
            {"norad_ids": [25544], "path_order": 4},
            {"norad_ids": [25544], "path_order": -1},
            {"norad_ids": [25544], "path_order": True},
            {"norad_ids": [25544], "path_nodes": 4},  # even
            {"norad_ids": [25544], "path_nodes": 3},  # <= path_order
            {"norad_ids": [25544], "min_elevation": True},
            {"norad_ids": [25544], "min_elevation": float("nan")},
            {"norad_ids": [25544], "remote_max_age_days": 1, "cache_reuse_max_age_days": 2},
            {"norad_ids": [True]},
            {"norad_ids": [-5]},
        ],
    )
    def test_fixed_orbit_rejects_invalid_fields(self, bad_fields):
        with pytest.raises(ConfigValidationError):
            validate_config(_minimal_raw([{"trajectory.FixedOrbit": bad_fields}]))

    def test_rfi_signal_defaults_differ_between_var_and_const_ant(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                    "rfi_signal.ComplexRFIVarAnt",
                    "rfi_vis.AnalyticVis",
                ]
            )
        )
        var_cfg = cfg.model.components[1].config
        assert var_cfg.gammas == (3.0, 3.0)
        assert var_cfg.cutoff == 1e-9

        cfg2 = validate_config(
            _minimal_raw(
                [
                    {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                    "rfi_signal.ComplexRFIConstAnt",
                    "rfi_vis.AnalyticVis",
                ]
            )
        )
        const_cfg = cfg2.model.components[1].config
        assert const_cfg.gammas == (100.0, 100.0)
        assert const_cfg.cutoff == 1e-6

    @pytest.mark.parametrize(
        "bad_fields",
        [
            {"std": 0},
            {"std": -1.0},
            {"std": True},
            {"std": float("inf")},
            {"corr_freq": 0},
            {"corr_time": -5},
            {"cutoff": 1.0},
            {"cutoff": 0.0},
            {"gammas": [3.0]},
            {"gammas": [0.0, 3.0]},
            {"freq_pad_factor": 0.5},
            {"init": "bogus"},
            {"mean": "bogus"},
        ],
    )
    def test_rfi_signal_rejects_invalid_fields(self, bad_fields):
        with pytest.raises(ConfigValidationError):
            validate_config(
                _minimal_raw(
                    [
                        {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                        {"rfi_signal.ComplexRFIVarAnt": bad_fields},
                        "rfi_vis.AnalyticVis",
                    ]
                )
            )

    def test_analytic_vis_rejects_bad_options(self):
        for bad in (
            {"stencil": 5},
            {"stencil": -1},
            {"segments": 0},
            {"terms": 33},
            {"terms": True},
            {"cubic_terms": 9},
            {"scratch_mb": 0},
            {"n_int_freq": 0},
            {"backend": "jax"},
            {"time_int_factor": 1},  # fine-grid only
        ):
            with pytest.raises(ConfigValidationError):
                validate_config(
                    _minimal_raw(
                        [
                            {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                            "rfi_signal.ComplexRFIVarAnt",
                            {"rfi_vis.AnalyticVis": bad},
                        ]
                    )
                )

    def test_fixed_discrete_sky_requires_point_sources(self):
        with pytest.raises(ConfigValidationError, match="point_sources"):
            validate_config(
                _minimal_raw(["ast_signal.FixedDiscreteSky", "ast_vis.DiscreteSkyVis"])
            )

        cfg = validate_config(
            _minimal_raw(
                [
                    {
                        "ast_signal.FixedDiscreteSky": {
                            "point_sources": [{"ra": 10.0, "dec": -20.0, "I": 5.0}]
                        }
                    },
                    "ast_vis.DiscreteSkyVis",
                ]
            )
        )
        assert len(cfg.model.components[0].config.point_sources) == 1

    def test_const_gains_defaults_and_validation(self):
        cfg = validate_config(_minimal_raw(["ast_vis.GPVisAst", "gains.ConstGains"]))
        g = cfg.model.components[1].config
        assert g.amp_std == 20.0
        assert g.phase_std == 180.0
        assert g.fix_flux_scale is True
        assert g.fit == "complex"

        for bad in (
            {"amp_std": 0},
            {"phase_std": -1},
            {"amp_mean": 0},
            {"amp_std": True},
            {"fit": "invalid"},
        ):
            with pytest.raises(ConfigValidationError):
                validate_config(
                    _minimal_raw(["ast_vis.GPVisAst", {"gains.ConstGains": bad}])
                )


# ---------------------------------------------------------------------------
# Cross-component & precision validation
# ---------------------------------------------------------------------------


class TestCrossComponentValidation:

    def test_requires_double_component_rejected_under_single_precision(self):
        with pytest.raises(ConfigValidationError, match="double"):
            validate_config(
                {
                    "model": {
                        "precision": "single",
                        "components": [
                            {"trajectory.OrbitFine": {"norad_ids": [25544]}},
                            "trajectory.PhaseCalculationRFIFine",
                        ],
                    }
                }
            )

    def test_requires_double_component_accepted_under_double_precision(self):
        cfg = validate_config(
            {
                "model": {
                    "precision": "double",
                    "components": [
                        {"trajectory.OrbitFine": {"norad_ids": [25544]}},
                        "trajectory.PhaseCalculationRFIFine",
                    ],
                }
            }
        )
        assert cfg.model.precision == "double"

    def test_component_dependency_order_checked_at_load_time(self):
        with pytest.raises(ConfigValidationError, match="ast_radec"):
            validate_config(_minimal_raw(["ast_vis.DiscreteSkyVis"]))

    def test_grid_mismatch_rejected_at_load_time(self):
        with pytest.raises(ConfigValidationError, match="fine grid"):
            validate_config(
                _minimal_raw(
                    [
                        {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                        "rfi_signal.ComplexRFIVarAntFine",
                        "rfi_vis.AnalyticVis",
                    ]
                )
            )

    def test_const_gains_free_flux_scale_requires_fixed_discrete_sky(self):
        with pytest.raises(ConfigValidationError, match="FixedDiscreteSky"):
            validate_config(
                _minimal_raw(
                    [
                        "ast_vis.GPVisAst",
                        {"gains.ConstGains": {"fix_flux_scale": False}},
                    ]
                )
            )

    def test_const_gains_free_flux_scale_allowed_for_phase_only_fit(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    "ast_vis.GPVisAst",
                    {"gains.ConstGains": {"fix_flux_scale": False, "fit": "phase"}},
                ]
            )
        )
        assert cfg.model.components[1].config.fix_flux_scale is False

    def test_duplicate_orbit_source_rejected(self):
        with pytest.raises(ConfigValidationError, match="at most one"):
            validate_config(
                _minimal_raw(
                    [
                        {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                        {"trajectory.FixedOrbit": {"norad_ids": [25544]}},
                    ]
                )
            )


# ---------------------------------------------------------------------------
# Input path discovery & YAML round-trip
# ---------------------------------------------------------------------------


class TestPathsAndRoundTrip:

    def test_schema_driven_input_paths_and_absolutise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = validate_config(
            {
                "data": {
                    "ms_path": "obs.ms",
                    "gain_table": ["cal1.G", "cal2.B"],
                    "noise": "noise.npz",
                },
                "model": {
                    "components": [
                        {
                            "trajectory.FixedOrbit": {
                                "norad_ids_path": "sats.txt",
                                "extra_orbit_dir": "tles",
                            }
                        },
                        {
                            "rfi_signal.ComplexRFIConstAnt": {
                                "init": "est",
                                "est": "lc.npz",
                            }
                        },
                        "rfi_vis.AnalyticVis",
                        {
                            "ast_signal.FixedDiscreteSky": {
                                "point_sources": "sky.osm",
                            }
                        },
                        "ast_vis.DiscreteSkyVis",
                        {
                            "gains.ConstGains": {
                                "init": "gains.npz",
                            }
                        },
                    ]
                },
            }
        )

        absolutise_inputs(cfg)
        paths = input_paths(cfg)
        base = tmp_path.resolve()
        assert paths["data.gain_table[0]"] == str(base / "cal1.G")
        assert paths["data.gain_table[1]"] == str(base / "cal2.B")
        assert paths["data.noise"] == str(base / "noise.npz")
        assert paths["trajectory.FixedOrbit.norad_ids_path"] == str(base / "sats.txt")
        assert paths["trajectory.FixedOrbit.extra_orbit_dir"] == str(base / "tles")
        assert paths["rfi_signal.ComplexRFIConstAnt.est"] == str(base / "lc.npz")
        assert paths["ast_signal.FixedDiscreteSky.point_sources"] == str(base / "sky.osm")
        assert paths["gains.ConstGains.init"] == str(base / "gains.npz")

    def test_yaml_round_trip_preserves_validated_config(self, tmp_path):
        original = load_config(str(REPO_ROOT / "examples/tab_target.yaml"))
        dumped = tmp_path / "roundtrip.yaml"
        dumped.write_text(yaml.safe_dump(original.to_yaml_dict(), sort_keys=False))
        reloaded = load_config(str(dumped))
        assert reloaded.to_yaml_dict() == original.to_yaml_dict()
