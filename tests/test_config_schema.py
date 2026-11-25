"""Tests for the Pydantic v2 compartmentalised configuration schema (tabascal.schema)."""

from pathlib import Path

import pytest
import yaml

from tabascal.components import in_tree_components
from tabascal.config import load_config
from tabascal.run_dir import absolutise_inputs, input_paths
from tabascal.schema import (
    ConfigValidationError,
    EmptyComponentConfig,
    ModelConfig,
    RunConfig,
    TOP_LEVEL_SECTIONS,
    validate_config,
)
from tabascal.scripts.migrate_config import build_parser as build_migrate_parser
from tabascal.scripts.migrate_config import migrate, run as run_migrate
from tabascal.scripts.run_tabascal import build_parser as build_cli_parser

REPO_ROOT = Path(__file__).resolve().parent.parent


def _minimal_raw(components=None, **sections):
    raw = {"model": {"components": components if components is not None else []}}
    raw.update(sections)
    return raw


# ---------------------------------------------------------------------------
# Shipped configs & schema self-documentation
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


def test_every_schema_field_has_a_description():
    """Every section field and every component Config field documents itself."""
    for cls in (RunConfig, ModelConfig, *TOP_LEVEL_SECTIONS.values()):
        for name, field in cls.model_fields.items():
            assert field.description and field.description.strip(), (
                f"{cls.__name__}.{name} has no description"
            )
    for ref, comp_cls in in_tree_components().items():
        for name, field in comp_cls.Config.model_fields.items():
            assert field.description and field.description.strip(), (
                f"{ref} ({comp_cls.Config.__name__}.{name}) has no description"
            )


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
        assert cfg.satellites.norad_ids == []
        assert cfg.satellites.remote_max_age_days == 3.0
        assert cfg.satellites.cache_reuse_max_age_days == 1.0

    def test_null_top_level_sections_take_defaults(self):
        cfg = validate_config(
            {
                "model": {"components": []},
                "data": None,
                "plots": None,
                "inference": None,
                "opt": None,
                "satellites": None,
            }
        )
        assert cfg.data.corr == "xx"
        assert cfg.opt.max_iter == 500
        assert cfg.satellites.norad_ids == []

    @pytest.mark.parametrize("legacy_section", ["ast", "rfi", "gains"])
    def test_legacy_top_level_sections_are_rejected_with_migration_hint(self, legacy_section):
        with pytest.raises(ConfigValidationError, match=r"migrate-config") as exc:
            validate_config(_minimal_raw(**{legacy_section: {}}))
        assert legacy_section in str(exc.value)

    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="bogus_section"):
            validate_config(_minimal_raw(bogus_section=123))

    def test_typo_in_top_level_section_suggests_close_match(self):
        with pytest.raises(ConfigValidationError, match="Did you mean 'satellites'"):
            validate_config(_minimal_raw(satelites={}))

    def test_unknown_key_in_top_level_section_is_rejected_with_typo_hint(self):
        with pytest.raises(ConfigValidationError, match="not_a_data_key"):
            validate_config(_minimal_raw(data={"not_a_data_key": 1}))
        with pytest.raises(ConfigValidationError, match="Did you mean 'data_col'"):
            validate_config(_minimal_raw(data={"data_cl": "DATA"}))

    def test_missing_model_section_is_rejected(self):
        with pytest.raises(ConfigValidationError):
            validate_config({})

    def test_non_mapping_root_is_rejected(self):
        with pytest.raises(ConfigValidationError):
            validate_config(["not", "a", "dict"])

    @pytest.mark.parametrize(
        "bad_sat",
        [
            {"remote_max_age_days": 1, "cache_reuse_max_age_days": 2},
            {"norad_ids": [True]},
            {"norad_ids": [-5]},
            {"norad_ids": "25544"},
            {"extra_orbit_max_age_days": -1},
        ],
    )
    def test_satellites_section_rejects_invalid_fields(self, bad_sat):
        with pytest.raises(ConfigValidationError):
            validate_config(_minimal_raw(satellites=bad_sat))


# ---------------------------------------------------------------------------
# Component entry syntax, hints, and multi-error accumulation
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

    def test_typo_on_component_field_suggests_close_match(self):
        with pytest.raises(ConfigValidationError, match="Did you mean 'gammas'") as exc:
            validate_config(
                _minimal_raw([{"ast_vis.GPVisAst": {"gamma": [3, 3]}}])
            )
        assert "gamma" in str(exc.value)

    def test_option_belonging_to_another_component_names_its_owner(self):
        with pytest.raises(ConfigValidationError, match="rfi_vis.AnalyticVis"):
            validate_config(
                _minimal_raw([{"trajectory.FixedOrbit": {"n_int_freq": 2}}])
            )

    def test_satellite_option_on_trajectory_points_to_satellites_section(self):
        with pytest.raises(ConfigValidationError, match="'satellites'"):
            validate_config(
                _minimal_raw([{"trajectory.FixedOrbit": {"norad_ids": [25544]}}])
            )

    def test_non_empty_config_on_unitary_gains_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="gains.ConstGains"):
            validate_config(
                _minimal_raw([{"gains.UnitaryGains": {"amp_std": 10}}])
            )

    def test_errors_across_multiple_components_and_sections_are_reported_together(self):
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(
                {
                    "model": {
                        "components": [
                            {"trajectory.FixedOrbit": {"path_order": 9}},
                            {"rfi_signal.ComplexRFIVarAnt": {"corr_tme": 24}},
                            {"rfi_vis.AnalyticVis": {"stencil": 99}},
                        ]
                    },
                    "data": {"freq": -10},
                }
            )
        msg = str(exc.value)
        assert "path_order" in msg
        assert "corr_tme" in msg
        assert "Did you mean 'corr_time'" in msg
        assert "stencil" in msg
        assert "data.freq" in msg


# ---------------------------------------------------------------------------
# Per-component schemas & strict scalar checks
# ---------------------------------------------------------------------------


class TestComponentSchemas:

    def test_fixed_orbit_defaults_and_validation(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    "trajectory.FixedOrbit",
                    "rfi_signal.ComplexRFIVarAnt",
                    "rfi_vis.AnalyticVis",
                ],
                satellites={"norad_ids": [25544]},
            )
        )
        traj = cfg.model.components[0].config
        assert traj.min_elevation == 0.0
        assert traj.path_order == 3
        assert traj.path_nodes is None
        assert traj.effective_path_nodes == 13
        assert cfg.satellites.norad_ids == [25544]
        assert cfg.satellites.remote_max_age_days == 3.0
        assert cfg.satellites.cache_reuse_max_age_days == 1.0

    @pytest.mark.parametrize(
        "bad_fields",
        [
            {"path_order": 4},
            {"path_order": -1},
            {"path_order": True},
            {"path_nodes": 4},  # even
            {"path_nodes": 3},  # <= path_order
            {"min_elevation": True},
            {"min_elevation": float("nan")},
        ],
    )
    def test_fixed_orbit_rejects_invalid_fields(self, bad_fields):
        with pytest.raises(ConfigValidationError):
            validate_config(_minimal_raw([{"trajectory.FixedOrbit": bad_fields}]))

    def test_rfi_signal_defaults_differ_between_var_and_const_ant(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    "trajectory.FixedOrbit",
                    "rfi_signal.ComplexRFIVarAnt",
                    "rfi_vis.AnalyticVis",
                ],
                satellites={"norad_ids": [25544]},
            )
        )
        var_cfg = cfg.model.components[1].config
        # null: the component's own exponents (BaseGPRFI.default_gammas)
        assert var_cfg.gammas is None
        assert var_cfg.cutoff == 1e-9
        assert var_cfg.corr_freq == 1e6
        assert var_cfg.corr_time == 24.0

        cfg2 = validate_config(
            _minimal_raw(
                [
                    "trajectory.FixedOrbit",
                    "rfi_signal.ComplexRFIConstAnt",
                    "rfi_vis.AnalyticVis",
                ],
                satellites={"norad_ids": [25544]},
            )
        )
        const_cfg = cfg2.model.components[1].config
        assert const_cfg.gammas is None
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
                        "trajectory.FixedOrbit",
                        {"rfi_signal.ComplexRFIVarAnt": bad_fields},
                        "rfi_vis.AnalyticVis",
                    ],
                    satellites={"norad_ids": [25544]},
                )
            )

    def test_riemann_vis_fine_rejects_variable_only_time_bins(self):
        for bad in ({"min_time_bins": 2}, {"max_time_bins": 10}):
            with pytest.raises(ConfigValidationError, match="RiemannVisVariableFine"):
                validate_config(
                    _minimal_raw(
                        [
                            "trajectory.FixedOrbitFine",
                            "rfi_signal.ComplexRFIVarAntFine",
                            {"rfi_vis.RiemannVisFine": bad},
                        ]
                    )
                )

    def test_riemann_vis_variable_fine_accepts_and_checks_time_bins(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    "trajectory.FixedOrbitFine",
                    "rfi_signal.ComplexRFIVarAntFine",
                    {"rfi_vis.RiemannVisVariableFine": {"min_time_bins": 2, "max_time_bins": 12}},
                ]
            )
        )
        vis_cfg = cfg.model.components[2].config
        assert vis_cfg.min_time_bins == 2
        assert vis_cfg.max_time_bins == 12

        with pytest.raises(ConfigValidationError, match="min_time_bins"):
            validate_config(
                _minimal_raw(
                    [
                        "trajectory.FixedOrbitFine",
                        "rfi_signal.ComplexRFIVarAntFine",
                        {"rfi_vis.RiemannVisVariableFine": {"min_time_bins": 10, "max_time_bins": 5}},
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
                            "trajectory.FixedOrbit",
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

    def test_inline_sources_read_numeric_names_and_quoted_numbers(self):
        """Catalogue-style entries that loaded before the schema move still load."""
        cfg = validate_config(
            _minimal_raw(
                [
                    {
                        "ast_signal.FixedDiscreteSky": {
                            "point_sources": [
                                {"name": 1934, "ra": "10.5", "dec": -20, "I": "750.0",
                                 "alpha": " -0.8 ", "ref_freq_mhz": "1400"},
                            ]
                        }
                    },
                    "ast_vis.DiscreteSkyVis",
                ]
            )
        )
        (source,) = cfg.model.components[0].config.point_sources
        assert source.name == "1934"
        assert (source.ra, source.dec, source.I) == (10.5, -20.0, 750.0)
        assert (source.alpha, source.ref_freq_mhz) == (-0.8, 1400.0)
        assert source.to_oskar_row(0)["name"] == "1934"

    @pytest.mark.parametrize("value", ["nan", "inf", "lots", ""])
    def test_quoted_inline_values_must_still_be_finite_numbers(self, value):
        with pytest.raises(ConfigValidationError, match="I"):
            validate_config(
                _minimal_raw(
                    [
                        {
                            "ast_signal.FixedDiscreteSky": {
                                "point_sources": [{"ra": 1.0, "dec": 2.0, "I": value}]
                            }
                        },
                        "ast_vis.DiscreteSkyVis",
                    ]
                )
            )

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
                            "trajectory.OrbitFine",
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
                        "trajectory.OrbitFine",
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
                        "trajectory.FixedOrbit",
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
                        "trajectory.FixedOrbit",
                        "trajectory.FixedOrbit",
                    ]
                )
            )

    def test_rfi_signal_components_must_agree_on_pad_factors(self):
        with pytest.raises(ConfigValidationError, match="pad_factor"):
            validate_config(
                _minimal_raw(
                    [
                        "trajectory.FixedOrbitFine",
                        {"rfi_signal.ComplexRFIVarAntFine": {"time_pad_factor": 3.0}},
                        "rfi_signal.ComplexRFIConstAntFine",
                        "rfi_vis.RiemannVisFine",
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
                "satellites": {
                    "norad_ids_path": "sats.txt",
                    "extra_orbit_dir": "tles",
                },
                "model": {
                    "components": [
                        "trajectory.FixedOrbit",
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
        assert paths["satellites.norad_ids_path"] == str(base / "sats.txt")
        assert paths["satellites.extra_orbit_dir"] == str(base / "tles")
        assert paths["rfi_signal.ComplexRFIConstAnt.est"] == str(base / "lc.npz")
        assert paths["ast_signal.FixedDiscreteSky.point_sources"] == str(base / "sky.osm")
        assert paths["gains.ConstGains.init"] == str(base / "gains.npz")

    def test_legacy_sectioned_dict_inputs_still_discovered(self, tmp_path):
        old_dict = {
            "data": {"ms_path": str(tmp_path / "obs.ms")},
            "satellites": {"extra_orbit_dir": str(tmp_path / "orbits")},
            "rfi": {"est": str(tmp_path / "lc.npz")},
            "gains": {"init": str(tmp_path / "gains.npz")},
            "ast": {"point_sources": str(tmp_path / "sky.osm")},
        }
        paths = input_paths(old_dict)
        assert paths["satellites.extra_orbit_dir"] == str(tmp_path / "orbits")
        assert paths["rfi.est"] == str(tmp_path / "lc.npz")
        assert paths["gains.init"] == str(tmp_path / "gains.npz")
        assert paths["ast.point_sources"] == str(tmp_path / "sky.osm")

    def test_yaml_round_trip_preserves_validated_config(self, tmp_path):
        original = load_config(str(REPO_ROOT / "examples/tab_target.yaml"))
        dumped = tmp_path / "roundtrip.yaml"
        dumped.write_text(yaml.safe_dump(original.to_yaml_dict(), sort_keys=False))
        reloaded = load_config(str(dumped))
        assert reloaded.to_yaml_dict() == original.to_yaml_dict()


# ---------------------------------------------------------------------------
# Migration from sectioned layout
# ---------------------------------------------------------------------------


class TestMigration:

    def test_migrate_sectioned_config_produces_valid_new_config(self):
        old = {
            "model": {
                "precision": "single",
                "components": [
                    "trajectory:FixedOrbit",
                    "rfi_signal:ComplexRFIVarAnt",
                    "rfi_vis:AnalyticVis",
                    "ast_vis:GPVisAst",
                    "gains:UnitaryGains",
                ],
            },
            "rfi": {
                "init": "truth",
                "mean": 0,
                "min_elevation": 5,
                "n_int_freq": 1,
                "time_int_factor": 5,  # unused by AnalyticVis -> dropped with note
                "r_seed": 123,  # dropped with r_seed note
                "analytic": {"stencil": 2, "segments": 4},
                "gp_cov": {"corr_time": 24, "gammas": None},
                "cutoff": None,
            },
            "ast": {
                "init": "prior",
                "gp_cov": {"std": 1.5, "fov_deg": 5},
                "cutoff": None,
            },
            "gains": {
                "amp_std": 10,  # unused by UnitaryGains -> dropped with note
            },
            "satellites": {"norad_ids": [20452, 38833]},
            "data": {"data_col": "DATA"},
        }

        new, notes = migrate(old)
        validated = validate_config(new)
        assert validated.model.precision == "single"
        assert validated.satellites.norad_ids == [20452, 38833]
        assert validated.model.components[0].config.min_elevation == 5.0
        assert validated.model.components[1].config.init == "truth"
        assert validated.model.components[1].config.corr_time == 24.0
        assert validated.model.components[2].config.stencil == 2
        assert validated.model.components[2].config.segments == 4
        assert validated.model.components[3].config.std == 1.5
        assert validated.model.components[3].config.fov_deg == 5.0
        assert any("rfi.time_int_factor" in n for n in notes)
        assert any("rfi.r_seed" in n and "inference.seed" in n for n in notes)
        assert any("gains.amp_std" in n for n in notes)

    def test_migrate_keeps_the_nulls_the_sectioned_config_documented(self):
        """``rfi.signal_type``, ``rfi.cutoff`` and ``rfi.gp_cov.gammas`` each
        documented ``null`` as "the default", and a config that wrote it out
        still converts to one that validates, with that default."""
        old = {
            "model": {"components": ["rfi_signal:ComplexRFIVarAnt"]},
            "rfi": {
                "signal_type": None,
                "cutoff": None,
                "gp_cov": {"gammas": None, "std": None},
            },
        }

        new, _ = migrate(old)
        cfg = validate_config(new).model.components[0].config
        assert cfg.signal_type == "complex"
        assert cfg.cutoff == 1e-9
        assert cfg.gammas is None
        assert cfg.std is None

    def test_migrate_config_cli_round_trip(self, tmp_path):
        old_path = tmp_path / "old.yaml"
        new_path = tmp_path / "new.yaml"
        old_path.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "components": [
                            "ast_vis:GPVisAst",
                            "gains:UnitaryGains",
                        ]
                    },
                    "ast": {"gp_cov": {"std": 2.0}},
                }
            )
        )
        args = build_cli_parser().parse_args(
            ["migrate-config", str(old_path), "-o", str(new_path)]
        )
        assert args.command == "migrate-config"
        assert run_migrate(args) == 0
        loaded = load_config(str(new_path))
        assert loaded.model.components[0].config.std == 2.0

    def test_migrate_config_cli_returns_nonzero_when_result_invalid(self, tmp_path, capsys):
        old_path = tmp_path / "bad_old.yaml"
        old_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"components": ["ast_vis:GPVisAst"]},
                    "ast": {"gp_cov": {"std": -5}},
                }
            )
        )
        args = build_migrate_parser().parse_args([str(old_path)])
        assert run_migrate(args) == 1
        assert "does not validate" in capsys.readouterr().err
