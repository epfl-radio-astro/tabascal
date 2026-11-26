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
        assert cfg.data.exclude_flagged is False
        assert cfg.plots.prior_samples == 100
        assert cfg.seed == 1
        assert cfg.opt.epsilon == 1e-2
        assert cfg.opt.max_iter == 1000
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
                "opt": None,
                "satellites": None,
            }
        )
        assert cfg.data.corr == "xx"
        assert cfg.opt.max_iter == 1000
        assert cfg.satellites.norad_ids == []

    @pytest.mark.parametrize("legacy_section", ["ast", "rfi", "gains"])
    def test_legacy_top_level_sections_are_rejected(self, legacy_section):
        with pytest.raises(ConfigValidationError, match=r"model\.components") as exc:
            validate_config(_minimal_raw(**{legacy_section: {}}))
        assert legacy_section in str(exc.value)

    def test_an_inference_section_is_refused_with_the_way_out(self):
        with pytest.raises(ConfigValidationError, match="top-level key 'seed'"):
            validate_config({"model": {"components": []}, "inference": {"seed": 3}})

    def test_light_curve_cli_exits_cleanly_on_invalid_config(self, tmp_path):
        from tabascal.scripts.rfi_estimate import build_parser, run as run_light_curve

        bad = tmp_path / "bad.yaml"
        bad.write_text("model:\n  components: []\nrfi:\n  init: truth\n")
        args = build_parser().parse_args(["-c", str(bad)])
        with pytest.raises(SystemExit, match=r"Error:.*model\.components"):
            run_light_curve(args)

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
            {"gains.UnitaryGains": None},
            {"gains.UnitaryGains": {}},
        ],
        ids=["dot-bare", "null-mapping", "empty-mapping"],
    )
    def test_bare_and_empty_component_entries_parse(self, entry):
        cfg = validate_config(_minimal_raw([entry]))
        (spec,) = cfg.model.components
        assert spec.ref == "gains.UnitaryGains"
        assert isinstance(spec.config, EmptyComponentConfig)

    @pytest.mark.parametrize("entry", ["gains:UnitaryGains", {"gains:UnitaryGains": {}}])
    def test_the_colon_spelling_is_refused_with_the_dotted_one(self, entry):
        with pytest.raises(ConfigValidationError, match="write 'gains.UnitaryGains'"):
            validate_config(_minimal_raw([entry]))

    @pytest.mark.parametrize(
        "component, key, alias",
        [
            ("rfi_signal.ComplexRFIVarAnt", "init", "mf"),
            ("rfi_signal.ComplexRFIVarAnt", "init", 0),
            ("rfi_signal.ComplexRFIVarAnt", "init", 1),
            ("rfi_signal.ComplexRFIVarAnt", "mean", 0),
            ("ast_vis.GPVisAst", "mean", 0),
            ("gains.ConstGains", "fit", "both"),
        ],
    )
    def test_an_alias_is_refused(self, component, key, alias):
        with pytest.raises(ConfigValidationError, match=key):
            validate_config(_minimal_raw([{component: {key: alias}}]))

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
                _minimal_raw([{"gains.UnitaryGains": {"amp_std_pct": 10}}])
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
                    "data": {"freq_hz": -10},
                }
            )
        msg = str(exc.value)
        assert "path_order" in msg
        assert "corr_tme" in msg
        assert "Did you mean 'corr_time_s'" in msg
        assert "stencil" in msg
        assert "data.freq_hz" in msg


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
        assert traj.min_elevation_deg == 0.0
        assert traj.path_order == 3
        assert traj.path_nodes == "auto"
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
            {"min_elevation_deg": True},
            {"min_elevation_deg": float("nan")},
        ],
    )
    def test_fixed_orbit_rejects_invalid_fields(self, bad_fields):
        with pytest.raises(ConfigValidationError):
            validate_config(_minimal_raw([{"trajectory.FixedOrbit": bad_fields}]))

    def test_rfi_signal_defaults_are_unified(self):
        cfg = validate_config(
            _minimal_raw(
                [
                    "trajectory.FixedOrbit",
                    "rfi_signal.ComplexRFIVarAnt",
                    "rfi_vis.AnalyticVis",
                    "ast_vis.GPVisAst",
                ],
                satellites={"norad_ids": [25544]},
            )
        )
        var_cfg = cfg.model.components[1].config
        ast_cfg = cfg.model.components[3].config
        assert var_cfg.gammas == (5.0, 5.0)
        assert var_cfg.std_jy == "data"
        assert var_cfg.cutoff == 1e-6
        assert var_cfg.corr_freq_hz == 1e6
        assert var_cfg.corr_time_s == 24.0
        assert (var_cfg.gammas, var_cfg.cutoff) == (ast_cfg.gammas, ast_cfg.cutoff)

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
        assert type(const_cfg) is type(var_cfg)
        assert const_cfg.gammas == (5.0, 5.0)
        assert const_cfg.cutoff == 1e-6

    @pytest.mark.parametrize(
        "bad_fields",
        [
            {"std_jy": 0},
            {"std_jy": -1.0},
            {"std_jy": True},
            {"std_jy": float("inf")},
            {"corr_freq_hz": 0},
            {"corr_time_s": -5},
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
                            "point_sources": [{"ra_deg": 10.0, "dec_deg": -20.0, "I_jy": 5.0}]
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
                                {"name": 1934, "ra_deg": "10.5", "dec_deg": -20, "I_jy": "750.0",
                                 "alpha": " -0.8 ", "ref_freq_hz": "1.4e9"},
                            ]
                        }
                    },
                    "ast_vis.DiscreteSkyVis",
                ]
            )
        )
        (source,) = cfg.model.components[0].config.point_sources
        assert source.name == "1934"
        assert (source.ra_deg, source.dec_deg, source.I_jy) == (10.5, -20.0, 750.0)
        assert (source.alpha, source.ref_freq_hz) == (-0.8, 1.4e9)
        assert source.to_oskar_row(0)["name"] == "1934"

    @pytest.mark.parametrize("value", ["nan", "inf", "lots", ""])
    def test_quoted_inline_values_must_still_be_finite_numbers(self, value):
        with pytest.raises(ConfigValidationError, match=r"point_sources\[0\]\.I_jy"):
            validate_config(
                _minimal_raw(
                    [
                        {
                            "ast_signal.FixedDiscreteSky": {
                                "point_sources": [{"ra_deg": 1.0, "dec_deg": 2.0, "I_jy": value}]
                            }
                        },
                        "ast_vis.DiscreteSkyVis",
                    ]
                )
            )

    def test_inline_source_typo_points_to_source_field_not_point_sources(self):
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(
                _minimal_raw(
                    [
                        {
                            "ast_signal.FixedDiscreteSky": {
                                "point_sources": [
                                    {"ra_deg": 50.0, "dec_deg": -37.0, "I_jy": 10.0, "fwhm_maj": 5.0}
                                ]
                            }
                        },
                        "ast_vis.DiscreteSkyVis",
                    ]
                )
            )
        msg = str(exc.value)
        assert "point_sources[0].fwhm_maj" in msg
        assert "Did you mean 'fwhm_major_arcsec'" in msg
        assert "takes: point_sources" not in msg
        assert "Input should be a valid string" not in msg

    def test_whitespace_noise_path_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="noise_jy"):
            validate_config(_minimal_raw(data={"noise_jy": "   "}))

    @pytest.mark.parametrize(
        "raw",
        [
            _minimal_raw(data={"ms_path": ""}),
            _minimal_raw(data={"ms_path": "   "}),
            _minimal_raw(data={"out_dir": "   "}),
            _minimal_raw(data={"truth_zarr": ""}),
            _minimal_raw(satellites={"norad_ids_path": "   "}),
            _minimal_raw(satellites={"extra_orbit_dir": ""}),
            _minimal_raw(opt={"trace_path": "   "}),
            _minimal_raw([{"rfi_signal.ComplexRFIVarAnt": {"init": "est", "est": ""}}]),
            _minimal_raw([{"gains.ConstGains": {"init": "est", "est": ""}}]),
            _minimal_raw([{"gains.ConstGains": {"init": "est", "est": "   "}}]),
        ],
    )
    def test_empty_or_whitespace_optional_paths_are_rejected(self, raw):
        with pytest.raises(ConfigValidationError, match="must not be empty"):
            validate_config(raw)

    @pytest.mark.parametrize("field", ["fwhm_major_arcsec", "fwhm_minor_arcsec"])
    def test_inline_source_rejects_negative_fwhm(self, field):
        with pytest.raises(ConfigValidationError, match=field):
            validate_config(
                _minimal_raw(
                    [
                        {
                            "ast_signal.FixedDiscreteSky": {
                                "point_sources": [
                                    {"ra_deg": 1.0, "dec_deg": 2.0, "I_jy": 3.0, field: -5.0}
                                ]
                            }
                        },
                        "ast_vis.DiscreteSkyVis",
                    ]
                )
            )

    def test_unnamed_inline_source_alpha_error_does_not_repeat_source(self):
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(
                _minimal_raw(
                    [
                        {
                            "ast_signal.FixedDiscreteSky": {
                                "point_sources": [
                                    {"ra_deg": 1.0, "dec_deg": 2.0, "I_jy": 3.0, "alpha": -0.7}
                                ]
                            }
                        },
                        "ast_vis.DiscreteSkyVis",
                    ]
                )
            )
        msg = str(exc.value)
        assert "source has a spectral index" in msg
        assert "source source" not in msg

    @pytest.mark.parametrize("section, key", [("opt", "inline"), ("data", "file")])
    def test_a_key_spelled_like_a_point_sources_tag_is_still_named(self, section, key):
        # 'file' and 'inline' tag point_sources' union and are hidden there only.
        with pytest.raises(ConfigValidationError, match=rf"{section}\.{key}: Extra inputs"):
            validate_config(_minimal_raw(**{section: {key: 1}}))

    def test_const_gains_defaults_and_validation(self):
        cfg = validate_config(_minimal_raw(["ast_vis.GPVisAst", "gains.ConstGains"]))
        g = cfg.model.components[1].config
        assert g.amp_std_pct == 20.0
        assert g.phase_std_deg == 180.0
        assert g.fix_flux_scale is True
        assert g.fit == "complex"

        for bad in (
            {"amp_std_pct": 0},
            {"phase_std_deg": -1},
            {"amp_mean": 0},
            {"amp_std_pct": True},
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

    @pytest.mark.parametrize(
        "components, kind",
        [
            (["trajectory.FixedOrbitFine", "rfi_signal.ComplexRFIVarAntFine",
              "rfi_signal.ComplexRFIConstAntFine", "rfi_vis.RiemannVisFine"], "rfi_signal"),
            (["trajectory.FixedOrbitFine", "rfi_signal.ComplexRFIVarAntFine",
              "rfi_vis.RiemannVisFine", "rfi_vis.RiemannVisFine"], "rfi_vis"),
            (["ast_vis.GPVisAst", "gains.UnitaryGains", "gains.ConstGains"], "gains"),
        ],
    )
    def test_at_most_one_component_of_a_kind_that_would_overwrite_or_double(self, components, kind):
        """Repeated settings that would have to agree are not repeated at all."""
        with pytest.raises(ConfigValidationError, match=f"2 {kind} components.*at most one"):
            validate_config(_minimal_raw(components, satellites={"norad_ids": [25544]}))

    def test_two_sky_models_still_add(self):
        """vis_ast accumulates, and a GP beside a fixed sky is the documented model."""
        validate_config(_minimal_raw([
            {"ast_signal.FixedDiscreteSky": {"point_sources": [{"ra_deg": 1.0, "dec_deg": 2.0, "I_jy": 3.0}]}},
            "ast_vis.GPVisAst", "ast_vis.DiscreteSkyVis", "gains.UnitaryGains",
        ]))

    @pytest.mark.parametrize(
        "components, precision",
        [
            (["ast_vis.GPVisAst", "ast_vis.GPVisAst"], "single"),
            (
                [
                    {"ast_signal.FixedDiscreteSky": {"point_sources": [{"ra_deg": 1.0, "dec_deg": 2.0, "I_jy": 3.0}]}},
                    {"ast_signal.FixedDiscreteSky": {"point_sources": [{"ra_deg": 4.0, "dec_deg": 5.0, "I_jy": 6.0}]}},
                    "ast_vis.DiscreteSkyVis",
                ],
                "single",
            ),
            (
                [
                    {"ast_signal.FixedDiscreteSky": {"point_sources": [{"ra_deg": 1.0, "dec_deg": 2.0, "I_jy": 3.0}]}},
                    "ast_vis.DiscreteSkyVis",
                    "ast_vis.DiscreteSkyVis",
                ],
                "single",
            ),
            (
                [
                    "trajectory.OrbitFine",
                    "trajectory.PhaseCalculationRFIFine",
                    "trajectory.PhaseCalculationRFIFine",
                ],
                "double",
            ),
        ],
    )
    def test_duplicate_component_classes_are_rejected(self, components, precision):
        with pytest.raises(ConfigValidationError, match="more than once"):
            validate_config({"model": {"precision": precision, "components": components}})


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
                    "noise_jy": "noise.npz",
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
                                "init": "est",
                                "est": "gains.npz",
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
        assert paths["data.noise_jy"] == str(base / "noise.npz")
        assert paths["satellites.norad_ids_path"] == str(base / "sats.txt")
        assert paths["satellites.extra_orbit_dir"] == str(base / "tles")
        assert paths["rfi_signal.ComplexRFIConstAnt.est"] == str(base / "lc.npz")
        assert paths["ast_signal.FixedDiscreteSky.point_sources"] == str(base / "sky.osm")
        assert paths["gains.ConstGains.est"] == str(base / "gains.npz")

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


class TestOptionsThatWouldBeIgnoredAreRefused:
    """A setting the run would not read is an error when the config loads."""

    @pytest.mark.parametrize("key", ["init", "mean"])
    def test_est_without_a_path_is_refused(self, key):
        with pytest.raises(ConfigValidationError, match=f"{key}: est.*est is not set"):
            validate_config(_minimal_raw([{"rfi_signal.ComplexRFIVarAnt": {key: "est"}}]))

    def test_a_path_nothing_reads_is_refused(self):
        with pytest.raises(ConfigValidationError, match="nothing reads it"):
            validate_config(_minimal_raw([{"rfi_signal.ComplexRFIVarAnt": {"est": "lc.npz"}}]))

    def test_est_with_a_reader_is_accepted(self):
        cfg = validate_config(
            _minimal_raw([{"rfi_signal.ComplexRFIVarAnt": {"mean": "est", "est": "lc.npz"}}])
        )
        assert cfg.model.components[0].config.est == "lc.npz"

    @pytest.mark.parametrize("dual_run", [True, False])
    def test_dual_run_beside_a_schedule_is_refused(self, dual_run):
        with pytest.raises(ConfigValidationError, match="dual_run"):
            validate_config(
                {"model": {"components": []},
                 "opt": {"schedule": {"kind": "constant"}, "dual_run": dual_run}}
            )

    def test_a_saved_schedule_config_validates_again(self):
        """The saved config leaves dual_run out, so a replay is not refused."""
        cfg = validate_config({"model": {"components": []}, "opt": {"schedule": {"kind": "constant"}}})
        dumped = cfg.to_yaml_dict()

        assert "dual_run" not in dumped["opt"]
        assert validate_config(dumped).opt.schedule.kind == "constant"

    @pytest.mark.parametrize("backend", [None, "ffi"])
    def test_a_block_size_the_ffi_kernels_ignore_is_refused(self, backend):
        options = {"baseline_block_size": 64, **({"backend": backend} if backend else {})}
        with pytest.raises(ConfigValidationError, match="baseline_block_size is read only with backend: jax"):
            validate_config(_minimal_raw([{"rfi_vis.RiemannVisFine": options}]))

    def test_a_block_size_round_trips_under_both_backends(self):
        for options in ({}, {"backend": "jax", "baseline_block_size": 64}):
            cfg = validate_config(_minimal_raw(
                ["trajectory.FixedOrbitFine", "rfi_signal.ComplexRFIVarAntFine",
                 {"rfi_vis.RiemannVisFine": options}],
                satellites={"norad_ids": [25544]},
            ))
            again = validate_config(cfg.to_yaml_dict())
            assert again.model.components[2].config == cfg.model.components[2].config


class TestTypedOptimiserSections:

    def test_a_name_alone_is_that_optimiser(self):
        cfg = validate_config({"model": {"components": []}, "opt": {"optimizer": "adam"}})
        assert cfg.opt.optimizer.name == "adam"

    @pytest.mark.parametrize(
        "opt, match",
        [
            ({"optimizer": {"name": "adamm"}}, r"opt\.optimizer\.name"),
            ({"schedule": {"kind": "stepp"}}, r"opt\.schedule"),
            ({"schedule": {"kind": "step"}}, r"boundaries"),
            ({"schedule": {"kind": "cosine", "final": 0.1}}, r"takes:.*final_factor"),
            ({"schedule": {"kind": "cosine", "final_factr": 0.1}}, r"Did you mean 'final_factor'"),
            ({"schedule": {"kind": "constant"}, "stop": {"window": 5, "rtol": 1e-3, "min_step": 10}}, r"Did you mean 'min_steps'"),
            ({"stop": {"window": 5}}, r"opt\.stop\.rtol"),
            ({"optimizer": {"name": "sgd", "nope": 1}}, r"does not take \['nope'\]"),
        ],
    )
    def test_a_malformed_section_is_refused_when_the_config_loads(self, opt, match):
        with pytest.raises(ConfigValidationError, match=match):
            validate_config({"model": {"components": []}, "opt": opt})

    def test_the_sections_reach_the_validators_as_mappings(self):
        from tabascal.opt_schedule import validate_opt

        cfg = validate_config({
            "model": {"components": []},
            "opt": {
                "max_iter": 100,
                "optimizer": {"name": "sgd", "momentum": 0.9, "clip_norm": 1},
                "schedule": {"kind": "step", "boundaries": [50]},
                "stop": {"window": 5, "rtol": 1e-3},
            },
        })
        checked = validate_opt(cfg.opt)

        assert checked["optimizer"] == {"name": "sgd", "clip_norm": 1.0, "args": {"momentum": 0.9}}
        assert checked["schedule"]["boundaries"] == [50]
        assert checked["stop"]["window"] == 5
