# Test Inventory

## `tests/components/test_trajectory.py`

No Space-Track credentials are required for any test in this file. `TestSGP4LEONoDragOrbit` and `TestSGP4LEOOrbit` use a bundled TLE cache file (`tabascal/data/tles/2026-04-15-bundled.json`, NORAD IDs 20452 / 38833) and a fixed observation epoch matching that file's date prefix, so `get_tles_by_id` reads from disk and never contacts the Space-Track API.


### TestPhaseCalculationRFI


| Test name | Description |
|-----------|-------------|
| [`test_setup_succeeds`](components/test_trajectory.py#L143) | Component initialises without error with a default mock config. |
| [`test_setup_validates_dimensions`](components/test_trajectory.py#L149) | If the config is self-consistent, _validate_dimensions must not raise. |
| [`test_set_params_is_identity`](components/test_trajectory.py#L157) | build_set_params returns a no-op pass-through. |
| [`test_forward_output_shape`](components/test_trajectory.py#L166) | Forward pass produces rfi_phase with shape (n_rfi, n_ant, n_freq_fine, n_time_fine). |
| [`test_forward_phase_is_finite`](components/test_trajectory.py#L186) | All phase values should be finite for a realistic satellite position. |
| [`test_forward_phase_varies_across_antennas`](components/test_trajectory.py#L201) | Different antennas should see different phase delays. |
| [`test_forward_preserves_rfi_xyz_in_state`](components/test_trajectory.py#L217) | Forward pass copies rfi_xyz through to the output state unchanged. |
| [`test_parametric_sizes[2-1-1-2-1]`](components/test_trajectory.py#L235) | Output shape and finiteness verified across a range of dimension combinations. |
| [`test_parametric_sizes[4-2-4-8-2]`](components/test_trajectory.py#L235) | Output shape and finiteness verified across a range of dimension combinations. |
| [`test_parametric_sizes[8-3-2-6-3]`](components/test_trajectory.py#L235) | Output shape and finiteness verified across a range of dimension combinations. |
| [`test_compute_ant_pos_xyz_earth_radius`](components/test_trajectory.py#L253) | ants_xyz (GCRF) should be at Earth's surface radius (~6.37e6 m). |
| [`test_compute_ant_pos_uvw_shape_and_finite`](components/test_trajectory.py#L262) | ants_uvw must have shape (n_ant, n_time_fine, 3) with all finite values. |
| [`test_compute_ant_pos_distinct_across_antennas`](components/test_trajectory.py#L271) | Different antennas must have distinct GCRF positions. |

### TestFixedOrbit


| Test name | Description |
|-----------|-------------|
| [`test_setup_succeeds`](components/test_trajectory.py#L285) | Component propagates the TLE orbit and pre-computes phase without error. |
| [`test_rfi_xyz_shape`](components/test_trajectory.py#L291) | Pre-computed satellite positions stored at setup have shape (n_rfi, n_time_fine, 3). |
| [`test_rfi_phase_shape`](components/test_trajectory.py#L300) | Pre-computed phase stored at setup has shape (n_rfi, n_ant, n_freq, n_time_fine). |
| [`test_rfi_xyz_nonzero`](components/test_trajectory.py#L312) | Propagated satellite positions must be non-zero (orbit was computed). |
| [`test_rfi_xyz_altitude_reasonable`](components/test_trajectory.py#L319) | ISS is at ~400 km altitude — distance from Earth's centre ≈ 6.8e6 m. |
| [`test_rfi_phase_finite`](components/test_trajectory.py#L328) | All pre-computed phase values are finite. |
| [`test_forward_adds_rfi_xyz_and_phase_to_state`](components/test_trajectory.py#L335) | Forward pass inserts rfi_xyz and rfi_phase into the state dict. |
| [`test_forward_output_matches_precomputed`](components/test_trajectory.py#L345) | Forward pass must return the same pre-computed arrays stored at setup time. |
| [`test_forward_is_deterministic`](components/test_trajectory.py#L354) | Calling build_forward twice with the same input gives the same result. |
| [`test_two_satellites_shape`](components/test_trajectory.py#L364) | Two distinct TLEs produce position and phase arrays of the correct shape. |
| [`test_two_satellites_have_different_positions`](components/test_trajectory.py#L383) | Two distinct TLEs must propagate to distinct positions. |
| [`test_build_set_params_is_identity`](components/test_trajectory.py#L394) | FixedOrbit.build_set_params returns a pass-through with no side effects. |
| [`test_compute_rfi_phase_consistent_with_get_rfi_phase`](components/test_trajectory.py#L408) | Phase stored at setup must equal get_rfi_phase called with the same arrays. |
| [`test_compute_rfi_phase_xyz_is_satellite_altitude`](components/test_trajectory.py#L416) | rfi_xyz computed during _compute_rfi_phase should be at LEO altitude. |

### TestSGP4LEONoDragOrbit

Uses the bundled TLE cache; no Space-Track credentials required.


| Test name | Description |
|-----------|-------------|
| [`test_setup_succeeds`](components/test_trajectory.py#L478) | Component loads TLEs from the repo cache and initialises without error. |
| [`test_rfi_xyz_shape`](components/test_trajectory.py#L485) | Initial state_outputs['rfi_xyz'] placeholder has shape (n_rfi, n_time_fine, 3). |
| [`test_init_params_base_shape`](components/test_trajectory.py#L493) | Initial base orbit parameters have shape (n_rfi, 6) — bstar excluded from learnable params. |
| [`test_prior_covariance_positive_definite`](components/test_trajectory.py#L501) | L_rfi_orbit must be lower-triangular with positive diagonal. |
| [`test_forward_output_shapes`](components/test_trajectory.py#L511) | Forward pass produces rfi_xyz (n_rfi, n_time_fine, 3) and elements (n_rfi, 6). |
| [`test_forward_rfi_xyz_finite`](components/test_trajectory.py#L524) | SGP4-propagated satellite positions from the forward pass are all finite. |
| [`test_build_set_params_samples_correct_shapes`](components/test_trajectory.py#L536) | build_set_params must sample rfi_orbit_base with shape (n_rfi, 6) inside a NumPyro trace. |
| [`test_forward_transform_roundtrip`](components/test_trajectory.py#L550) | inv_transform(forward_transform(x)) == x. |
| [`test_inv_transform_roundtrip`](components/test_trajectory.py#L563) | forward_transform(inv_transform(x)) == x. |
| [`test_sats_init_direct_call`](components/test_trajectory.py#L578) | sats_init called directly with comp.elements must return without error. |
| [`test_sats_init_produces_valid_positions`](components/test_trajectory.py#L587) | sats_init output propagated via sgp4jax must yield finite LEO positions. |

### TestSGP4LEOOrbit

Uses the bundled TLE cache; no Space-Track credentials required.


| Test name | Description |
|-----------|-------------|
| [`test_setup_succeeds`](components/test_trajectory.py#L639) | Component loads TLEs from the repo cache and initialises without error. |
| [`test_rfi_xyz_shape`](components/test_trajectory.py#L646) | Initial state_outputs['rfi_xyz'] placeholder has shape (n_rfi, n_time_fine, 3). |
| [`test_init_params_base_shape`](components/test_trajectory.py#L654) | SGP4LEOOrbit has 7 orbit parameters (includes bstar). |
| [`test_prior_covariance_positive_definite`](components/test_trajectory.py#L662) | Cholesky factor L_rfi_orbit (7x7) has a positive diagonal for every satellite. |
| [`test_forward_output_shapes`](components/test_trajectory.py#L672) | Forward pass produces rfi_xyz (n_rfi, n_time_fine, 3) and elements (n_rfi, 7). |
| [`test_forward_rfi_xyz_finite`](components/test_trajectory.py#L685) | SGP4-propagated satellite positions from the forward pass are all finite. |
| [`test_build_set_params_samples_correct_shapes`](components/test_trajectory.py#L697) | build_set_params must sample rfi_orbit_base with shape (n_rfi, 7) inside a NumPyro trace. |
| [`test_forward_transform_roundtrip`](components/test_trajectory.py#L711) | inv_transform(forward_transform(x)) == x. |
| [`test_inv_transform_roundtrip`](components/test_trajectory.py#L724) | forward_transform(inv_transform(x)) == x. |
| [`test_sats_init_direct_call`](components/test_trajectory.py#L739) | sats_init called directly with comp.elements must return without error. |
| [`test_sats_init_produces_valid_positions`](components/test_trajectory.py#L748) | sats_init output propagated via sgp4jax must yield finite LEO positions. |

---

## `tests/components/test_gains.py`


### TestGainsConfigValidation


| Test name | Description |
|-----------|-------------|
| [`test_null_values_get_defaults`](components/test_gains.py#L94) | None corr_time / corr_freq values are replaced with defaults derived from the observation grid. |
| [`test_explicit_values_stored_correctly`](components/test_gains.py#L118) | Non-null correlation times and frequencies are stored unchanged on the component. |
| [`test_invalid_amp_mean_type_raises`](components/test_gains.py#L144) | A non-numeric amp_mean raises ValueError during gains_config_validation. |
| [`test_invalid_phase_std_type_raises`](components/test_gains.py#L162) | A non-numeric phase_std raises ValueError during gains_config_validation. |
| [`test_single_freq_single_time_defaults`](components/test_gains.py#L180) | Single channel/integration — corr lengths should default to step size. |

### TestUnitaryGains


| Test name | Description |
|-----------|-------------|
| [`test_setup_succeeds`](components/test_gains.py#L199) | Component initialises without error. |
| [`test_state_outputs_shapes`](components/test_gains.py#L205) | state_outputs['gains'] placeholder has shape (n_ant, n_freq, n_time). |
| [`test_no_learnable_params`](components/test_gains.py#L216) | init_params_base is empty — UnitaryGains has no free parameters. |
| [`test_forward_vis_obs_equals_sum`](components/test_gains.py#L223) | UnitaryGains applies no actual gains: vis_obs = vis_rfi + vis_ast. |
| [`test_forward_preserves_other_state_keys`](components/test_gains.py#L237) | Forward does not drop pre-existing keys in the state dict. |
| [`test_forward_output_shapes[2-1-4]`](components/test_gains.py#L252) | gains and vis_obs output shapes are correct for the given (n_ant, n_freq, n_time) dimensions. |
| [`test_forward_output_shapes[6-8-10]`](components/test_gains.py#L252) | gains and vis_obs output shapes are correct for the given (n_ant, n_freq, n_time) dimensions. |
| [`test_forward_output_shapes[16-4-12]`](components/test_gains.py#L252) | gains and vis_obs output shapes are correct for the given (n_ant, n_freq, n_time) dimensions. |

### TestGPGains


| Test name | Description |
|-----------|-------------|
| [`test_setup_succeeds`](components/test_gains.py#L270) | Component initialises without error. |
| [`test_prior_params_shapes`](components/test_gains.py#L276) | Prior mean and Cholesky L arrays have shapes consistent with the GP parameterisation. |
| [`test_init_params_base_shapes`](components/test_gains.py#L292) | Initial base parameter arrays match the GP amplitude and phase grid sizes. |
| [`test_forward_output_shapes`](components/test_gains.py#L308) | Forward produces gains (n_ant, n_freq, n_time) and vis_obs (n_bl, n_freq, n_time). |
| [`test_forward_gains_at_prior_mean_amplitude`](components/test_gains.py#L331) | At init params (prior mean), gain amplitudes should be close to amp_mean. |
| [`test_forward_gains_last_antenna_phase_zero`](components/test_gains.py#L355) | Last antenna phase is fixed to zero (reference antenna). |
| [`test_forward_output_is_complex`](components/test_gains.py#L375) | gains and vis_obs from the forward pass are complex-valued. |
| [`test_resample_matrices_shapes`](components/test_gains.py#L391) | GP resampling matrices have shapes consistent with the coarse/fine time grids. |
| [`test_setup_and_forward_various_sizes[2-1-4]`](components/test_gains.py#L410) | Setup and forward succeed end-to-end for the given (n_ant, n_freq, n_time). |
| [`test_setup_and_forward_various_sizes[5-3-12]`](components/test_gains.py#L410) | Setup and forward succeed end-to-end for the given (n_ant, n_freq, n_time). |
| [`test_setup_and_forward_various_sizes[8-4-16]`](components/test_gains.py#L410) | Setup and forward succeed end-to-end for the given (n_ant, n_freq, n_time). |
| [`test_build_set_params_samples_correct_shapes`](components/test_gains.py#L433) | build_set_params must produce correctly shaped samples inside a NumPyro trace. |
| [`test_forward_transform_roundtrip`](components/test_gains.py#L453) | inv_transform(forward_transform(x)) == x up to floating-point precision. |
| [`test_inv_transform_roundtrip`](components/test_gains.py#L473) | forward_transform(inv_transform(x)) == x up to floating-point precision. |

### TestBaseGPGainsInheritedSetParams


| Test name | Description |
|-----------|-------------|
| [`test_unitary_gains_build_set_params_is_identity`](components/test_gains.py#L500) | UnitaryGains inherits BaseGPGains.build_set_params which is a no-op. |

---

## `tests/components/test_rfi_vis.py`

Tests for `RiemannVisTimeFreqCalculation` and `RiemannVisTimeFreqCalculationFFI`. Each parametrized over three size tuples: `(1,1,1,1,1,1)`, `(4,5,6,7,8,9)`, `(64,20,16,12,4,2)`.


| Test name | Description |
|-----------|-------------|
| [`test_ffi[1-1-1-1-1-1]`](components/test_rfi_vis.py#L49) | FFI and reference Riemann kernels produce identical vis_rfi outputs. |
| [`test_ffi[4-5-6-7-8-9]`](components/test_rfi_vis.py#L49) | FFI and reference Riemann kernels produce identical vis_rfi outputs. |
| [`test_ffi[64-20-16-12-4-2]`](components/test_rfi_vis.py#L49) | FFI and reference Riemann kernels produce identical vis_rfi outputs. |
| [`test_ffi_jvp[1-1-1-1-1-1]`](components/test_rfi_vis.py#L65) | Forward-mode Jacobian-vector products of FFI and reference kernels match. |
| [`test_ffi_jvp[4-5-6-7-8-9]`](components/test_rfi_vis.py#L65) | Forward-mode Jacobian-vector products of FFI and reference kernels match. |
| [`test_ffi_jvp[64-20-16-12-4-2]`](components/test_rfi_vis.py#L65) | Forward-mode Jacobian-vector products of FFI and reference kernels match. |
| [`test_ffi_vjp[1-1-1-1-1-1]`](components/test_rfi_vis.py#L85) | Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between FFI and reference. |
| [`test_ffi_vjp[4-5-6-7-8-9]`](components/test_rfi_vis.py#L85) | Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between FFI and reference. |
| [`test_ffi_vjp[64-20-16-12-4-2]`](components/test_rfi_vis.py#L85) | Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between FFI and reference. |

---

## `tests/test_fft_gp.py`


### TestSupersample


| Test name | Description |
|-----------|-------------|
| [`test_supersample_domain_specs_1d`](test_fft_gp.py#L45) | Test supersampling specs for 1D domain. |
| [`test_supersample_domain_specs_2d`](test_fft_gp.py#L55) | Test supersampling specs for 2D domain. |
| [`test_supersample_domain_specs_length_mismatch`](test_fft_gp.py#L66) | Test that length mismatch raises error. |
| [`test_supersample_domain_1d`](test_fft_gp.py#L74) | Test domain supersampling in 1D. |
| [`test_supersample_domain_k_1d`](test_fft_gp.py#L88) | Test k-domain supersampling in 1D. |
| [`test_supersample_signal_1d`](test_fft_gp.py#L99) | Test signal supersampling in 1D. |
| [`test_supersample_signal_2d`](test_fft_gp.py#L113) | Test signal supersampling in 2D. |
| [`test_supersample_jit_compatible`](test_fft_gp.py#L122) | Test that supersample is JIT-compatible. |

### TestDomainK


| Test name | Description |
|-----------|-------------|
| [`test_domain_k_1d`](test_fft_gp.py#L136) | Test k-domain calculation in 1D. |
| [`test_domain_k_2d`](test_fft_gp.py#L148) | Test k-domain calculation in 2D. |

### TestPadding


| Test name | Description |
|-----------|-------------|
| [`test_pad_domain_specs_validation`](test_fft_gp.py#L164) | Test that pad_domain_specs validates inputs. |
| [`test_pad_domain_specs_1d`](test_fft_gp.py#L178) | Test padding specs for 1D domain. |
| [`test_pad_domain_1d`](test_fft_gp.py#L186) | Test domain padding in 1D. |
| [`test_pad_domain_k_1d`](test_fft_gp.py#L198) | Test k-domain padding in 1D. |
| [`test_pad_signal_1d`](test_fft_gp.py#L209) | Test signal padding in 1D. |
| [`test_pad_signal_2d`](test_fft_gp.py#L219) | Test signal padding in 2D. |
| [`test_pad_signal_dimension_mismatch`](test_fft_gp.py#L227) | Test that pad validates dimension mismatch. |
| [`test_pad_jit_compatible`](test_fft_gp.py#L233) | Test that pad is JIT-compatible. |

### TestFourierCutting


| Test name | Description |
|-----------|-------------|
| [`test_pk_cut_1d`](test_fft_gp.py#L247) | Test power spectrum cutting in 1D. |
| [`test_pk_cut_2d`](test_fft_gp.py#L260) | Test power spectrum cutting in 2D. |
| [`test_fourier_cut_uncut_roundtrip`](test_fft_gp.py#L272) | Test that fourier_cut followed by fourier_uncut preserves shape. |
| [`test_supersample_fourier_1d`](test_fft_gp.py#L288) | Test Fourier-based supersampling in 1D. |
| [`test_supersample_fourier_2d`](test_fft_gp.py#L296) | Test Fourier-based supersampling in 2D. |

### TestPowerSpectrum


| Test name | Description |
|-----------|-------------|
| [`test_pow_spec_1d_shape`](test_fft_gp.py#L307) | Test 1D power spectrum shape. |
| [`test_pow_spec_1d_symmetry`](test_fft_gp.py#L314) | Test that 1D power spectrum is symmetric. |
| [`test_pow_spec_1d_peak_at_zero`](test_fft_gp.py#L323) | Test that 1D power spectrum peaks at k=0. |
| [`test_pow_spec_nd_validation`](test_fft_gp.py#L331) | Test that pow_spec_nd validates input lengths. |
| [`test_pow_spec_nd_2d_shape`](test_fft_gp.py#L339) | Test 2D power spectrum shape. |
| [`test_pow_spec_nd_3d_shape`](test_fft_gp.py#L347) | Test 3D power spectrum shape. |

### TestDomainSS


| Test name | Description |
|-----------|-------------|
| [`test_domain_ss_1d`](test_fft_gp.py#L358) | Test supersampled domain in 1D. |
| [`test_domain_ss_2d`](test_fft_gp.py#L372) | Test supersampled domain in 2D. |

### TestLatentSpace


| Test name | Description |
|-----------|-------------|
| [`test_signal_to_latent_1d`](test_fft_gp.py#L387) | Test latent representation extraction in 1D. |
| [`test_latent_init_returns_correct_types`](test_fft_gp.py#L403) | Test that latent_to_signal_init returns correct data types. |
| [`test_latent_init_predict_roundtrip`](test_fft_gp.py#L427) | Test that latent_to_signal_init and latent_to_signal work together. |
| [`test_latent_operations_jit_compatible`](test_fft_gp.py#L454) | Test that latent operations are JIT-compatible. |
| [`test_signal_to_latent_jit_compatible`](test_fft_gp.py#L481) | Test that signal_to_latent_init + signal_to_latent is JIT-compatible. |

### TestJAXCompatibility


| Test name | Description |
|-----------|-------------|
| [`test_pk_cut_jit`](test_fft_gp.py#L515) | Test that pk_cut can be called (even though it's not fully JIT-compatible). |

### TestNumericalAccuracy


| Test name | Description |
|-----------|-------------|
| [`test_supersample_preserves_dc_component`](test_fft_gp.py#L530) | Test that supersampling preserves DC component. |
| [`test_fourier_cut_uncut_preserves_kept_modes`](test_fft_gp.py#L541) | Test that cutting and uncutting preserves kept Fourier modes. |

---

## `tests/test_timing.py`


| Test name | Description |
|-----------|-------------|
| [`test_timing_collection_enabled`](test_timing.py#L25) | Test that timings are collected when enabled. |
| [`test_timing_collection_disabled`](test_timing.py#L44) | Test that timings are NOT collected when disabled. |
| [`test_hierarchical_timing`](test_timing.py#L61) | Test that hierarchical timings are correctly captured. |
| [`test_timer_context_manager`](test_timing.py#L85) | Test the manual timer context manager. |
| [`test_measure_runtime_data_structures`](test_timing.py#L98) | Test that measure_runtime works with various data structures. |
| [`test_measure_runtime_mixed_types`](test_timing.py#L126) | Test the decorator with mixed JAX and non-JAX types. |

---

## `tests/test_tabascal_pipeline.py`

End-to-end integration tests. Each test invokes `run_tabascal.py` as a subprocess, checks `returncode == 0`, and validates the `Reduced Chi^2 @ opt params` value printed to stdout.

`TabConfig` fetches TLEs using the mean observation epoch from the MS file. The simulation data (from HuggingFace) uses the tabsim default epoch of 2023-02-21; the repo ships `tabascal/data/tles/2023-02-21-HMZGLE.json` containing NORAD IDs 20452, 38833, and 45854 (all three listed in `tests/data/tab_target.yaml`), so the `TabConfig` TLE lookup is always satisfied from disk. Tests that additionally use `SGP4LEONoDragOrbit` or `SGP4LEOOrbit` call `fetch_standard_orbital_elements` a second time during component `setup`; that second call is also satisfied by the same bundled cache file — but only when the component's observation epoch likewise falls on 2023-02-21. Tests marked **requires Space-Track** are skipped automatically when `tabascal.tle.load_spacetrack_credentials` returns `(None, None)`.


| Test name | Description |
|-----------|-------------|
| [`test_tabascal_pipeline[RiemannVisTimeFreqCalculation]`](test_tabascal_pipeline.py#L168) | Test the complete Tabascal pipeline execution. |
| [`test_tabascal_pipeline[RiemannVisTimeFreqCalculationFFI]`](test_tabascal_pipeline.py#L168) | Test the complete Tabascal pipeline execution. |
| [`test_gpgains_pipeline[GPGains]`](test_tabascal_pipeline.py#L333) | Integration test for GPGains replacing UnitaryGains in the standard pipeline. |
| [`test_phase_calculation_rfi_pipeline[FixedOrbit+PhaseCalculationRFI]`](test_tabascal_pipeline.py#L370) | Integration test for PhaseCalculationRFI in the full pipeline. |
| [`test_sgp4_component_pipeline[SGP4LEONoDragOrbit+PhaseCalculationRFI+UnitaryGains]`](test_tabascal_pipeline.py#L452) | Integration tests for SGP4LEONoDragOrbit + PhaseCalculationRFI pipelines. |
| [`test_sgp4_component_pipeline[SGP4LEONoDragOrbit+PhaseCalculationRFI+GPGains]`](test_tabascal_pipeline.py#L452) | Integration tests for SGP4LEONoDragOrbit + PhaseCalculationRFI pipelines. |
| [`test_sgp4_component_pipeline[SGP4LEOOrbit+PhaseCalculationRFI+UnitaryGains]`](test_tabascal_pipeline.py#L452) | Integration tests for SGP4LEONoDragOrbit + PhaseCalculationRFI pipelines. |
