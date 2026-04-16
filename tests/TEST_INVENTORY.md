# Test Inventory

## `tests/components/test_trajectory.py`

No Space-Track credentials are required for any test in this file. `TestSGP4LEONoDragOrbit` and `TestSGP4LEOOrbit` use a bundled TLE cache file (`tabascal/data/tles/2026-04-15-bundled.json`, NORAD IDs 20452 / 38833) and a fixed observation epoch matching that file's date prefix, so `get_tles_by_id` reads from disk and never contacts the Space-Track API.

### TestPhaseCalculationRFI

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises without error with a default mock config |
| `test_setup_validates_dimensions` | After `setup`, `_validate_dimensions` passes and `ants_uvw` / `ants_xyz` have shape `(n_ant, n_time_fine, 3)` |
| `test_set_params_is_identity` | `build_set_params` returns a no-op that passes the state dict through unchanged |
| `test_forward_output_shape` | Forward pass produces `rfi_phase` with shape `(n_rfi, n_ant, n_freq_fine, n_time_fine)` |
| `test_forward_phase_is_finite` | All computed phase values are finite for a realistic ISS-like satellite position |
| `test_forward_phase_varies_across_antennas` | Different antennas produce different phase delays (phases are not all identical) |
| `test_forward_preserves_rfi_xyz_in_state` | Forward pass copies `rfi_xyz` through to the output state unchanged |
| `test_parametric_sizes[2-1-1-2-1]` | Output shape and finiteness verified for `(n_ant=2, n_rfi=1, n_freq=1, n_time=2, n_int_time=1)` |
| `test_parametric_sizes[4-2-4-8-2]` | Output shape and finiteness verified for medium dimensions |
| `test_parametric_sizes[8-3-2-6-3]` | Output shape and finiteness verified for larger dimensions |
| `test_compute_ant_pos_xyz_earth_radius` | `_compute_ant_pos`: `ants_xyz` GCRF radii lie in the Earth-surface band (6.35–6.40 Mm) |
| `test_compute_ant_pos_uvw_shape_and_finite` | `_compute_ant_pos`: `ants_uvw` has shape `(n_ant, n_time_fine, 3)` with all-finite values |
| `test_compute_ant_pos_distinct_across_antennas` | `_compute_ant_pos`: two different antennas have distinct GCRF positions at the same time step |

### TestFixedOrbit

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component propagates the TLE orbit and pre-computes phase without error |
| `test_rfi_xyz_shape` | Pre-computed satellite positions stored at setup have shape `(n_rfi, n_time_fine, 3)` |
| `test_rfi_phase_shape` | Pre-computed phase stored at setup has shape `(n_rfi, n_ant, n_freq, n_time_fine)` |
| `test_rfi_xyz_nonzero` | Propagated positions are non-zero (TLE orbit was actually computed, not a zero placeholder) |
| `test_rfi_xyz_altitude_reasonable` | ISS GCRF radius from Earth centre is in the 6–8 Mm LEO range |
| `test_rfi_phase_finite` | All pre-computed phase values are finite |
| `test_forward_adds_rfi_xyz_and_phase_to_state` | Forward pass inserts `rfi_xyz` and `rfi_phase` into the state dict |
| `test_forward_output_matches_precomputed` | Forward returns the exact `rfi_xyz` and `rfi_phase` arrays stored during `setup` |
| `test_forward_is_deterministic` | Calling `build_forward` twice with identical inputs produces identical results |
| `test_two_satellites_shape` | Two distinct TLEs produce position and phase arrays of the correct shape |
| `test_two_satellites_have_different_positions` | Two distinct TLEs propagate to different positions (orbits are independent) |
| `test_build_set_params_is_identity` | `build_set_params` returns a pass-through with no side effects on the state dict |
| `test_compute_rfi_phase_consistent_with_get_rfi_phase` | `_compute_rfi_phase`: stored `rfi_phase` equals `get_rfi_phase` called directly with the same arrays |
| `test_compute_rfi_phase_xyz_is_satellite_altitude` | `_compute_rfi_phase`: `rfi_xyz` radii from TLE propagation lie in the LEO altitude range (6–8 Mm) |

### TestSGP4LEONoDragOrbit

Uses the bundled TLE cache; no Space-Track credentials required.

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component loads TLEs from the repo cache and initialises without error |
| `test_rfi_xyz_shape` | Initial `state_outputs["rfi_xyz"]` placeholder has shape `(n_rfi, n_time_fine, 3)` |
| `test_init_params_base_shape` | Initial base orbit parameters have shape `(n_rfi, 6)` — bstar is excluded from the learnable parameters |
| `test_prior_covariance_positive_definite` | Cholesky factor `L_rfi_orbit` has a positive diagonal for every satellite |
| `test_forward_output_shapes` | Forward pass produces `rfi_xyz` `(n_rfi, n_time_fine, 3)` and `elements` `(n_rfi, 6)` |
| `test_forward_rfi_xyz_finite` | SGP4-propagated satellite positions from the forward pass are all finite |
| `test_build_set_params_samples_correct_shapes` | NumPyro trace samples `rfi_orbit_base` with shape `(n_rfi, 6)` |
| `test_forward_transform_roundtrip` | `inv_transform(forward_transform(x)) ≈ x` to floating-point precision |
| `test_inv_transform_roundtrip` | `forward_transform(inv_transform(x)) ≈ x` to floating-point precision |
| `test_sats_init_direct_call` | `sats_init` called directly with `comp.elements` returns a satrec object without error |
| `test_sats_init_produces_valid_positions` | Satrec from `sats_init` propagates via `sgp4jax` to finite positions of shape `(n_rfi, n_time_fine, 3)` |

### TestSGP4LEOOrbit

Uses the bundled TLE cache; no Space-Track credentials required.

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component loads TLEs from the repo cache and initialises without error |
| `test_rfi_xyz_shape` | Initial `state_outputs["rfi_xyz"]` placeholder has shape `(n_rfi, n_time_fine, 3)` |
| `test_init_params_base_shape` | Initial base orbit parameters have shape `(n_rfi, 7)` — bstar drag term is included as a learnable parameter |
| `test_prior_covariance_positive_definite` | Cholesky factor `L_rfi_orbit` (7×7) has a positive diagonal for every satellite |
| `test_forward_output_shapes` | Forward pass produces `rfi_xyz` `(n_rfi, n_time_fine, 3)` and `elements` `(n_rfi, 7)` |
| `test_forward_rfi_xyz_finite` | SGP4-propagated satellite positions from the forward pass are all finite |
| `test_build_set_params_samples_correct_shapes` | NumPyro trace samples `rfi_orbit_base` with shape `(n_rfi, 7)` |
| `test_forward_transform_roundtrip` | `inv_transform(forward_transform(x)) ≈ x` to floating-point precision |
| `test_inv_transform_roundtrip` | `forward_transform(inv_transform(x)) ≈ x` to floating-point precision |
| `test_sats_init_direct_call` | `sats_init` called directly with 7-element `comp.elements` (including bstar) returns a satrec without error |
| `test_sats_init_produces_valid_positions` | Satrec from `sats_init` propagates via `sgp4jax` to finite positions of shape `(n_rfi, n_time_fine, 3)` |

---

## `tests/components/test_gains.py`

### TestGainsConfigValidation

| Test name | Description |
|-----------|-------------|
| `test_null_values_get_defaults` | `None` corr_time / corr_freq values are replaced with defaults derived from the observation grid |
| `test_explicit_values_stored_correctly` | Non-null correlation times and frequencies are stored unchanged on the component |
| `test_invalid_amp_mean_type_raises` | A non-numeric `amp_mean` raises an error during `gains_config_validation` |
| `test_invalid_phase_std_type_raises` | A non-numeric `phase_std` raises an error during `gains_config_validation` |
| `test_single_freq_single_time_defaults` | Config with `n_freq=1`, `n_time=1` and null corr values initialises without error |

### TestUnitaryGains

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises without error |
| `test_state_outputs_shapes` | `state_outputs["gains"]` placeholder has shape `(n_ant, n_freq, n_time)` |
| `test_no_learnable_params` | `init_params_base` is empty — `UnitaryGains` has no free parameters |
| `test_forward_vis_obs_equals_sum` | Forward computes `vis_obs = vis_ast + vis_rfi` exactly (unit gains applied, no scaling) |
| `test_forward_preserves_other_state_keys` | Forward does not drop pre-existing keys in the state dict |
| `test_forward_output_shapes[2-1-4]` | `gains` and `vis_obs` output shapes are correct for `(n_ant=2, n_freq=1, n_time=4)` |
| `test_forward_output_shapes[6-8-10]` | `gains` and `vis_obs` output shapes are correct for medium dimensions |
| `test_forward_output_shapes[16-4-12]` | `gains` and `vis_obs` output shapes are correct for large dimensions |

### TestGPGains

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises without error |
| `test_prior_params_shapes` | Prior mean and Cholesky L arrays have shapes consistent with the GP parameterisation |
| `test_init_params_base_shapes` | Initial base parameter arrays match the GP amplitude and phase grid sizes |
| `test_forward_output_shapes` | Forward produces `gains` `(n_ant, n_freq, n_time)` and `vis_obs` `(n_bl, n_freq, n_time)` |
| `test_forward_gains_at_prior_mean_amplitude` | At zero base params the recovered gain amplitude equals `amp_mean` |
| `test_forward_gains_last_antenna_phase_zero` | The last antenna is used as phase reference and its gain phase is zero |
| `test_forward_output_is_complex` | `gains` and `vis_obs` from the forward pass are complex-valued |
| `test_resample_matrices_shapes` | GP resampling matrices have shapes consistent with the coarse/fine frequency and time grids |
| `test_setup_and_forward_various_sizes[2-1-4]` | Setup and forward succeed end-to-end for `(n_ant=2, n_freq=1, n_time=4)` |
| `test_setup_and_forward_various_sizes[5-3-12]` | Setup and forward succeed end-to-end for medium dimensions |
| `test_setup_and_forward_various_sizes[8-4-16]` | Setup and forward succeed end-to-end for large dimensions |
| `test_build_set_params_samples_correct_shapes` | NumPyro trace samples amplitude and phase base params with shapes matching the GP grid sizes |
| `test_forward_transform_roundtrip` | `inv_transform(forward_transform(x)) ≈ x` for the gains affine reparameterisation |
| `test_inv_transform_roundtrip` | `forward_transform(inv_transform(x)) ≈ x` for the gains affine reparameterisation |

### TestBaseGPGainsInheritedSetParams

| Test name | Description |
|-----------|-------------|
| `test_unitary_gains_build_set_params_is_identity` | `UnitaryGains.build_set_params` returns the state dict unchanged (inherited base-class default) |

---

## `tests/components/test_rfi_vis.py`

Tests for `RiemannVisTimeFreqCalculation` and `RiemannVisTimeFreqCalculationFFI`. Each parametrized over three size tuples: `(1,1,1,1,1,1)`, `(4,5,6,7,8,9)`, `(64,20,16,12,4,2)`.

| Test name | Description |
|-----------|-------------|
| `test_ffi[1-1-1-1-1-1]` | FFI and reference Riemann kernels produce identical `vis_rfi` for minimal (all-ones) dimensions |
| `test_ffi[4-5-6-7-8-9]` | FFI and reference kernels agree for medium dimensions |
| `test_ffi[64-20-16-12-4-2]` | FFI and reference kernels agree for large dimensions |
| `test_ffi_jvp[1-1-1-1-1-1]` | Forward-mode Jacobian-vector products of FFI and reference kernels match for minimal dimensions |
| `test_ffi_jvp[4-5-6-7-8-9]` | JVP outputs agree for medium dimensions |
| `test_ffi_jvp[64-20-16-12-4-2]` | JVP outputs agree for large dimensions |
| `test_ffi_vjp[1-1-1-1-1-1]` | Reverse-mode VJP gradients w.r.t. `rfi_A` and `rfi_phase` match between FFI and reference for minimal dimensions |
| `test_ffi_vjp[4-5-6-7-8-9]` | VJP gradients agree for medium dimensions |
| `test_ffi_vjp[64-20-16-12-4-2]` | VJP gradients agree for large dimensions |

---

## `tests/test_fft_gp.py`

### TestSupersample

| Test name | Description |
|-----------|-------------|
| `test_supersample_domain_specs_1d` | `supersample_domain_specs` returns correct `n_ss` and `dx_ss` for a 1D domain |
| `test_supersample_domain_specs_2d` | `supersample_domain_specs` returns correct specs for a 2D domain with mixed factors |
| `test_supersample_domain_specs_length_mismatch` | `supersample_domain_specs` raises `ValueError` when input lengths differ |
| `test_supersample_domain_1d` | `supersample_domain` returns a domain twice as long with the correct start point |
| `test_supersample_domain_k_1d` | `supersample_domain_k` returns a k-domain array twice as long |
| `test_supersample_signal_1d` | `supersample` doubles the length of a 1D signal and preserves total energy |
| `test_supersample_signal_2d` | `supersample` doubles both dimensions of a 2D constant signal and preserves the constant value |
| `test_supersample_jit_compatible` | `supersample` can be called inside `jax.jit` without error |

### TestDomainK

| Test name | Description |
|-----------|-------------|
| `test_domain_k_1d` | `domain_k` produces a k-array of the correct length with zero at the centre |
| `test_domain_k_2d` | `domain_k` produces two k-arrays matching the lengths of the input 2D domain |

### TestPadding

| Test name | Description |
|-----------|-------------|
| `test_pad_domain_specs_validation` | `pad_domain_specs` raises `ValueError` for pad factors < 1.0 and for length mismatches |
| `test_pad_domain_specs_1d` | `pad_domain_specs` computes the correct number of padding samples for a factor-2 pad |
| `test_pad_domain_1d` | `pad_domain` returns a domain of the expected padded length |
| `test_pad_domain_k_1d` | `pad_domain_k` returns a k-domain of the expected padded length |
| `test_pad_signal_1d` | `pad` zero-pads a 1D array and preserves the original values in the central region |
| `test_pad_signal_2d` | `pad` pads both dimensions of a 2D array |
| `test_pad_signal_dimension_mismatch` | `pad` raises `ValueError` when the number of pad factors does not match the array rank |
| `test_pad_jit_compatible` | `pad` can be called inside `jax.jit` without error |

### TestFourierCutting

| Test name | Description |
|-----------|-------------|
| `test_pk_cut_1d` | `pk_cut` returns a single slice and pad that covers fewer modes than the full 1D spectrum |
| `test_pk_cut_2d` | `pk_cut` returns two slices and two pads for a 2D power spectrum |
| `test_fourier_cut_uncut_roundtrip` | `fourier_uncut(fourier_cut(pk, cutoff, y), ...)` restores the original array shape |
| `test_supersample_fourier_1d` | `supersample_fourier` doubles the size of a 1D Fourier array |
| `test_supersample_fourier_2d` | `supersample_fourier` doubles both dimensions of a 2D Fourier array |

### TestPowerSpectrum

| Test name | Description |
|-----------|-------------|
| `test_pow_spec_1d_shape` | `pow_spec` output has the same shape as the input k-array |
| `test_pow_spec_1d_symmetry` | `pow_spec` is symmetric around k=0 |
| `test_pow_spec_1d_peak_at_zero` | `pow_spec` peaks at k=0 |
| `test_pow_spec_nd_validation` | `pow_spec_nd` raises `ValueError` when `k0s` and `gammas` lengths differ from the number of k-arrays |
| `test_pow_spec_nd_2d_shape` | `pow_spec_nd` produces a `(21, 31)` array for two k-arrays of those lengths |
| `test_pow_spec_nd_3d_shape` | `pow_spec_nd` produces an `(11, 11, 11)` array for three equal-length k-arrays |

### TestDomainSS

| Test name | Description |
|-----------|-------------|
| `test_domain_ss_1d` | `domain_ss` halves the grid spacing for a 2× supersample factor in 1D |
| `test_domain_ss_2d` | `domain_ss` returns two domain arrays for a 2D input |

### TestLatentSpace

| Test name | Description |
|-----------|-------------|
| `test_signal_to_latent_1d` | `signal_to_latent` compresses a 1D signal into a latent representation no larger than the original |
| `test_latent_init_returns_correct_types` | `latent_to_signal_init` returns `jnp.ndarray`, `list`, `list`, and `list[slice]` in the correct order |
| `test_latent_init_predict_roundtrip` | `latent_to_signal_init` + `latent_to_signal` produce an output of at least the original size |
| `test_latent_operations_jit_compatible` | `latent_to_signal` can be wrapped in `jax.jit` and returns a non-empty array |
| `test_signal_to_latent_jit_compatible` | JIT-compiled `signal_to_latent` matches the non-JIT result exactly |

### TestJAXCompatibility

| Test name | Description |
|-----------|-------------|
| `test_pk_cut_jit` | `pk_cut` called outside JIT returns a slice and pad (documents that it is a setup function, not JIT-able) |

### TestNumericalAccuracy

| Test name | Description |
|-----------|-------------|
| `test_supersample_preserves_dc_component` | Supersampling preserves the DC Fourier component to floating-point precision |
| `test_fourier_cut_uncut_preserves_kept_modes` | After cut and uncut, kept Fourier modes are bit-for-bit identical to the originals |

---

## `tests/test_timing.py`

| Test name | Description |
|-----------|-------------|
| `test_timing_collection_enabled` | `@measure_runtime` records a timing entry when timings are enabled |
| `test_timing_collection_disabled` | `@measure_runtime` does not record any timing entry when timings are disabled |
| `test_hierarchical_timing` | Nested `@measure_runtime` decorators produce parent/child timing entries with correct call counts |
| `test_timer_context_manager` | The `timer()` context manager records a named timing block when timings are enabled |
| `test_measure_runtime_data_structures` | `@measure_runtime` correctly wraps a function returning nested dict/list/tuple of JAX arrays |
| `test_measure_runtime_mixed_types` | `@measure_runtime` handles functions with both JAX array and plain Python return values |

---

## `tests/test_tabascal_pipeline.py`

End-to-end integration tests. Each test invokes `run_tabascal.py` as a subprocess, checks `returncode == 0`, and validates the `Reduced Chi^2 @ opt params` value printed to stdout.

`TabConfig` fetches TLEs using the mean observation epoch from the MS file. The simulation data (from HuggingFace) uses the tabsim default epoch of 2023-02-21; the repo ships `tabascal/data/tles/2023-02-21-HMZGLE.json` containing NORAD IDs 20452, 38833, and 45854 (all three listed in `tests/data/tab_target.yaml`), so the `TabConfig` TLE lookup is always satisfied from disk. Tests that additionally use `SGP4LEONoDragOrbit` or `SGP4LEOOrbit` call `fetch_standard_orbital_elements` a second time during component `setup`; that second call is also satisfied by the same bundled cache file — but only when the component's observation epoch likewise falls on 2023-02-21. Tests marked **requires Space-Track** are skipped automatically when `tabascal.tle.load_spacetrack_credentials` returns `(None, None)`.

| Test name | Description |
|-----------|-------------|
| `test_tabascal_pipeline[RiemannVisTimeFreqCalculation]` | Full pipeline with `FixedOrbit + ComplexRFI + RiemannVisTimeFreqCalculation + FourierTimeFreqGPAst + UnitaryGains`; reduced χ² must match the pinned reference value to 1% |
| `test_tabascal_pipeline[RiemannVisTimeFreqCalculationFFI]` | Same pipeline as above but with the FFI-optimised visibility kernel; reduced χ² must match the same reference value to 1% |
| `test_gpgains_pipeline[GPGains]` | Pipeline substituting `GPGains` for `UnitaryGains`; verifies the pipeline completes and reduced χ² lies in the plausible range (0, 5) |
| `test_phase_calculation_rfi_pipeline[FixedOrbit+PhaseCalculationRFI]` | Pipeline chaining `FixedOrbit` then `PhaseCalculationRFI` (phase recomputed from xyz rather than stored at setup); reduced χ² in (0, 5); no Space-Track required |
| `test_sgp4_component_pipeline[SGP4LEONoDragOrbit+PhaseCalculationRFI+UnitaryGains]` | Pipeline using SGP4 orbit propagation without drag term combined with `UnitaryGains`; **requires Space-Track**; reduced χ² in (0, 5) |
| `test_sgp4_component_pipeline[SGP4LEONoDragOrbit+PhaseCalculationRFI+GPGains]` | Same orbit model as above but with `GPGains`; **requires Space-Track**; reduced χ² in (0, 5) |
| `test_sgp4_component_pipeline[SGP4LEOOrbit+PhaseCalculationRFI+UnitaryGains]` | Pipeline using full SGP4 orbit with bstar drag included as a learnable parameter combined with `UnitaryGains`; **requires Space-Track**; reduced χ² in (0, 5) |
