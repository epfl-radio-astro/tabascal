# Test Inventory

## `tests/components/test_trajectory.py`

### TestPhaseCalculationRFI

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises without error with default config |
| `test_setup_validates_dimensions` | `_validate_dimensions` passes after setup; `ants_uvw` and `ants_xyz` have correct shapes |
| `test_set_params_is_identity` | `build_set_params` returns a no-op that passes the state dict through unchanged |
| `test_forward_output_shape` | Forward pass returns `rfi_phase` with shape `(n_rfi, n_ant, n_freq_fine, n_time_fine)` |
| `test_forward_phase_is_finite` | All phase values are finite for a realistic ISS-like satellite position |
| `test_forward_phase_varies_across_antennas` | Different antennas see different phase delays (phases are not identical) |
| `test_forward_preserves_rfi_xyz_in_state` | Forward pass copies `rfi_xyz` through to the output state unchanged |
| `test_parametric_sizes[2-1-1-2-1]` | Phase output has correct shape and is finite for small `(n_ant, n_rfi, n_freq, n_time, n_int_time)` |
| `test_parametric_sizes[4-2-4-8-2]` | Phase output has correct shape and is finite for medium dimensions |
| `test_parametric_sizes[8-3-2-6-3]` | Phase output has correct shape and is finite for larger dimensions |
| `test_compute_ant_pos_xyz_earth_radius` | `_compute_ant_pos`: `ants_xyz` GCRF radii lie in the Earth surface range (6.35–6.40 Mm) |
| `test_compute_ant_pos_uvw_shape_and_finite` | `_compute_ant_pos`: `ants_uvw` has shape `(n_ant, n_time_fine, 3)` with all-finite values |
| `test_compute_ant_pos_distinct_across_antennas` | `_compute_ant_pos`: two different antennas have distinct GCRF positions |

### TestFixedOrbit

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises and pre-computes trajectory without error |
| `test_rfi_xyz_shape` | Pre-computed satellite positions stored at setup have shape `(n_rfi, n_time_fine, 3)` |
| `test_rfi_phase_shape` | Pre-computed phase stored at setup has shape `(n_rfi, n_ant, n_freq, n_time_fine)` |
| `test_rfi_xyz_nonzero` | Propagated positions are non-zero (TLE orbit was actually computed) |
| `test_rfi_xyz_altitude_reasonable` | ISS satellite radius from Earth centre is in the 6–8 Mm LEO range |
| `test_rfi_phase_finite` | All pre-computed phase values are finite |
| `test_forward_adds_rfi_xyz_and_phase_to_state` | Forward pass inserts `rfi_xyz` and `rfi_phase` into the state dict |
| `test_forward_output_matches_precomputed` | Forward returns the exact arrays computed and stored during `setup` |
| `test_forward_is_deterministic` | Calling `build_forward` twice with identical inputs produces identical outputs |
| `test_two_satellites_shape` | Two distinct TLEs produce position and phase arrays of the correct shape |
| `test_two_satellites_have_different_positions` | Two distinct TLEs propagate to different positions (orbits are independent) |
| `test_build_set_params_is_identity` | `build_set_params` returns a pass-through with no side effects on the state |
| `test_compute_rfi_phase_consistent_with_get_rfi_phase` | `_compute_rfi_phase`: stored `rfi_phase` equals `get_rfi_phase` called directly with the same arrays |
| `test_compute_rfi_phase_xyz_is_satellite_altitude` | `_compute_rfi_phase`: `rfi_xyz` radii from TLE propagation lie in the LEO altitude range (6–8 Mm) |

### TestSGP4LEONoDragOrbit *(requires Space-Track credentials)*

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component fetches TLEs from Space-Track and initialises without error |
| `test_rfi_xyz_shape` | Initial `state_outputs["rfi_xyz"]` placeholder has shape `(n_rfi, n_time_fine, 3)` |
| `test_init_params_base_shape` | Initial base orbit parameters have shape `(n_rfi, 6)` — no bstar term |
| `test_prior_covariance_positive_definite` | Cholesky factor `L_rfi_orbit` has a positive diagonal for every satellite |
| `test_forward_output_shapes` | Forward pass produces `rfi_xyz` `(n_rfi, n_time_fine, 3)` and `elements` `(n_rfi, 6)` |
| `test_forward_rfi_xyz_finite` | SGP4-propagated satellite positions from the forward pass are all finite |
| `test_build_set_params_samples_correct_shapes` | NumPyro trace samples `rfi_orbit_base` with shape `(n_rfi, 6)` |
| `test_forward_transform_roundtrip` | `inv_transform(forward_transform(x)) ≈ x` to floating-point precision |
| `test_inv_transform_roundtrip` | `forward_transform(inv_transform(x)) ≈ x` to floating-point precision |
| `test_sats_init_direct_call` | `sats_init` called directly with `comp.elements` returns a satrec without error |
| `test_sats_init_produces_valid_positions` | Satrec returned by `sats_init` propagates via `sgp4jax` to finite positions of shape `(n_rfi, n_time_fine, 3)` |

### TestSGP4LEOOrbit *(requires Space-Track credentials)*

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component fetches TLEs from Space-Track and initialises without error |
| `test_rfi_xyz_shape` | Initial `state_outputs["rfi_xyz"]` placeholder has shape `(n_rfi, n_time_fine, 3)` |
| `test_init_params_base_shape` | Initial base orbit parameters have shape `(n_rfi, 7)` — includes bstar drag term |
| `test_prior_covariance_positive_definite` | Cholesky factor `L_rfi_orbit` (7×7) has a positive diagonal for every satellite |
| `test_forward_output_shapes` | Forward pass produces `rfi_xyz` `(n_rfi, n_time_fine, 3)` and `elements` `(n_rfi, 7)` |
| `test_forward_rfi_xyz_finite` | SGP4-propagated satellite positions from the forward pass are all finite |
| `test_build_set_params_samples_correct_shapes` | NumPyro trace samples `rfi_orbit_base` with shape `(n_rfi, 7)` |
| `test_forward_transform_roundtrip` | `inv_transform(forward_transform(x)) ≈ x` to floating-point precision |
| `test_inv_transform_roundtrip` | `forward_transform(inv_transform(x)) ≈ x` to floating-point precision |
| `test_sats_init_direct_call` | `sats_init` called directly with 7-element `comp.elements` returns a satrec without error |
| `test_sats_init_produces_valid_positions` | Satrec returned by `sats_init` propagates via `sgp4jax` to finite positions of shape `(n_rfi, n_time_fine, 3)` |

---

## `tests/components/test_gains.py`

### TestGainsConfigValidation

| Test name | Description |
|-----------|-------------|
| `test_null_values_get_defaults` | `None` corr_time/corr_freq values are replaced with default arrays derived from the observation grid |
| `test_explicit_values_stored_correctly` | Non-null correlation times/frequencies are stored unchanged on the component |
| `test_invalid_amp_mean_type_raises` | Non-numeric `amp_mean` raises an error during `gains_config_validation` |
| `test_invalid_phase_std_type_raises` | Non-numeric `phase_std` raises an error during `gains_config_validation` |
| `test_single_freq_single_time_defaults` | Config with `n_freq=1`, `n_time=1` and null corr values initialises without error |

### TestUnitaryGains

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises without error |
| `test_state_outputs_shapes` | `state_outputs["gains"]` placeholder has shape `(n_ant, n_freq, n_time)` |
| `test_no_learnable_params` | `init_params_base` is empty — `UnitaryGains` has no free parameters |
| `test_forward_vis_obs_equals_sum` | Forward computes `vis_obs = vis_ast + vis_rfi` exactly (unit gains applied) |
| `test_forward_preserves_other_state_keys` | Forward does not drop pre-existing keys in the state dict |
| `test_forward_output_shapes[2-1-4]` | `gains` and `vis_obs` output shapes are correct for small `(n_ant, n_freq, n_time)` |
| `test_forward_output_shapes[6-8-10]` | `gains` and `vis_obs` output shapes are correct for medium dimensions |
| `test_forward_output_shapes[16-4-12]` | `gains` and `vis_obs` output shapes are correct for large dimensions |

### TestGPGains

| Test name | Description |
|-----------|-------------|
| `test_setup_succeeds` | Component initialises without error |
| `test_prior_params_shapes` | Prior mean and Cholesky L arrays have shapes consistent with the GP parameterisation |
| `test_init_params_base_shapes` | Initial base parameter arrays have shapes matching the GP amplitude/phase decomposition |
| `test_forward_output_shapes` | Forward produces `gains` `(n_ant, n_freq, n_time)` and `vis_obs` `(n_bl, n_freq, n_time)` |
| `test_forward_gains_at_prior_mean_amplitude` | At zero base params the recovered gain amplitude equals `amp_mean` |
| `test_forward_gains_last_antenna_phase_zero` | The last antenna is used as phase reference; its gain phase is zero |
| `test_forward_output_is_complex` | `gains` and `vis_obs` from the forward pass are complex-valued |
| `test_resample_matrices_shapes` | GP resampling matrices have shapes consistent with the coarse/fine frequency and time grids |
| `test_setup_and_forward_various_sizes[2-1-4]` | Setup and forward succeed end-to-end for small `(n_ant, n_freq, n_time)` |
| `test_setup_and_forward_various_sizes[5-3-12]` | Setup and forward succeed end-to-end for medium dimensions |
| `test_setup_and_forward_various_sizes[8-4-16]` | Setup and forward succeed end-to-end for large dimensions |
| `test_build_set_params_samples_correct_shapes` | NumPyro trace samples amplitude and phase base params with shapes matching the GP grid sizes |
| `test_forward_transform_roundtrip` | `inv_transform(forward_transform(x)) ≈ x` for the gains affine reparameterisation |
| `test_inv_transform_roundtrip` | `forward_transform(inv_transform(x)) ≈ x` for the gains affine reparameterisation |

### TestBaseGPGainsInheritedSetParams

| Test name | Description |
|-----------|-------------|
| `test_unitary_gains_build_set_params_is_identity` | `UnitaryGains.build_set_params` returns the state dict unchanged (base-class default behaviour) |

---

## `tests/test_tabascal_pipeline.py`

| Test name | Description |
|-----------|-------------|
| `test_tabascal_pipeline[RiemannVisTimeFreqCalculation]` | Full pipeline (`FixedOrbit + ComplexRFI + RiemannVisTimeFreqCalculation + FourierTimeFreqGPAst + UnitaryGains`) completes and the reduced χ² matches the pinned reference value |
| `test_tabascal_pipeline[RiemannVisTimeFreqCalculationFFI]` | Same pipeline as above but with the FFI-optimised visibility kernel; reduced χ² matches the same reference value |
| `test_gpgains_pipeline[GPGains]` | Pipeline substituting `GPGains` for `UnitaryGains` completes and the reduced χ² falls in the plausible range (0, 5) |
| `test_phase_calculation_rfi_pipeline[FixedOrbit+PhaseCalculationRFI]` | Pipeline chaining `FixedOrbit` then `PhaseCalculationRFI` (phase recomputed from xyz) completes and the reduced χ² falls in (0, 5); no Space-Track credentials needed |
| `test_sgp4_component_pipeline[SGP4LEONoDragOrbit+PhaseCalculationRFI+UnitaryGains]` | Pipeline using SGP4 orbit propagation without drag term plus `UnitaryGains`; requires Space-Track; reduced χ² in (0, 5) |
| `test_sgp4_component_pipeline[SGP4LEONoDragOrbit+PhaseCalculationRFI+GPGains]` | Pipeline using SGP4 orbit propagation without drag term plus `GPGains`; requires Space-Track; reduced χ² in (0, 5) |
| `test_sgp4_component_pipeline[SGP4LEOOrbit+PhaseCalculationRFI+UnitaryGains]` | Pipeline using full SGP4 orbit (bstar drag included) plus `UnitaryGains`; requires Space-Track; reduced χ² in (0, 5) |
