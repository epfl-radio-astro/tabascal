"""Offline tests for :mod:`tabascal.orbit_config`.

Two responsibilities, tested separately:

* **Configuration normalisation** — the single path both the preflight check and
  the actual resolution build their inputs from. Every malformed value must
  surface as :class:`TLEConfigurationError`, which the CLI renders as one line
  instead of a traceback, and never as a raw pandas/NumPy exception.
* **Observation epoch derivation** — the single Measurement Set epoch helper.
  Preflight and execution must agree exactly on it: it sets every TLE age
  comparison, so a divergence would mean the run was
  checked at one instant and modelled at another.
"""

import numpy as np
import pytest

from tabascal import orbit_config as tle_config
from tabascal.orbit_config import TLEConfigurationError
from tabascal.time import mjd_to_jd

from .tle_helpers import (
    block_network,  # noqa: F401  autouse fixture: no live SatChecker access
    jd,
)

TLEError = tle_config.TLEError

_OBS = jd(2023, 2, 21, 12, 30)


# ---------------------------------------------------------------------------
# Configuration normalisation
# ---------------------------------------------------------------------------


def _config(**satellites):
    return {"model": {"components": []}, "satellites": satellites}


class TestConfigNormalisation:

    def test_null_norad_ids_is_an_empty_list_not_a_traceback(self):
        assert tle_config.normalise_tle_config(_config(norad_ids=None)).norad_ids == []

    @pytest.mark.parametrize(
        "component", ["trajectory:FixedOrbit", "trajectory.FixedOrbit"]
    )
    def test_null_norad_ids_with_a_tle_model_is_a_clean_error(self, component):
        # import_components accepts both separators, so the guard has to see
        # both. A dotted reference that slipped past it would leave the run
        # modelling no satellites at all instead of stopping here.
        config = {
            "model": {"components": [component]},
            "satellites": {"norad_ids": None},
        }
        with pytest.raises(TLEConfigurationError, match="no NORAD catalogue IDs"):
            tle_config.normalise_tle_config(config)

    def test_the_guard_names_only_trajectories_that_exist(self):
        """The hand-maintained list must not outlive a component it names.

        A deleted or renamed trajectory left in it can send a config that names
        the old class off to hunt for NORAD IDs it does not need, instead of
        being told by the importer that the component is gone: under
        ``model.precision: double`` this guard is reached first, since the
        precision check that resolves the components returns early there.
        """
        from tabascal.components import in_tree_components

        trajectories = {
            ref.split(":")[1]
            for ref in in_tree_components()
            if ref.startswith("trajectory:")
        }
        assert tle_config._TLE_TRAJECTORY_COMPONENTS <= trajectories

    def test_ids_are_deduplicated_preserving_order(self):
        cfg = tle_config.normalise_tle_config(_config(norad_ids=[333, 111, 333, 222]))
        assert cfg.norad_ids == [333, 111, 222]

    @pytest.mark.parametrize(
        "value", [25544.5, float("nan"), float("inf"), "not-an-id", None, -5, 0, True]
    )
    def test_malformed_ids_are_configuration_errors(self, value):
        with pytest.raises(TLEConfigurationError):
            tle_config.normalise_norad_ids([25544, value])

    @pytest.mark.parametrize("value", [25544, "25544", 25544.0])
    def test_integral_ids_are_accepted(self, value):
        assert tle_config.normalise_norad_ids([value]) == [25544]

    @pytest.mark.parametrize(
        "field",
        [
            "extra_orbit_max_age_days",
            "remote_max_age_days",
            "cache_reuse_max_age_days",
        ],
    )
    @pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "soon", []])
    def test_malformed_ages_are_configuration_errors(self, field, value):
        with pytest.raises(TLEConfigurationError, match=field):
            tle_config.normalise_tle_config(_config(**{field: value}))

    def test_cache_reuse_age_above_the_ceiling_is_a_configuration_error(self):
        with pytest.raises(TLEConfigurationError, match="must not exceed"):
            tle_config.normalise_tle_config(
                _config(remote_max_age_days=1, cache_reuse_max_age_days=2)
            )

    def test_cache_reuse_age_equal_to_the_ceiling_is_allowed(self):
        cfg = tle_config.normalise_tle_config(
            _config(remote_max_age_days=2, cache_reuse_max_age_days=2)
        )
        assert cfg.cache_reuse_max_age_days == 2.0

    def test_cache_reuse_age_with_a_null_ceiling_is_allowed(self):
        cfg = tle_config.normalise_tle_config(
            _config(remote_max_age_days=None, cache_reuse_max_age_days=5)
        )
        assert cfg.cache_reuse_max_age_days == 5.0

    def test_defaults_are_the_documented_ones(self):
        cfg = tle_config.normalise_tle_config(_config(norad_ids=[25544]))
        assert cfg.extra_orbit_max_age_days is None      # exact replay stays possible
        assert cfg.remote_max_age_days == 3.0
        assert cfg.cache_reuse_max_age_days == 1.0

    def test_null_ages_are_explicit_opt_outs(self):
        cfg = tle_config.normalise_tle_config(
            _config(remote_max_age_days=None, cache_reuse_max_age_days=None)
        )
        assert cfg.remote_max_age_days is None
        assert cfg.cache_reuse_max_age_days is None

    def test_configuration_error_is_reported_without_a_traceback_by_the_cli(self):
        # run_tabascal catches TLEError; TLEConfigurationError must be caught by it.
        assert issubclass(TLEConfigurationError, TLEError)
        assert issubclass(TLEConfigurationError, ValueError)


class TestNoradIdFile:

    def _write(self, tmp_path, text):
        path = tmp_path / "norad_ids.txt"
        path.write_text(text)
        return path

    def test_reads_one_id_per_line(self, tmp_path):
        path = self._write(tmp_path, "25544\n38833\n20452\n")
        assert tle_config.read_norad_ids_file(path) == [25544, 38833, 20452]

    def test_blank_lines_and_comments_are_ignored(self, tmp_path):
        path = self._write(tmp_path, "# GPS\n\n25544  \n\t38833\n# trailing\n")
        assert tle_config.read_norad_ids_file(path) == [25544, 38833]

    def test_inline_comments_are_stripped(self, tmp_path):
        path = self._write(tmp_path, "25544  # ISS\n38833 # NAVSTAR\n")
        assert tle_config.read_norad_ids_file(path) == [25544, 38833]

    def test_duplicates_collapse_preserving_order(self, tmp_path):
        path = self._write(tmp_path, "333\n111\n333\n")
        assert tle_config.read_norad_ids_file(path) == [333, 111]

    def test_errors_name_the_offending_line(self, tmp_path):
        path = self._write(tmp_path, "25544\nnot-a-number\n38833\n")
        with pytest.raises(TLEConfigurationError, match=r"norad_ids\.txt:2"):
            tle_config.read_norad_ids_file(path)

    def test_missing_file_is_a_configuration_error(self, tmp_path):
        with pytest.raises(TLEConfigurationError, match="could not be read"):
            tle_config.read_norad_ids_file(tmp_path / "nope.txt")

    def test_path_takes_precedence_over_the_inline_list(self, tmp_path):
        path = self._write(tmp_path, "25544\n")
        cfg = tle_config.normalise_tle_config(
            _config(norad_ids=[111, 222], norad_ids_path=str(path))
        )
        assert cfg.norad_ids == [25544]

    def test_cli_override_takes_precedence_over_both(self, tmp_path):
        from_config = self._write(tmp_path, "111\n")
        cli = tmp_path / "cli.txt"
        cli.write_text("999\n")
        cfg = tle_config.normalise_tle_config(
            _config(norad_ids=[222], norad_ids_path=str(from_config)),
            norad_ids_path_override=str(cli),
        )
        assert cfg.norad_ids == [999]


# ---------------------------------------------------------------------------
# Measurement Set observation epoch
# ---------------------------------------------------------------------------

class TestMeasurementSetEpoch:
    """The one MS epoch helper both preflight and execution derive their epoch from."""

    def _times_seconds(self, n_time=4, n_bl=3):
        # MJD seconds, as an MS stores them: one block of baselines per integration.
        base = (_OBS - 2400000.5) * 86400.0
        return np.repeat(base + np.arange(n_time) * 8.0, n_bl)

    def _column(self, monkeypatch, times, scale="utc", unit=None):
        """Patch the one seam with a TIME column and what it declares.

        ``unit=None`` is an MS that declares no ``QuantumUnits``, which is the
        case the magnitude heuristic exists for.
        """

        monkeypatch.setattr(
            tle_config, "_ms_times_and_scale", lambda ms: (times, scale, unit)
        )

    def test_matches_read_ms_row_selection_and_unit_guard(self, monkeypatch):
        times = self._times_seconds()
        self._column(monkeypatch, times)
        # What read_ms computes: one timestamp per integration, seconds -> days,
        # converted to Julian Dates and averaged in that order -- the same
        # reduction observation_epoch_jd applies to what read_ms returned.
        read_ms_times = times.reshape(4, 3)[:, 0] / 86400.0
        expected = float(mjd_to_jd(read_ms_times).mean())
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(expected, abs=1e-12)

    def test_days_column_is_not_rescaled(self, monkeypatch):
        days = np.repeat((_OBS - 2400000.5) + np.arange(4) * 8.0 / 86400.0, 3)
        self._column(monkeypatch, days)
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(_OBS + 1.5 * 8 / 86400, abs=1e-9)

    def test_pre_1970_epoch_supported(self, monkeypatch):
        old = jd(1965, 3, 4, 6, 0)
        base = (old - 2400000.5) * 86400.0
        self._column(monkeypatch, np.repeat(base + np.arange(3) * 8.0, 2))
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(old + 8.0 / 86400, abs=1e-9)

    def test_single_integration_unit_inferred_from_magnitude(self, monkeypatch):
        base = (_OBS - 2400000.5) * 86400.0
        self._column(monkeypatch, np.full(5, base))
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(_OBS, abs=1e-9)

    def test_empty_time_column_is_an_error(self, monkeypatch):
        self._column(monkeypatch, np.array([]))
        with pytest.raises(TLEError, match="empty TIME column"):
            tle_config.ms_observation_epoch_jd("ms")


class TestMeasurementSetEpochScale:
    """The epoch is a physical instant, so it is read on the scale the MS declares.

    Preflight runs before ``read_ms``, and everything it decides -- which record
    is nearest, whether that record is inside the age limits, which archive to
    ask -- is measured from this epoch. Reading a TAI column as UTC would put it
    37 s late while the fit itself, which normalises the times, propagated at the
    right instant: the two would then disagree about which TLE the run was
    checked against, and an age limit tight enough to resolve 37 s would reject
    a record that is in fact exact.
    """

    LEAP_SECS = 37.0

    def _times_seconds(self, n_time=4, n_bl=3):
        base = (_OBS - 2400000.5) * 86400.0
        return np.repeat(base + np.arange(n_time) * 8.0, n_bl)

    def _epoch(self, monkeypatch, times, scale, unit=None):
        monkeypatch.setattr(
            tle_config, "_ms_times_and_scale", lambda ms: (times, scale, unit)
        )
        return tle_config.ms_observation_epoch_jd("ms")

    def test_two_columns_naming_one_instant_give_one_epoch(self, monkeypatch):
        """The check the ruling asked for: same instant, two declarations, one epoch.

        A TAI column reads 37 s higher than a UTC column covering the same
        instants, so honouring the declaration has to bring the two back
        together -- to the ~40 us an f64 Julian Date resolves to, not to the
        37 s that dropping the declaration would leave.
        """
        on_utc = self._times_seconds()
        on_tai = on_utc + self.LEAP_SECS

        from_utc = self._epoch(monkeypatch, on_utc, "utc")
        from_tai = self._epoch(monkeypatch, on_tai, "tai")

        assert (from_tai - from_utc) * 86400.0 == pytest.approx(0.0, abs=1e-3)

    def test_the_same_numbers_on_tai_are_an_earlier_epoch(self, monkeypatch):
        times = self._times_seconds()

        from_utc = self._epoch(monkeypatch, times, "utc")
        from_tai = self._epoch(monkeypatch, times, "tai")

        assert (from_tai - from_utc) * 86400.0 == pytest.approx(-37.0, abs=1e-3)

    def test_a_utc_column_is_unchanged(self, monkeypatch):
        """The common case goes through no scale arithmetic at all."""
        times = self._times_seconds()
        expected = float(mjd_to_jd(times.reshape(4, 3)[:, 0] / 86400.0).mean())

        assert self._epoch(monkeypatch, times, "utc") == expected

    def test_a_reference_that_is_not_a_time_scale_stops_preflight(self, monkeypatch):
        """Better than aging TLEs against a sidereal angle."""
        with pytest.raises(ValueError, match="valid Measurement Set epoch reference"):
            self._epoch(monkeypatch, self._times_seconds(), "gast")

    def test_the_integration_times_stay_on_the_declared_scale(self, monkeypatch):
        """``ms_integration_times_mjd`` reports the column, not the instant.

        It exists to mirror ``read_ms``'s row selection and unit rule, and
        ``read_ms`` leaves ``times_mjd`` on the declared scale too, so the two
        stay comparable. Only the epoch, which is a physical instant, is moved.
        """
        times = self._times_seconds()
        monkeypatch.setattr(
            tle_config, "_ms_times_and_scale", lambda ms: (times, "tai", None)
        )

        np.testing.assert_allclose(
            tle_config.ms_integration_times_mjd("ms"),
            np.unique(times) / 86400.0,
            rtol=1e-15,
        )
