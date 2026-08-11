"""Offline tests for :mod:`tabascal.tle_config`.

Two responsibilities, tested separately:

* **Configuration normalisation** — the single path both the preflight check and
  the actual resolution build their inputs from. Every malformed value must
  surface as :class:`TLEConfigurationError`, which the CLI renders as one line
  instead of a traceback, and never as a raw pandas/NumPy exception.
* **Observation epoch derivation** — the single Measurement Set epoch helper.
  Preflight and execution must agree exactly on it: it sets the canonical cache
  bucket *and* every TLE age comparison, so a divergence would mean the run was
  checked at one instant and modelled at another.
"""

import numpy as np
import pytest

from tabascal import tle_config
from tabascal.tle_config import TLEConfigurationError
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

    def test_null_norad_ids_with_a_tle_model_is_a_clean_error(self):
        config = {
            "model": {"components": ["trajectory:FixedOrbit"]},
            "satellites": {"norad_ids": None},
        }
        with pytest.raises(TLEConfigurationError, match="no NORAD catalogue IDs"):
            tle_config.normalise_tle_config(config)

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
        "field", ["extra_tle_max_age_days", "remote_tle_max_age_days", "tle_catalogue_settle_days"]
    )
    @pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "soon", []])
    def test_malformed_ages_are_configuration_errors(self, field, value):
        with pytest.raises(TLEConfigurationError, match=field):
            tle_config.normalise_tle_config(_config(**{field: value}))

    @pytest.mark.parametrize("value", [0, -2, float("inf"), "often", None])
    def test_malformed_intervals_are_configuration_errors(self, value):
        with pytest.raises(TLEConfigurationError, match="tle_catalogue_interval_hours"):
            tle_config.normalise_tle_config(_config(tle_catalogue_interval_hours=value))

    def test_sub_second_bucket_is_rejected(self):
        with pytest.raises(TLEConfigurationError, match="1 second"):
            tle_config.normalise_tle_config(_config(tle_catalogue_interval_hours=1e-5))

    @pytest.mark.parametrize("value", [0, -1, "later", None])
    def test_malformed_provisional_hours_are_configuration_errors(self, value):
        with pytest.raises(TLEConfigurationError, match="tle_provisional_cache_hours"):
            tle_config.normalise_tle_config(_config(tle_provisional_cache_hours=value))

    def test_defaults_are_the_documented_ones(self):
        cfg = tle_config.normalise_tle_config(_config(norad_ids=[25544]))
        assert cfg.extra_tle_max_age_days is None      # exact replay stays possible
        assert cfg.remote_tle_max_age_days == 3.0
        assert cfg.catalogue_settle_days == 45.0
        assert cfg.provisional_cache_hours == 12.0
        assert cfg.catalogue_interval_hours == 2.0

    def test_null_ages_are_explicit_opt_outs(self):
        cfg = tle_config.normalise_tle_config(
            _config(remote_tle_max_age_days=None, tle_catalogue_settle_days=None)
        )
        assert cfg.remote_tle_max_age_days is None
        assert cfg.catalogue_settle_days is None

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

    def test_matches_read_ms_row_selection_and_unit_guard(self, monkeypatch):
        times = self._times_seconds()
        monkeypatch.setattr(tle_config, "_ms_time_column", lambda ms: times)
        # What read_ms computes: one timestamp per integration, seconds -> days.
        read_ms_times = times.reshape(4, 3)[:, 0] / 86400.0
        expected = float(mjd_to_jd(read_ms_times.mean()))
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(expected, abs=1e-12)

    def test_days_column_is_not_rescaled(self, monkeypatch):
        days = np.repeat((_OBS - 2400000.5) + np.arange(4) * 8.0 / 86400.0, 3)
        monkeypatch.setattr(tle_config, "_ms_time_column", lambda ms: days)
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(_OBS + 1.5 * 8 / 86400, abs=1e-9)

    def test_pre_1970_epoch_supported(self, monkeypatch):
        old = jd(1965, 3, 4, 6, 0)
        base = (old - 2400000.5) * 86400.0
        monkeypatch.setattr(
            tle_config, "_ms_time_column", lambda ms: np.repeat(base + np.arange(3) * 8.0, 2)
        )
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(old + 8.0 / 86400, abs=1e-9)

    def test_single_integration_unit_inferred_from_magnitude(self, monkeypatch):
        base = (_OBS - 2400000.5) * 86400.0
        monkeypatch.setattr(tle_config, "_ms_time_column", lambda ms: np.full(5, base))
        assert tle_config.ms_observation_epoch_jd("ms") == pytest.approx(_OBS, abs=1e-9)

    def test_empty_time_column_is_an_error(self, monkeypatch):
        monkeypatch.setattr(tle_config, "_ms_time_column", lambda ms: np.array([]))
        with pytest.raises(TLEError, match="empty TIME column"):
            tle_config.ms_observation_epoch_jd("ms")
