"""Offline contract tests for the two-endpoint SatChecker client."""

import email.utils
import json
import socket
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from tabascal.satchecker import client
from tabascal.satchecker.records import record_epoch_jd
from tabascal.satchecker.client import (
    SatCheckerRateLimitError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    fetch_nearest_omm,
    fetch_nearest_tle,
)

from .tle_helpers import (  # noqa: F401
    block_network,
    jd,
    make_nearest_json,
    make_nearest_omm_json,
)


EPOCH = jd(2023, 1, 1)


def test_nearest_response_is_normalised(monkeypatch):
    monkeypatch.setattr(client, "_http_get", lambda *args, **kwargs: make_nearest_json([(25544, EPOCH)]))
    frame = fetch_nearest_tle(25544, EPOCH)
    assert list(frame.columns) == client.TLE_COLUMNS + ["RECORD_KIND"]
    assert frame.loc[0, "NORAD_CAT_ID"] == 25544
    assert frame.loc[0, "RECORD_KIND"] == "tle"


@pytest.mark.parametrize("payload", [b"[]", b'[{"orbital_data": []}]'])
def test_no_record_is_an_empty_frame(monkeypatch, payload):
    monkeypatch.setattr(client, "_http_get", lambda *args, **kwargs: payload)
    assert fetch_nearest_tle(99999, EPOCH).empty


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"satellite_id": None, "tle_line1": "x", "tle_line2": "y"}], "missing satellite IDs"),
        ([{"satellite_id": 1.5, "tle_line1": "x", "tle_line2": "y"}], "non-integer"),
        ([{"satellite_id": 1, "tle_line1": None, "tle_line2": "y"}], "missing TLE_LINE1"),
    ],
)
def test_malformed_rows_raise_response_error(monkeypatch, rows, message):
    payload = json.dumps({"orbital_data": rows}).encode()
    monkeypatch.setattr(client, "_http_get", lambda *args, **kwargs: payload)
    with pytest.raises(SatCheckerResponseError, match=message):
        fetch_nearest_tle(1, EPOCH)


def test_invalid_json_is_a_response_error(monkeypatch):
    monkeypatch.setattr(client, "_http_get", lambda *args, **kwargs: b"not-json")
    with pytest.raises(SatCheckerResponseError, match="invalid JSON"):
        fetch_nearest_tle(25544, EPOCH)


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_client_statuses_are_response_errors(monkeypatch, status):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("url", status, "reason", {}, None)

    monkeypatch.setattr(client.urllib.request, "urlopen", fail)
    with pytest.raises(SatCheckerResponseError, match=f"HTTP {status}"):
        fetch_nearest_tle(25544, EPOCH)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_backoff_and_server_statuses_are_transport_errors(monkeypatch, status):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("url", status, "reason", {}, None)

    monkeypatch.setattr(client.urllib.request, "urlopen", fail)
    with pytest.raises(SatCheckerTransportError, match=f"HTTP {status}"):
        fetch_nearest_tle(25544, EPOCH)


def _raise_429(monkeypatch, headers):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("url", 429, "Too Many Requests", headers, None)

    monkeypatch.setattr(client.urllib.request, "urlopen", fail)


def test_rate_limit_is_its_own_error_type(monkeypatch):
    """429 must be distinguishable: it is about this client, not this satellite."""
    _raise_429(monkeypatch, {})
    with pytest.raises(SatCheckerRateLimitError) as caught:
        fetch_nearest_tle(25544, EPOCH)
    assert caught.value.retry_after is None
    # Still a transport error, so a batch stops on it rather than working onward.
    assert isinstance(caught.value, SatCheckerTransportError)


def test_retry_after_delta_seconds_is_reported(monkeypatch):
    _raise_429(monkeypatch, {"Retry-After": "120"})
    with pytest.raises(SatCheckerRateLimitError, match="120 s before the next request"):
        fetch_nearest_tle(25544, EPOCH)


def test_retry_after_http_date_is_reported(monkeypatch):
    """RFC 9110 permits an HTTP-date as well as delta-seconds."""
    when = datetime.now(timezone.utc) + timedelta(seconds=90)
    _raise_429(monkeypatch, {"Retry-After": email.utils.format_datetime(when)})
    with pytest.raises(SatCheckerRateLimitError) as caught:
        fetch_nearest_tle(25544, EPOCH)
    assert caught.value.retry_after == pytest.approx(90, abs=5)


@pytest.mark.parametrize("value", ["", "not-a-date", "Mon, 99 Xxx 9999"])
def test_unusable_retry_after_does_not_become_a_second_failure(monkeypatch, value):
    """A bad hint must degrade to 'no hint', never to an exception of its own."""
    _raise_429(monkeypatch, {"Retry-After": value})
    with pytest.raises(SatCheckerRateLimitError) as caught:
        fetch_nearest_tle(25544, EPOCH)
    assert caught.value.retry_after is None


def test_elapsed_retry_after_clamps_to_zero(monkeypatch):
    when = datetime.now(timezone.utc) - timedelta(hours=1)
    _raise_429(monkeypatch, {"Retry-After": email.utils.format_datetime(when)})
    with pytest.raises(SatCheckerRateLimitError) as caught:
        fetch_nearest_tle(25544, EPOCH)
    assert caught.value.retry_after == 0.0


@pytest.mark.parametrize("error", [socket.timeout("slow"), OSError("dropped")])
def test_network_failures_are_transport_errors(monkeypatch, error):
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(SatCheckerTransportError, match="request failed"):
        fetch_nearest_tle(25544, EPOCH)


# ---------------------------------------------------------------------------
# get-nearest-omm
# ---------------------------------------------------------------------------

class TestFetchNearestOmm:
    """The second endpoint, and the ways its response differs from the first.

    The envelope is the same and the transport, retry and status classification
    are shared verbatim. What is new is a level of nesting, an epoch that
    appears twice in two different spellings, and the absence of any structural
    check as strong as a TLE checksum.
    """

    def _serve(self, monkeypatch, payload):
        monkeypatch.setattr(client, "_http_get", lambda *a, **k: payload)

    def test_response_is_normalised(self, monkeypatch):
        self._serve(monkeypatch, make_nearest_omm_json([(25544, EPOCH)]))
        frame = fetch_nearest_omm(25544, EPOCH)
        assert list(frame.columns) == client.OMM_COLUMNS + ["RECORD_KIND"]
        assert frame.loc[0, "NORAD_CAT_ID"] == 25544
        assert frame.loc[0, "RECORD_KIND"] == "omm"
        assert frame.loc[0, "OBJECT_ID"] == "1998-067A"

    def test_nested_elements_are_lifted_onto_flat_columns(self, monkeypatch):
        self._serve(monkeypatch, make_nearest_omm_json([(25544, EPOCH)]))
        frame = fetch_nearest_omm(25544, EPOCH)
        for column in (
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ECCENTRICITY",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
        ):
            assert isinstance(frame.loc[0, column], float), column
        assert "orbital_elements" not in frame.columns

    def test_the_nested_iso_epoch_wins_over_the_row_level_one(self, monkeypatch):
        # The row carries "2023-01-01 00:00:00 UTC", which is neither ISO 8601
        # nor sub-second; the nested object carries the parseable spelling. Take
        # the wrong one and every OMM record fails to yield an epoch at all.
        self._serve(monkeypatch, make_nearest_omm_json([(25544, EPOCH)]))
        frame = fetch_nearest_omm(25544, EPOCH)
        assert "UTC" not in frame.loc[0, "EPOCH"]
        assert record_epoch_jd(frame.loc[0]) == pytest.approx(EPOCH, abs=1e-6)

    def test_unused_element_fields_are_dropped(self, monkeypatch):
        self._serve(monkeypatch, make_nearest_omm_json([(25544, EPOCH)]))
        frame = fetch_nearest_omm(25544, EPOCH)
        for dropped in ("REV_AT_EPOCH", "ELEMENT_SET_NO", "MEAN_MOTION_DOT"):
            assert dropped not in frame.columns

    @pytest.mark.parametrize("payload", [b"[]", b'[{"orbital_data": []}]'])
    def test_no_record_is_an_empty_frame(self, monkeypatch, payload):
        # Confirmed against the live service: an unknown catalogue number comes
        # back as HTTP 200 with an empty orbital_data, not as a 404.
        self._serve(monkeypatch, payload)
        assert fetch_nearest_omm(999998, EPOCH).empty

    def test_a_row_without_the_nested_object_is_a_response_error(self, monkeypatch):
        payload = json.dumps([{"orbital_data": [{"satellite_id": 25544}]}]).encode()
        self._serve(monkeypatch, payload)
        with pytest.raises(SatCheckerResponseError, match="no orbital_elements"):
            fetch_nearest_omm(25544, EPOCH)

    @pytest.mark.parametrize(
        "missing",
        ["EPOCH", "INCLINATION", "MEAN_MOTION", "BSTAR", "ECCENTRICITY"],
    )
    def test_a_missing_element_field_is_a_response_error(self, monkeypatch, missing):
        raw = json.loads(make_nearest_omm_json([(25544, EPOCH)]))
        del raw[0]["orbital_data"][0]["orbital_elements"][missing]
        self._serve(monkeypatch, json.dumps(raw).encode())
        with pytest.raises(SatCheckerResponseError, match=f"missing {missing}"):
            fetch_nearest_omm(25544, EPOCH)

    @pytest.mark.parametrize(
        "satellite_id, message",
        [
            (None, "missing satellite IDs"),
            (1.5, "non-integer"),
            ("abc", "non-numeric"),
        ],
    )
    def test_the_id_checks_are_shared_with_the_tle_path(
        self, monkeypatch, satellite_id, message
    ):
        raw = json.loads(make_nearest_omm_json([(25544, EPOCH)]))
        raw[0]["orbital_data"][0]["satellite_id"] = satellite_id
        self._serve(monkeypatch, json.dumps(raw).encode())
        with pytest.raises(SatCheckerResponseError, match=message):
            fetch_nearest_omm(25544, EPOCH)

    def test_a_single_row_object_is_accepted(self, monkeypatch):
        raw = json.loads(make_nearest_omm_json([(25544, EPOCH)]))
        raw[0]["orbital_data"] = raw[0]["orbital_data"][0]
        self._serve(monkeypatch, json.dumps(raw).encode())
        assert len(fetch_nearest_omm(25544, EPOCH)) == 1

    def test_transport_failures_are_not_endpoint_specific(self, monkeypatch):
        def boom(*a, **k):
            raise SatCheckerTransportError("down")

        monkeypatch.setattr(client, "_http_get", boom)
        with pytest.raises(SatCheckerTransportError):
            fetch_nearest_omm(25544, EPOCH)

    def test_the_request_names_the_omm_endpoint(self, monkeypatch):
        seen = {}

        def capture(url, *a, **k):
            seen["url"] = url
            return make_nearest_omm_json([(25544, EPOCH)])

        monkeypatch.setattr(client, "_http_get", capture)
        fetch_nearest_omm(25544, EPOCH)
        assert "get-nearest-omm" in seen["url"]
        assert f"epoch={repr(float(EPOCH))}" in seen["url"].replace("%20", " ")

    def test_a_clamped_pre_handover_record_is_returned_not_hidden(self, monkeypatch):
        # The service answers a pre-handover request with its earliest record
        # instead of reporting that it has none. The client's job is to hand
        # that back faithfully; rejecting it on age is the policy layer's.
        earliest = jd(2026, 7, 11, 19, 56)
        self._serve(monkeypatch, make_nearest_omm_json([(25544, earliest)]))
        frame = fetch_nearest_omm(25544, jd(2021, 11, 1))
        assert len(frame) == 1
        assert record_epoch_jd(frame.loc[0]) == pytest.approx(earliest, abs=1e-6)
