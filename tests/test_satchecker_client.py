"""Offline contract tests for the single-endpoint SatChecker client."""

import email.utils
import json
import socket
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from tabascal.satchecker import client
from tabascal.satchecker.client import (
    SatCheckerRateLimitError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    fetch_nearest_tle,
)

from .tle_helpers import block_network, jd, make_nearest_json  # noqa: F401


EPOCH = jd(2023, 1, 1)


def test_nearest_response_is_normalised(monkeypatch):
    monkeypatch.setattr(client, "_http_get", lambda *args, **kwargs: make_nearest_json([(25544, EPOCH)]))
    frame = fetch_nearest_tle(25544, EPOCH)
    assert list(frame.columns) == client.TLE_COLUMNS
    assert frame.loc[0, "NORAD_CAT_ID"] == 25544


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
