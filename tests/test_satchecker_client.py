"""Offline contract tests for the single-endpoint SatChecker client."""

import json
import socket
import urllib.error

import pytest

from tabascal.satchecker import client
from tabascal.satchecker.client import (
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


@pytest.mark.parametrize("error", [socket.timeout("slow"), OSError("dropped")])
def test_network_failures_are_transport_errors(monkeypatch, error):
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(SatCheckerTransportError, match="request failed"):
        fetch_nearest_tle(25544, EPOCH)
