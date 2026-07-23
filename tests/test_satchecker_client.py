"""Offline tests for the SatChecker HTTP client (transport + normalisation).

The single network choke point ``client._http_get`` is monkeypatched to return
canned bytes routed by URL, so nothing here touches the network.
"""

import urllib.error
import urllib.request

import pytest

from tabascal.satchecker import client
from tabascal.satchecker.client import (
    CatalogueResult,
    SatCheckerError,
    fetch_full_catalogue,
    fetch_nearest_tle,
)

from .tle_helpers import (
    jd,
    make_empty_zip_bytes,
    make_info_json,
    make_json_page,
    make_nearest_json,
    make_zip_bytes,
)

_EPOCH = jd(2023, 2, 21, 13)


def _route(monkeypatch, handler):
    """Install a ``_http_get`` that dispatches on the request URL."""
    monkeypatch.setattr(client, "_http_get", lambda url, timeout=client.REQUEST_TIMEOUT: handler(url))


class TestFullCatalogue:

    def test_zip_response_is_normalised(self, monkeypatch):
        pairs = [(25544, _EPOCH), (38833, _EPOCH)]

        def handler(url):
            if "format=zip" in url:
                return make_zip_bytes(pairs)
            return make_info_json(total=2, version="1.6.0")

        _route(monkeypatch, handler)
        result = fetch_full_catalogue(_EPOCH)
        assert isinstance(result, CatalogueResult)
        assert result.source == "zip"
        assert result.expected_count == 2 and result.actual_count == 2
        assert result.service_version == "1.6.0"
        assert sorted(result.records["NORAD_CAT_ID"]) == [25544, 38833]
        assert list(result.records.columns) == client.CATALOGUE_COLUMNS

    def test_truncated_zip_falls_back_to_json(self, monkeypatch):
        # info says 4 records; the zip only ever yields 1 (< 99%), so after two
        # zip attempts the client falls back to the paginated JSON endpoint.
        full = [(1, _EPOCH), (2, _EPOCH), (3, _EPOCH), (4, _EPOCH)]

        def handler(url):
            if "format=zip" in url:
                return make_zip_bytes([(1, _EPOCH)])
            if "per_page=1" in url:
                return make_info_json(total=4)
            return make_json_page(full, total=4)

        _route(monkeypatch, handler)
        result = fetch_full_catalogue(_EPOCH)
        assert result.source == "json"
        assert result.actual_count == 4
        assert sorted(result.records["NORAD_CAT_ID"]) == [1, 2, 3, 4]

    def test_paginated_json_spans_multiple_pages(self, monkeypatch):
        # Drive _fetch_catalogue_json directly with a small per_page so >1 page is
        # required (total 3, per_page 2 -> pages 1 and 2).
        page1 = [(1, _EPOCH), (2, _EPOCH)]
        page2 = [(3, _EPOCH)]

        def handler(url):
            return make_json_page(page2 if "&page=2" in url else page1, total=3)

        _route(monkeypatch, handler)
        df = client._fetch_catalogue_json(_EPOCH, per_page=2)
        assert sorted(df["NORAD_CAT_ID"]) == [1, 2, 3]

    def test_empty_zip_and_no_json_raises(self, monkeypatch):
        def handler(url):
            if "format=zip" in url:
                return make_empty_zip_bytes()
            if "per_page=1" in url:
                return make_info_json(total=5)
            return make_json_page([], total=0)  # JSON yields nothing either

        _route(monkeypatch, handler)
        with pytest.raises(SatCheckerError):
            fetch_full_catalogue(_EPOCH)

    def test_http_error_becomes_satchecker_error(self, monkeypatch):
        # Patch at the urlopen layer so the real _http_get wrapping is exercised.
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(SatCheckerError):
            fetch_full_catalogue(_EPOCH)

    def test_unavailable_expected_count_skips_unvalidated_zip(self, monkeypatch):
        calls = {"zip": 0, "json": 0}

        def handler(url):
            if "format=zip" in url:
                calls["zip"] += 1
                return make_zip_bytes([(1, _EPOCH)])
            if "per_page=1" in url:
                return b"[]"  # catalogue_info cannot establish completeness
            calls["json"] += 1
            return make_json_page([(1, _EPOCH), (2, _EPOCH)], total=2)

        _route(monkeypatch, handler)
        result = fetch_full_catalogue(_EPOCH)
        assert result.source == "json"
        assert sorted(result.records["NORAD_CAT_ID"]) == [1, 2]
        assert calls == {"zip": 0, "json": 1}


class TestNearestTle:

    def test_nearest_response_is_normalised(self, monkeypatch):
        _route(monkeypatch, lambda url: make_nearest_json([(25544, _EPOCH)]))
        df = fetch_nearest_tle(25544, _EPOCH)
        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert list(df.columns) == client.CATALOGUE_COLUMNS

    def test_no_record_returns_empty(self, monkeypatch):
        _route(monkeypatch, lambda url: b'{"orbital_data": []}')
        assert fetch_nearest_tle(99999, _EPOCH).empty

    def test_malformed_json_becomes_satchecker_error(self, monkeypatch):
        _route(monkeypatch, lambda url: b"<html>not json</html>")
        with pytest.raises(SatCheckerError):
            fetch_nearest_tle(25544, _EPOCH)

    def test_timeout_becomes_satchecker_error(self, monkeypatch):
        def boom(req, timeout=None):
            raise TimeoutError("timed out")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(SatCheckerError):
            fetch_nearest_tle(25544, _EPOCH)


class TestZipParsing:

    def test_bad_zip_bytes_become_satchecker_error(self, monkeypatch):
        _route(monkeypatch, lambda url: b"this is not a zip file")
        with pytest.raises(SatCheckerError):
            client._fetch_catalogue_zip(_EPOCH)


class TestJsonPaginationValidation:
    """Completion is judged by record count, never by ``page * per_page``, and an
    incomplete/inconsistent JSON response is rejected rather than returned."""

    def test_premature_empty_page_raises(self, monkeypatch):
        # page 1: 2 rows, total 4; page 2: 0 rows, total 4 -> truncated -> raise.
        def handler(url):
            if "&page=2" in url:
                return make_json_page([], total=4)
            return make_json_page([(1, _EPOCH), (2, _EPOCH)], total=4)

        _route(monkeypatch, handler)
        with pytest.raises(SatCheckerError):
            client._fetch_catalogue_json(_EPOCH, per_page=2)

    def test_short_effective_page_size_is_followed(self, monkeypatch):
        # Request a large per_page but the service caps each page at 2 rows,
        # total 5: pages 1, 2 and 3 must all be requested and all 5 rows returned.
        requested_pages = []

        def handler(url):
            page = 1
            for part in url.split("&"):
                if part.startswith("page="):
                    page = int(part.split("=")[1])
            requested_pages.append(page)
            rows = {1: [(1, _EPOCH), (2, _EPOCH)],
                    2: [(3, _EPOCH), (4, _EPOCH)],
                    3: [(5, _EPOCH)]}[page]
            return make_json_page(rows, total=5)

        _route(monkeypatch, handler)
        df = client._fetch_catalogue_json(_EPOCH, per_page=100)
        assert sorted(requested_pages) == [1, 2, 3]
        assert sorted(df["NORAD_CAT_ID"]) == [1, 2, 3, 4, 5]

    def test_inconsistent_totals_raise(self, monkeypatch):
        def handler(url):
            if "&page=2" in url:
                return make_json_page([(3, _EPOCH), (4, _EPOCH)], total=5)  # total changed
            return make_json_page([(1, _EPOCH), (2, _EPOCH)], total=4)

        _route(monkeypatch, handler)
        with pytest.raises(SatCheckerError):
            client._fetch_catalogue_json(_EPOCH, per_page=2)

    def test_complete_final_short_page_succeeds(self, monkeypatch):
        # Pages of 2, 2, 1 rows, total 5: accumulated count matches total -> ok.
        def handler(url):
            page = 1
            for part in url.split("&"):
                if part.startswith("page="):
                    page = int(part.split("=")[1])
            rows = {1: [(1, _EPOCH), (2, _EPOCH)],
                    2: [(3, _EPOCH), (4, _EPOCH)],
                    3: [(5, _EPOCH)]}[page]
            return make_json_page(rows, total=5)

        _route(monkeypatch, handler)
        df = client._fetch_catalogue_json(_EPOCH, per_page=2)
        assert len(df) == 5

    def test_repeated_page_cannot_satisfy_total(self, monkeypatch):
        # A count-only check would accept [1, 2, 1, 2] as four complete rows.
        _route(
            monkeypatch,
            lambda url: make_json_page([(1, _EPOCH), (2, _EPOCH)], total=4),
        )
        with pytest.raises(SatCheckerError, match="duplicate NORAD IDs"):
            client._fetch_catalogue_json(_EPOCH, per_page=2)

    def test_mismatched_response_page_raises(self, monkeypatch):
        import json

        def handler(url):
            payload = json.loads(make_json_page([(1, _EPOCH)], total=2))
            payload["page"] = 99
            return json.dumps(payload).encode()

        _route(monkeypatch, handler)
        with pytest.raises(SatCheckerError, match="when page 1 was requested"):
            client._fetch_catalogue_json(_EPOCH, per_page=1)


class TestResponseShapes:
    """Empty and scalar top-level payloads must not leak IndexError/AttributeError."""

    def test_catalogue_info_empty_list_raises(self, monkeypatch):
        _route(monkeypatch, lambda url: b"[]")
        with pytest.raises(SatCheckerError):
            client.catalogue_info(_EPOCH)

    def test_catalogue_info_scalar_raises(self, monkeypatch):
        _route(monkeypatch, lambda url: b"42")
        with pytest.raises(SatCheckerError):
            client.catalogue_info(_EPOCH)

    def test_nearest_empty_list_is_not_found(self, monkeypatch):
        _route(monkeypatch, lambda url: b"[]")
        assert fetch_nearest_tle(25544, _EPOCH).empty

    def test_nearest_scalar_raises(self, monkeypatch):
        _route(monkeypatch, lambda url: b"42")
        with pytest.raises(SatCheckerError):
            fetch_nearest_tle(25544, _EPOCH)
