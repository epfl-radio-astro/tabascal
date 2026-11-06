"""Chunk tuning reaches both dask-ms paths without changing the default."""
import pytest
from tabascal import ms, write


@pytest.mark.parametrize("chunk", [None, 100000])
@pytest.mark.parametrize("writer", [False, True])
def test_chunk_forwarding(monkeypatch, chunk, writer):
    module = write if writer else ms
    class ReachedReader(Exception):
        pass
    def reader(path, **kwargs):
        expected = {"column_keywords": True}
        if chunk is not None:
            expected["chunks"] = {"row": chunk}
        assert kwargs == expected
        raise ReachedReader
    monkeypatch.setattr(module, "xds_from_ms", reader)
    monkeypatch.setattr(write, "is_process_0", lambda: True)
    with pytest.raises(ReachedReader):
        if writer:
            write.write_results_ms("test.ms", "results.zarr", row_chunk=chunk)
        else:
            ms.read_ms("test.ms", row_chunk=chunk)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5, "10000"])
def test_invalid_chunk(chunk):
    with pytest.raises(ValueError, match="data.row_chunk"):
        ms.ms_row_chunks(chunk)


@pytest.mark.parametrize("chunk", [None, 100000])
def test_config_forwards_chunk(monkeypatch, chunk):
    from types import SimpleNamespace
    from tabascal import config
    class ReachedReader(Exception):
        pass
    def reader(*args, **kwargs):
        assert kwargs == {"row_chunk": chunk}
        raise ReachedReader
    monkeypatch.setattr(config, "read_ms", reader)
    tab_config = SimpleNamespace(ms_path="test.ms", args={"data": {"row_chunk": chunk}})
    with pytest.raises(ReachedReader):
        config.TabConfig.read_ms_params(tab_config, None, "xx", "DATA")
