"""The optional MS export leaves the results zarr as the recorded output."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tabascal import tab_tools
from tabascal.scripts import _run_tabascal_impl as runner
from tabascal.scripts.run_tabascal import build_parser


@pytest.mark.parametrize("skip", [None, False, True])
@pytest.mark.parametrize("path", ["initial.zarr", "optimized.zarr"])
def test_export_policy(monkeypatch, capsys, skip, path):
    writer = Mock()
    monkeypatch.setattr(tab_tools, "write_results_ms", writer)
    data = {"data_col": "DATA", "row_chunk": 100000}
    if skip is not None:
        data["skip_ms_write"] = skip
    config = SimpleNamespace(args={"data": data}, gain_table=["external.B"])
    tab_tools.write_ms_if_enabled(config, "input.ms", path)
    assert path in capsys.readouterr().out
    if skip:
        writer.assert_not_called()
    else:
        writer.assert_called_once_with("input.ms", path, "DATA", gain_table=["external.B"], row_chunk=100000)


@pytest.mark.parametrize("cli, configured, expected", [(False, False, False), (False, True, True), (True, False, True)])
def test_cli_and_config_policy(monkeypatch, cli, configured, expected):
    argv = ["run", "-c", "config.yaml"] + (["--skip-ms-write"] if cli else [])
    args = build_parser().parse_args(argv)
    config = {"data": {"skip_ms_write": configured}}
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(runner, "set_precision", lambda _: None)
    monkeypatch.setattr(runner, "is_process_0", lambda: False)
    subtract = Mock()
    monkeypatch.setattr(runner, "tabascal_subtraction", subtract)
    runner.run(args)
    assert subtract.call_args.args[0]["data"]["skip_ms_write"] is expected


@pytest.mark.parametrize("skip", [False, True])
def test_optimized_run_keeps_zarr(monkeypatch, skip):
    import jax.numpy as jnp
    data = jnp.ones((1, 1, 4), dtype=complex)
    config = SimpleNamespace(
        args={"data": {"data_col": "DATA", "skip_ms_write": skip},
              "opt": {"max_iter": 1, "epsilon": 0.1, "dual_run": False}},
        vis_obs=data, noise=1., flags=jnp.zeros(data.shape, dtype=bool),
    )
    pred = {"vis_obs": data[None]}
    monkeypatch.setattr(tab_tools, "loss_trace_path", lambda _: None)
    monkeypatch.setattr(tab_tools, "run_custom_svi", lambda **kw: SimpleNamespace(params={}, losses=[1.]))
    monkeypatch.setattr(tab_tools, "Predictive", lambda **kw: lambda *a, **k: pred)
    zarr_writer, ms_writer = Mock(), Mock()
    monkeypatch.setattr(tab_tools, "write_results_xds", zarr_writer)
    monkeypatch.setattr(tab_tools, "write_results_ms", ms_writer)
    result = tab_tools.run_opt(config, None, [None, None], {}, "input.ms", "optimized.zarr", "params")
    zarr_writer.assert_called_once_with(pred, config, "optimized.zarr")
    assert ms_writer.call_count == (0 if skip else 1)
    assert result[0] is pred
