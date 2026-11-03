"""Run directories: their creation, the record written into them, and reading one back."""

import os
from datetime import datetime

import pytest
import yaml

import tabascal.run_dir as run_dir_mod
import tabascal.write as write_mod
from tabascal.scripts import results_to_MS

NOW = datetime(2026, 11, 3, 9, 15, 0)


@pytest.fixture
def impl():
    from tabascal.scripts import _run_tabascal_impl

    return _run_tabascal_impl


def test_runs_get_distinct_directories_and_latest_follows(tmp_path):
    """Same name, same second: the token tells them apart, and latest names the newer."""
    first = run_dir_mod.create_run_dir(str(tmp_path), "sweep")
    second = run_dir_mod.create_run_dir(str(tmp_path), "sweep")

    assert first != second
    assert os.path.basename(first).startswith("sweep_")
    assert os.readlink(tmp_path / "runs" / "latest") == os.path.basename(second)


def test_an_existing_directory_is_never_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dir_mod.secrets, "token_hex", lambda n: "abcdef")
    run_dir_mod.make_run_dir(str(tmp_path), "sweep", NOW)

    with pytest.raises(FileExistsError):
        run_dir_mod.make_run_dir(str(tmp_path), "sweep", NOW)


@pytest.mark.parametrize("error", [None, RuntimeError("out of memory")])
def test_the_record_is_written_first_and_closed_with_the_outcome(
    impl, tmp_path, monkeypatch, error
):
    """config.yaml and a running run.json exist before the work; the status follows it."""
    monkeypatch.chdir(tmp_path)
    config = {
        "data": {"ms_path": "obs.ms", "gain_table": "flux.B"},  # recorded as a list
        "rfi": {"est": "curves.npz"},
        "gains": {"init": "prior"},
        "opt": {"trace_path": "trace.npz"},
        "model": {},
    }
    seen = {}

    def subtract(config, paths):
        with open(os.path.join(paths.run_dir, "config.yaml")) as f:
            seen["config"] = yaml.safe_load(f)
        seen["status"] = run_dir_mod.read_record(paths.run_dir)["status"]
        seen["run_dir"] = paths.run_dir
        if error is not None:
            raise error

    monkeypatch.setattr(impl, "_subtract", subtract)

    if error is None:
        impl.tabascal_subtraction(config, None, name="t", log=False)
    else:
        with pytest.raises(RuntimeError):
            impl.tabascal_subtraction(config, None, name="t", log=False)

    data = seen["config"]["data"]
    assert seen["status"] == "running"
    assert data["ms_path"] == str(tmp_path / "obs.ms")
    assert data["gain_table"] == [str(tmp_path / "flux.B")]
    assert seen["config"]["rfi"]["est"] == str(tmp_path / "curves.npz")
    assert seen["config"]["gains"]["init"] == "prior"
    assert seen["config"]["opt"]["trace_path"] == os.path.join(seen["run_dir"], "trace.npz")

    record = run_dir_mod.read_record(seen["run_dir"])
    assert record["status"] == ("finished" if error is None else "failed")
    assert set(record["inputs"]) == {"data.ms_path", "data.gain_table[0]", "rfi.est", "data.truth_zarr"}


def test_a_failed_record_stops_the_run_before_it_starts(impl, tmp_path, monkeypatch):
    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(impl, "start_record", broken)
    monkeypatch.setattr(impl, "_subtract", lambda config, paths: pytest.fail("ran"))

    with pytest.raises(OSError, match="disk full"):
        impl.tabascal_subtraction(
            {"data": {"ms_path": str(tmp_path / "o.ms")}, "model": {}}, None, name="t", log=False
        )


def test_inputs_through_latest_are_resolved_once_and_shared(tmp_path, monkeypatch):
    """Rank 0 resolves latest to the earlier run; a worker arriving after it moved gets the same."""
    import tabascal.distributed as dist

    earlier = run_dir_mod.create_run_dir(str(tmp_path), "first")
    via_latest = {"satellites": {"extra_orbit_dir": str(tmp_path / "runs" / "latest" / "results")}}
    sent = {}
    broadcast = dist.broadcast_bytes_from_rank0
    monkeypatch.setattr(
        dist, "broadcast_bytes_from_rank0",
        lambda payload, name: sent.setdefault("payload", broadcast(payload, name)),
    )
    rank0 = {"satellites": dict(via_latest["satellites"])}
    run = run_dir_mod.create_run_dir(str(tmp_path), "t", rank0)

    # latest now names the new run, so a worker resolving for itself would differ.
    monkeypatch.setattr(dist, "is_process_0", lambda: False)
    worker = {"satellites": dict(via_latest["satellites"])}

    assert run_dir_mod.create_run_dir(str(tmp_path), "t", worker) == run
    assert worker == rank0 == {"satellites": {"extra_orbit_dir": os.path.join(earlier, "results")}}


@pytest.mark.parametrize(
    "section, key, changed, warns",
    [
        ("rfi", "est", "", True),  # a plain file
        ("rfi", "est", "part", True),  # a directory
        ("satellites", "extra_orbit_dir", "orbits.json", True),
        # An earlier run's results/: tab2MS rewrites its .B, which no orbit reader reads.
        ("satellites", "extra_orbit_dir", "map_pred.B/table.dat", False),
    ],
)
def test_a_changed_input_is_named(tmp_path, section, key, changed, warns):
    path = tmp_path / "input"
    files = [path] if not changed else [path / "part", path / "orbits.json", path / "map_pred.B" / "table.dat"]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("one")
    record = {"inputs": run_dir_mod.fingerprint_inputs({section: {key: str(path)}})}

    (path / changed if changed else path).write_text("three")

    if warns:
        with pytest.warns(RuntimeWarning, match=rf"{section}\.{key}"):
            assert run_dir_mod.check_inputs(record) == [f"{section}.{key}"]
    else:
        assert run_dir_mod.check_inputs(record) == []


def _fake_run(tmp_path, opt, max_iter, gain_table=None):
    run = run_dir_mod.create_run_dir(str(tmp_path), "r")
    config = {
        "data": {"ms_path": str(tmp_path / "obs.ms"), "data_col": "DATA", "gain_table": gain_table},
        "inference": {"opt": opt},
        "opt": {"max_iter": max_iter},
    }
    run_dir_mod.write_config(run, config)
    inputs = run_dir_mod.fingerprint_inputs({"data": {"gain_table": gain_table}})
    run_dir_mod.write_record(run, {"status": "finished", "inputs": inputs})
    for zarr in ("init_pred.zarr", "map_pred.zarr"):
        os.mkdir(os.path.join(run, "results", zarr))
    return run


@pytest.mark.parametrize(
    "opt, max_iter, exported",
    [(True, 10, "map_pred.zarr"), (True, 0, "init_pred.zarr"), (False, 10, "init_pred.zarr")],
)
def test_tab2ms_exports_what_the_run_did(tmp_path, monkeypatch, opt, max_iter, exported):
    """Read through runs/latest, written from the record, the .B left in the run's results."""
    run = _fake_run(tmp_path, opt, max_iter)
    captured = {}
    monkeypatch.setattr(write_mod, "write_results_ms", lambda **kw: captured.update(kw))

    results_to_MS.run(results_to_MS.build_parser().parse_args([str(tmp_path / "runs" / "latest")]))

    assert captured["results_zarr_path"] == os.path.join(run, "results", exported)
    assert captured["ms_path"] == str(tmp_path / "obs.ms")
    assert captured["data_col"] == "DATA"
    assert captured.get("caltable_path") is None


@pytest.mark.parametrize("flag, field", [("-m", "ms_path"), ("-gt", "gain_table")])
def test_tab2ms_warns_when_an_override_differs(tmp_path, monkeypatch, recwarn, flag, field):
    """And fingerprints the override: a moved gain table identical to the recorded one is fine."""
    recorded = tmp_path / "flux.B"
    recorded.write_text("table")
    run = _fake_run(tmp_path, True, 10, gain_table=str(recorded))
    moved = tmp_path / "moved"
    moved.write_text("table")
    recorded.unlink()
    captured = {}
    monkeypatch.setattr(write_mod, "write_results_ms", lambda **kw: captured.update(kw))

    results_to_MS.run(results_to_MS.build_parser().parse_args([run, flag, str(moved)]))

    messages = [str(w.message) for w in recwarn]
    assert any(m.startswith(f"data.{field}: using") and str(moved) in m for m in messages)
    assert captured[field] in (str(moved), [str(moved)])
    # Only the -m run still compares the recorded table, which is gone.
    changed = [m for m in messages if "is not what the run read" in m]
    assert len(changed) == (0 if flag == "-gt" else 1)
    assert all("data.gain_table[0]" in m for m in changed)


@pytest.mark.parametrize(
    "old",
    [["-z", "r.zarr"], ["--results_zarr_path", "r.zarr"], ["-d", "DATA"],
     ["--data_col", "DATA"], ["-o", "x.B"], ["--caltable-path", "x.B"]],
)
def test_tab2ms_old_flags_name_the_new_command(capsys, old):
    with pytest.raises(SystemExit):
        results_to_MS.build_parser().parse_args(["-m", "a.ms", *old])

    assert "tab2MS <out_dir>/runs/<run>" in capsys.readouterr().err
