"""The ``tabascal`` CLI: its argument surface, and what ``run()`` always reports.

Every command form the documentation shows a user must actually parse. The
parser tests exercise the parser alone, so nothing is imported from the heavy
JAX/NumPyro run implementation and nothing is executed; the ``run()`` tests
import that module lazily and stub the run itself out.
"""

import re
from pathlib import Path

import pytest

from tabascal.scripts.run_tabascal import build_parser

_DOCS = Path(__file__).parent.parent / "docs"

# A shell prompt some docs put in front of a command.
_PROMPT = re.compile(r"^\$\s+")


def documented_commands(docs_dir=_DOCS):
    """Every ``tabascal ...`` command in the docs' fenced code blocks.

    Only fenced code is scanned. A prose sentence that happens to start a line
    with the word "tabascal" is not a command, and treating it as one turns a
    docs edit into a confusing parser failure.
    """

    commands = []
    for page in sorted(docs_dir.glob("*.md")):
        in_code = False
        for line in page.read_text().splitlines():
            line = line.strip()
            if line.startswith("```"):
                in_code = not in_code
                continue
            if not in_code:
                continue
            line = _PROMPT.sub("", line)
            # `tabascal ...` as a command, not `tabascal/` in a directory tree.
            if not re.match(r"^tabascal(\s|$)", line):
                continue
            # Strip trailing comments used to annotate help invocations.
            line = re.split(r"\s+#", line, maxsplit=1)[0]
            commands.append((page.name, line.split()))
    return commands


def _parse(*argv):
    return build_parser().parse_args(list(argv))


def _raise(error):
    """A no-argument stub that raises ``error``."""

    def stub():
        raise error

    return stub


class TestRunSubcommand:

    def test_config_and_sim_dir(self):
        args = _parse("run", "-c", "tab_target.yaml", "-s", "sim_dir")
        assert args.command == "run"
        assert args.config == "tab_target.yaml"
        assert args.sim_dir == "sim_dir"

    def test_config_and_ms_path(self):
        args = _parse("run", "-c", "config.yaml", "-ms", "file.ms")
        assert args.ms_path == "file.ms"

    def test_extra_orbit_dir(self):
        assert _parse("run", "-c", "c.yaml", "--extra-orbit-dir", "d").extra_orbit_dir == "d"

    @pytest.mark.parametrize("flag", ["-np", "--norad-path"])
    def test_norad_path(self, flag):
        assert _parse("run", "-c", "c.yaml", flag, "ids.txt").norad_path == "ids.txt"

    def test_norad_path_defaults_to_none(self):
        assert _parse("run", "-c", "c.yaml").norad_path is None

    def test_config_is_required(self):
        with pytest.raises(SystemExit):
            _parse("run")

    def test_subcommand_is_required(self):
        # The bare `tabascal -c ...` form the docs used to show is not valid.
        with pytest.raises(SystemExit):
            _parse("-c", "config.yaml")


class TestLightCurveSubcommand:
    """``tabascal light-curve``: the matched-filter light-curve extractor.

    Two ways in, and they must not be mixed: a tabascal config, whose
    ``satellites`` section already names the satellites and whose MS is read
    once; or an MS plus an explicit list of NORAD IDs.
    """

    def test_a_config_names_the_satellites(self):
        args = _parse("light-curve", "-c", "tab_target.yaml")
        assert args.command == "light-curve"
        assert args.config == "tab_target.yaml"
        assert args.norad_ids is None and args.norad_path is None

    def test_a_config_can_be_pointed_at_a_measurement_set(self):
        args = _parse("light-curve", "-c", "c.yaml", "-ms", "obs.ms", "-s", "sim")
        assert args.ms_path == "obs.ms"
        assert args.sim_dir == "sim"

    @pytest.mark.parametrize("flag", ["-n", "--norad-ids"])
    def test_ids_can_be_given_directly(self, flag):
        args = _parse("light-curve", "-ms", "obs.ms", flag, "27868,57865")
        assert args.norad_ids == "27868,57865"

    @pytest.mark.parametrize("flag", ["-np", "--norad-path"])
    def test_ids_can_come_from_a_file(self, flag):
        assert _parse(
            "light-curve", "-ms", "obs.ms", flag, "ids.txt"
        ).norad_path == "ids.txt"

    def test_a_config_and_explicit_ids_are_mutually_exclusive(self):
        """Both would name the satellites, and there is no rule for which wins."""
        with pytest.raises(SystemExit):
            _parse("light-curve", "-c", "c.yaml", "-n", "27868")
        with pytest.raises(SystemExit):
            _parse("light-curve", "-c", "c.yaml", "-np", "ids.txt")

    def test_the_two_id_sources_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse("light-curve", "-ms", "o.ms", "-n", "1", "-np", "ids.txt")

    def test_the_column_and_correlation(self):
        args = _parse(
            "light-curve", "-ms", "o.ms", "-n", "1",
            "-dc", "TAB_RES_DATA", "-cr", "yy", "-f", "1.4e9",
        )
        assert args.data_col == "TAB_RES_DATA"
        assert args.corr == "yy"
        assert args.freq == 1.4e9

    def test_the_column_and_correlation_are_unset_when_not_given(self):
        """Not defaulted in the parser: a config names them, and must win.

        A parser default is indistinguishable from a value the user typed, so
        defaulting here would silently overwrite `data.data_col` on every
        `-c` run.
        """
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1")
        assert args.data_col is None
        assert args.corr is None

    def test_an_unknown_correlation_is_refused(self):
        with pytest.raises(SystemExit):
            _parse("light-curve", "-ms", "o.ms", "-n", "1", "-cr", "rr")

    def test_the_residual_mode_takes_a_results_zarr(self):
        args = _parse("light-curve", "-c", "c.yaml", "-z", "map_pred.zarr")
        assert args.zarr == "map_pred.zarr"

    def test_the_orbit_directory_flag_is_the_run_s(self):
        """--extra-orbit-dir, not the retired --extra-tle-dir."""
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1",
                      "--extra-orbit-dir", "orbits/")
        assert args.extra_orbit_dir == "orbits/"

    def test_the_elevation_cut(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1",
                      "--min-elevation", "15")
        assert args.min_elevation == 15.0
        assert args.elevation_cut is True

    def test_the_elevation_cut_can_be_turned_off(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1", "--no-elevation-cut")
        assert args.elevation_cut is False

    def test_a_cut_and_no_cut_together_are_refused(self):
        """They contradict each other, and silently letting one win is worse."""
        with pytest.raises(SystemExit):
            _parse("light-curve", "-ms", "o.ms", "-n", "1",
                   "--min-elevation", "15", "--no-elevation-cut")

    def test_the_defaults(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1")
        assert args.freq is None
        assert args.output is None and args.tag is None
        assert args.plot is False
        assert args.exclude_autos is True
        # Not given, so the config value (or 0) decides -- see resolve_min_elevation.
        assert args.min_elevation is None
        assert args.max_mem_gb == 1.0

    def test_autocorrelations_can_be_kept(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1", "--include-autos")
        assert args.exclude_autos is False


class TestLightCurveInputs:
    """Resolution of the inputs argparse cannot express on its own."""

    @staticmethod
    def _mod():
        # Deliberately importable without JAX: the parser module is built by the
        # top-level parser, which --help must not pay the run stack for.
        from tabascal.scripts import rfi_estimate

        return rfi_estimate

    def test_norad_ids_parse_from_either_separator(self):
        parse = self._mod()._parse_norad_ids
        assert parse("1,2 3\n4") == [1, 2, 3, 4]

    def test_norad_ids_come_off_a_file(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("27868\n57865\n")
        args = _parse("light-curve", "-ms", "o.ms", "-np", str(path))
        assert self._mod().resolve_norad_ids(args) == [27868, 57865]

    def test_an_ms_is_required_without_a_config(self):
        with pytest.raises(SystemExit, match="-ms"):
            self._mod().resolve_ms_path(_parse("light-curve", "-n", "1"), None)

    def test_satellites_are_required_without_a_config(self):
        with pytest.raises(SystemExit, match="NORAD"):
            self._mod().resolve_norad_ids(_parse("light-curve", "-ms", "o.ms"))

    def test_the_ms_comes_from_the_config_when_not_given(self):
        args = _parse("light-curve", "-c", "c.yaml")
        config = {"data": {"ms_path": "/data/obs.ms", "sim_dir": None}}
        assert self._mod().resolve_ms_path(args, config) == "/data/obs.ms"

    def test_the_ms_is_derived_from_the_simulation_directory(self):
        args = _parse("light-curve", "-c", "c.yaml", "-s", "/data/sim_run")
        config = {"data": {"ms_path": None, "sim_dir": None}}
        assert self._mod().resolve_ms_path(args, config).endswith(
            "sim_run/sim_run.ms"
        )

    def test_the_flag_beats_the_config_for_the_elevation_cut(self):
        args = _parse("light-curve", "-c", "c.yaml", "--min-elevation", "20")
        assert self._mod().resolve_min_elevation(args, {"rfi": {"min_elevation": 0}}) == 20.0

    def test_the_config_cut_is_used_when_the_flag_is_absent(self):
        args = _parse("light-curve", "-c", "c.yaml")
        assert self._mod().resolve_min_elevation(args, {"rfi": {"min_elevation": 5}}) == 5

    def test_a_null_config_cut_stays_off(self):
        args = _parse("light-curve", "-c", "c.yaml")
        assert self._mod().resolve_min_elevation(
            args, {"rfi": {"min_elevation": None}}
        ) is None

    def test_the_manual_default_is_the_horizon(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1")
        assert self._mod().resolve_min_elevation(args, None) == 0.0

    def test_no_elevation_cut_overrides_everything(self):
        args = _parse("light-curve", "-c", "c.yaml", "--no-elevation-cut")
        assert self._mod().resolve_min_elevation(args, {"rfi": {"min_elevation": 5}}) is None

    def test_the_output_defaults_beside_the_measurement_set(self):
        args = _parse("light-curve", "-ms", "/data/obs.ms", "-n", "1")
        path = self._mod().resolve_output(args, "/data/obs.ms", "DATA")
        assert path == "/data/light_curves/DATA.npz"

    def test_the_output_is_named_for_the_resolved_column(self):
        """Not the parser default: with -c the config names the column."""
        args = _parse("light-curve", "-c", "c.yaml")
        path = self._mod().resolve_output(args, "/data/obs.ms", "TAB_RES_DATA")
        assert path.endswith("light_curves/TAB_RES_DATA.npz")

    def test_the_tag_names_the_output(self):
        args = _parse("light-curve", "-ms", "/data/obs.ms", "-n", "1", "-sx", "runA")
        assert self._mod().resolve_output(args, "/data/obs.ms", "DATA").endswith(
            "light_curves/runA.npz"
        )

    def test_an_explicit_output_wins(self, tmp_path):
        out = str(tmp_path / "curves.npz")
        args = _parse("light-curve", "-ms", "/data/obs.ms", "-n", "1", "-o", out)
        assert self._mod().resolve_output(args, "/data/obs.ms", "DATA") == out

    def test_an_output_without_a_suffix_gets_one(self):
        """`-o curves` writes curves.npz, so it must also say curves.npz."""
        args = _parse("light-curve", "-ms", "/data/obs.ms", "-n", "1", "-o", "curves")
        assert self._mod().resolve_output(args, "/data/obs.ms", "DATA") == "curves.npz"

    def test_an_output_that_already_ends_in_npz_is_left_alone(self):
        args = _parse("light-curve", "-ms", "/o.ms", "-n", "1", "-o", "a/b.npz")
        assert self._mod().resolve_output(args, "/o.ms", "DATA") == "a/b.npz"

    # --- the column and the correlation: the config wins unless overridden ---

    def test_the_config_column_is_used_when_the_flag_is_absent(self):
        args = _parse("light-curve", "-c", "c.yaml")
        config = {"data": {"data_col": "TAB_RES_DATA", "corr": "yy"}}
        assert self._mod().resolve_data_col(args, config) == "TAB_RES_DATA"
        assert self._mod().resolve_corr(args, config) == "yy"

    def test_the_flag_beats_the_config_column(self):
        args = _parse("light-curve", "-c", "c.yaml", "-dc", "DATA", "-cr", "xx")
        config = {"data": {"data_col": "TAB_RES_DATA", "corr": "yy"}}
        assert self._mod().resolve_data_col(args, config) == "DATA"
        assert self._mod().resolve_corr(args, config) == "xx"

    def test_the_manual_defaults_apply_without_a_config(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1")
        assert self._mod().resolve_data_col(args, None) == "DATA"
        assert self._mod().resolve_corr(args, None) == "xx"

    def test_a_config_that_names_neither_falls_back_to_the_defaults(self):
        args = _parse("light-curve", "-c", "c.yaml")
        assert self._mod().resolve_data_col(args, {"data": {}}) == "DATA"
        assert self._mod().resolve_corr(args, {"data": {}}) == "xx"


class TestRunReporting:
    """``run()`` reports peak memory however the run ends; timings only on success."""

    @pytest.fixture
    def impl(self):
        """The run implementation module, imported lazily.

        Kept out of the module scope so the parser tests above go on paying
        nothing for the JAX/NumPyro import.
        """
        from tabascal.scripts import _run_tabascal_impl

        return _run_tabascal_impl

    @pytest.fixture
    def calls(self, impl, monkeypatch):
        """Record what ``run()`` reports, with everything heavy stubbed out."""
        calls = []
        monkeypatch.setattr(impl, "load_config", lambda path: {})
        # set_precision and enable_timings both mutate global state (x64, the
        # timing manager); neither is under test here.
        monkeypatch.setattr(impl, "set_precision", lambda config: False)
        monkeypatch.setattr(impl, "enable_timings", lambda: None)
        monkeypatch.setattr(impl, "print_memory_usage", lambda: calls.append("memory"))
        monkeypatch.setattr(impl, "print_timings", lambda: calls.append("timings"))
        return calls

    @staticmethod
    def _stub_run(impl, monkeypatch, calls, raises=None):
        def tabascal_subtraction(*args, **kwargs):
            calls.append("run")
            if raises is not None:
                raise raises

        monkeypatch.setattr(impl, "tabascal_subtraction", tabascal_subtraction)

    def test_reports_memory_when_the_run_raises(self, impl, monkeypatch, calls):
        """An OOM -- or any other mid-run death -- still gets its peak-memory number."""
        error = RuntimeError("RESOURCE_EXHAUSTED: out of memory")
        self._stub_run(impl, monkeypatch, calls, raises=error)

        with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
            impl.run(_parse("run", "-c", "c.yaml", "-t"))

        # Timings are deliberately left out: a timing table for a dead run misleads.
        assert calls == ["run", "memory"]

    def test_reports_memory_when_the_run_exits_on_an_orbit_error(
        self, impl, monkeypatch, calls
    ):
        """The TLE/truth error path exits 1 -- and reports on the way out."""
        from tabascal.orbit import TLEError

        self._stub_run(impl, monkeypatch, calls, raises=TLEError("no orbit for 99999"))

        with pytest.raises(SystemExit) as exc:
            impl.run(_parse("run", "-c", "c.yaml"))

        assert exc.value.code == 1
        assert calls == ["run", "memory"]

    @pytest.mark.parametrize(
        "argv, expected",
        [
            (("run", "-c", "c.yaml"), ["run", "memory"]),
            (("run", "-c", "c.yaml", "-t"), ["run", "memory", "timings"]),
        ],
    )
    def test_success_path_is_unchanged(self, impl, monkeypatch, calls, argv, expected):
        """A completed run reports memory, and timings when they were asked for."""
        self._stub_run(impl, monkeypatch, calls)

        impl.run(_parse(*argv))

        assert calls == expected

    def test_a_failed_report_does_not_mask_the_run_failure(
        self, impl, monkeypatch, calls, capsys
    ):
        """The report queries the backend that may have just died -- it must not raise.

        Without this the reporting exception would replace the OOM (or the
        TLE/truth ``SystemExit``) on the way out, losing the primary diagnostic.
        """
        self._stub_run(impl, monkeypatch, calls, raises=RuntimeError("RESOURCE_EXHAUSTED"))
        monkeypatch.setattr(
            impl,
            "print_memory_usage",
            _raise(RuntimeError("backend is gone")),
        )

        with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
            impl.run(_parse("run", "-c", "c.yaml"))

        assert "backend is gone" in capsys.readouterr().err

    def test_a_failed_report_does_not_mask_the_orbit_error_exit(
        self, impl, monkeypatch, calls, capsys
    ):
        """Nor may it turn the documented exit-1 into an unhandled exception."""
        from tabascal.orbit import TLEError

        self._stub_run(impl, monkeypatch, calls, raises=TLEError("no orbit for 99999"))
        monkeypatch.setattr(
            impl, "print_memory_usage", _raise(RuntimeError("backend is gone"))
        )

        with pytest.raises(SystemExit) as exc:
            impl.run(_parse("run", "-c", "c.yaml"))

        assert exc.value.code == 1
        assert "backend is gone" in capsys.readouterr().err

    def test_a_worker_rank_reports_nothing_on_success(self, impl, monkeypatch, calls):
        """Only rank 0 talks: the report tables are printed once, not once per process."""
        monkeypatch.setattr(impl, "is_process_0", lambda: False)
        self._stub_run(impl, monkeypatch, calls)

        impl.run(_parse("run", "-c", "c.yaml", "-t"))

        assert calls == ["run"]

    def test_a_worker_rank_reports_nothing_but_still_fails(self, impl, monkeypatch, calls):
        """The rank gate silences the report, never the failure."""
        monkeypatch.setattr(impl, "is_process_0", lambda: False)
        self._stub_run(impl, monkeypatch, calls, raises=RuntimeError("RESOURCE_EXHAUSTED"))

        with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
            impl.run(_parse("run", "-c", "c.yaml", "-t"))

        assert calls == ["run"]


class TestDocumentedCommandScraper:
    """The scraper reads commands, and only commands, out of the docs."""

    def _page(self, tmp_path, text):
        (tmp_path / "page.md").write_text(text)
        return documented_commands(tmp_path)

    def test_reads_a_fenced_command(self, tmp_path):
        found = self._page(tmp_path, "```bash\ntabascal run -c c.yaml\n```\n")
        assert found == [("page.md", ["tabascal", "run", "-c", "c.yaml"])]

    def test_ignores_prose_that_starts_with_the_word(self, tmp_path):
        """The case that bit: a sentence is not an invocation."""
        found = self._page(
            tmp_path, "tabascal fits a single correlation, named by `data.corr`.\n"
        )
        assert found == []

    def test_ignores_prose_between_fenced_blocks(self, tmp_path):
        found = self._page(
            tmp_path,
            "```bash\ntabascal run -c a.yaml\n```\n"
            "tabascal reads visibilities as (n_time, n_bl).\n"
            "```bash\ntabascal run -c b.yaml\n```\n",
        )
        assert [argv[-1] for _, argv in found] == ["a.yaml", "b.yaml"]

    def test_accepts_a_shell_prompt(self, tmp_path):
        found = self._page(tmp_path, "```console\n$ tabascal run -h\n```\n")
        assert found == [("page.md", ["tabascal", "run", "-h"])]

    def test_strips_a_trailing_comment(self, tmp_path):
        found = self._page(tmp_path, "```bash\ntabascal -h   # lists subcommands\n```\n")
        assert found == [("page.md", ["tabascal", "-h"])]

    def test_ignores_a_directory_tree_entry(self, tmp_path):
        found = self._page(tmp_path, "```\ntabascal/\n  write.py\n```\n")
        assert found == []


class TestDocumentedCommands:
    """Every ``tabascal ...`` invocation in the docs must parse."""

    def test_docs_contain_tabascal_commands(self):
        assert documented_commands(), "no tabascal commands found in docs/"

    def test_every_documented_command_parses(self):
        for page, argv in documented_commands():
            argv = argv[1:]  # drop the program name
            if "-h" in argv or "--help" in argv:
                continue  # argparse exits on help; the flag itself is always valid
            try:
                build_parser().parse_args(argv)
            except SystemExit as e:  # pragma: no cover - failure path
                pytest.fail(f"{page}: `tabascal {' '.join(argv)}` does not parse ({e})")
