"""The ``tabascal`` CLI: its argument surface, where a run writes, and what
``run()`` always reports.

Every command form the documentation shows a user must actually parse. The
parser tests exercise the parser alone, so nothing is imported from the heavy
JAX/NumPyro run implementation and nothing is executed; the ``run()`` and path
tests import that module lazily and stub the run itself out.
"""

import os
import re
import shlex
from datetime import datetime
from pathlib import Path

import pytest

from tabascal.scripts.run_tabascal import build_parser

_REPO = Path(__file__).parent.parent
_DOCS = _REPO / "docs"

# A shell prompt some docs put in front of a command.
_PROMPT = re.compile(r"^\$\s+")


@pytest.fixture
def impl():
    """The run implementation module.

    Imported inside the fixture so the parser tests go on paying nothing for the
    JAX/NumPyro import.
    """
    from tabascal.scripts import _run_tabascal_impl

    return _run_tabascal_impl


#: The README shows the same commands and is the first thing a new user copies,
#: but it lived outside the docs glob -- so its "run it on a Measurement Set"
#: example went on naming a form of the command that no longer worked.
_README = _REPO / "README.md"


def _documented_pages(docs_dir, readme):
    """The pages to scan: every docs page, and the README beside them.

    ``readme`` is passed rather than derived from ``docs_dir``, which the
    scraper's own tests point at a ``tmp_path``: deriving it would make those
    tests scrape whatever README happened to sit in pytest's basetemp.
    """

    pages = sorted(docs_dir.glob("*.md"))
    if readme is not None and readme.exists():
        pages.append(readme)

    return pages


def documented_commands(docs_dir=_DOCS, readme=_README):
    """Every ``tabascal ...`` command in the docs' fenced code blocks.

    Only fenced code is scanned. A prose sentence that happens to start a line
    with the word "tabascal" is not a command, and treating it as one turns a
    docs edit into a confusing parser failure.
    """

    commands = []
    for page in _documented_pages(docs_dir, readme):
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
            # shlex, not split(): a shell hands `-s ""` an empty argument and
            # `-s "output dir"` a single one, and a check that reads the parsed
            # options has to see what the shell would have passed.
            commands.append((page.name, shlex.split(line)))
    return commands


def _parse(*argv):
    return build_parser().parse_args(list(argv))


def _raise(error):
    """A no-argument stub that raises ``error``."""

    def stub():
        raise error

    return stub


class TestRunSubcommand:

    def test_config_and_out_dir(self):
        args = _parse("run", "-c", "tab_target.yaml", "-od", "out_dir")
        assert args.command == "run"
        assert args.config == "tab_target.yaml"
        assert args.out_dir == "out_dir"

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
        args = _parse("light-curve", "-c", "c.yaml", "-ms", "obs.ms", "-od", "sim")
        assert args.ms_path == "obs.ms"
        assert args.out_dir == "sim"

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

    @pytest.mark.parametrize("corr", ["xx", "yy", "rr", "ll", "rl", "i", "v"])
    def test_every_correlation_the_reader_supports_is_accepted(self, corr):
        """Circular and Stokes MSs exist, and read_ms resolves them all."""
        assert _parse(
            "light-curve", "-ms", "o.ms", "-n", "1", "-cr", corr
        ).corr == corr

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


class TestRfiPerSatSubcommand:
    """``tabascal rfi-per-sat``: the per-satellite RFI columns, post hoc.

    Both inputs are required -- there is nothing to derive an MS from here, and
    a results zarr the run did not write per-satellite visibilities into has
    nothing to export.
    """

    def test_the_ms_and_the_results_zarr(self):
        args = _parse("rfi-per-sat", "-m", "obs.ms", "-z", "map_pred.zarr")
        assert args.command == "rfi-per-sat"
        assert args.ms_path == "obs.ms"
        assert args.results_zarr_path == "map_pred.zarr"

    def test_both_are_required(self):
        with pytest.raises(SystemExit):
            _parse("rfi-per-sat", "-m", "obs.ms")
        with pytest.raises(SystemExit):
            _parse("rfi-per-sat", "-z", "map_pred.zarr")

    def test_the_defaults(self):
        args = _parse("rfi-per-sat", "-m", "o.ms", "-z", "r.zarr")
        # The columns the docs name, and the correlation off the zarr.
        assert args.prefix == "TAB_RFI_"
        assert args.corr is None

    def test_the_prefix_and_correlation_can_be_given(self):
        args = _parse(
            "rfi-per-sat", "-m", "o.ms", "-z", "r.zarr", "-p", "RFI_SRC_", "-c", "yy"
        )
        assert args.prefix == "RFI_SRC_"
        assert args.corr == "yy"

    def test_the_writer_is_imported_only_when_it_runs(self):
        """The top-level parser builds this one, and `tabascal -h` stays cheap.

        ``tabascal.write`` pulls in JAX, so it may not be imported at module
        level here: the parser is built on every ``tabascal`` invocation.
        """
        from tabascal.scripts import rfi_per_sat_to_MS

        assert not hasattr(rfi_per_sat_to_MS, "write_per_sat_rfi_ms")
        assert rfi_per_sat_to_MS.build_parser() is not None

    def test_building_the_whole_parser_imports_no_jax(self):
        """In a clean process: the test session has JAX imported long before this.

        Checking ``sys.modules`` in-process proves nothing -- conftest imports
        JAX at collection. A subprocess that does nothing but build the parser is
        the only place the claim can be made, and it covers every subcommand's
        parser, not only this one's.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from tabascal.scripts.run_tabascal import build_parser\n"
                "build_parser().parse_args(['rfi-per-sat', '-m', 'o.ms', '-z', 'r.zarr'])\n"
                "print(sorted(m for m in ('jax', 'numpyro', 'tabascal.write')"
                " if m in sys.modules))\n",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout


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
        config = {"data": {"ms_path": "/data/obs.ms", "out_dir": None}}
        assert self._mod().resolve_ms_path(args, config) == "/data/obs.ms"

    def test_the_ms_is_derived_from_the_simulation_directory(self):
        args = _parse("light-curve", "-c", "c.yaml", "-od", "/data/sim_run")
        config = {"data": {"ms_path": None, "out_dir": None}}
        assert self._mod().resolve_ms_path(args, config).endswith(
            "sim_run/sim_run.ms"
        )

    @pytest.mark.parametrize("value", [1227000000, ["a.ms", "b.ms"], 0, []])
    def test_a_config_ms_path_that_is_not_a_path_names_itself(self, value):
        """The same config, refused the same way as by ``tabascal run``.

        ``resolve_ms_path`` and ``_resolve_paths`` read the same two keys with
        the same precedence, so they read them through the same helpers. This
        one used to hand the value straight to ``os.path.abspath`` (a bare
        TypeError out of posixpath) or, for the falsy ones, ignore it and go on
        to the sim-dir fallback.
        """
        args = _parse("light-curve", "-c", "c.yaml")
        config = {"data": {"ms_path": value, "out_dir": None}}

        with pytest.raises(SystemExit, match=r"data\.ms_path.*is not a path"):
            self._mod().resolve_ms_path(args, config)

    def test_an_unset_config_ms_path_still_falls_through(self):
        """``null`` and ``""`` remain the way to say the key is not in use."""
        args = _parse("light-curve", "-c", "c.yaml", "-od", "/data/sim_run")
        config = {"data": {"ms_path": "", "out_dir": None}}

        assert self._mod().resolve_ms_path(args, config).endswith("sim_run/sim_run.ms")

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

    def test_an_unknown_correlation_is_refused(self):
        """Not by the parser -- which would have to hard-code the list -- but
        against tabascal.ms's own table, named in the error."""
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1", "-cr", "zz")

        with pytest.raises(SystemExit, match="zz"):
            self._mod().resolve_corr(args, None)

    def test_a_correlation_the_reader_supports_passes_the_check(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1", "-cr", "rl")

        assert self._mod().resolve_corr(args, None) == "rl"

    def test_a_config_correlation_is_checked_too(self):
        args = _parse("light-curve", "-c", "c.yaml")

        with pytest.raises(SystemExit, match="nonsense"):
            self._mod().resolve_corr(args, {"data": {"corr": "nonsense"}})

    def test_the_manual_defaults_apply_without_a_config(self):
        args = _parse("light-curve", "-ms", "o.ms", "-n", "1")
        assert self._mod().resolve_data_col(args, None) == "DATA"
        assert self._mod().resolve_corr(args, None) == "xx"

    def test_a_config_that_names_neither_falls_back_to_the_defaults(self):
        args = _parse("light-curve", "-c", "c.yaml")
        assert self._mod().resolve_data_col(args, {"data": {}}) == "DATA"
        assert self._mod().resolve_corr(args, {"data": {}}) == "xx"

    def test_a_flag_stands_alone_when_the_config_names_nothing(self):
        """The remaining corner of the matrix: flag given, config key absent."""
        args = _parse("light-curve", "-c", "c.yaml", "-dc", "AST_DATA", "-cr", "yx")
        assert self._mod().resolve_data_col(args, {"data": {}}) == "AST_DATA"
        assert self._mod().resolve_corr(args, {"data": {}}) == "yx"

    def test_a_null_config_value_falls_back_to_the_default(self):
        """`data_col: null` names nothing, so it is not an answer to be kept."""
        args = _parse("light-curve", "-c", "c.yaml")
        config = {"data": {"data_col": None, "corr": None}}
        assert self._mod().resolve_data_col(args, config) == "DATA"
        assert self._mod().resolve_corr(args, config) == "xx"

    def test_the_column_flag_does_not_carry_the_correlation_with_it(self):
        """One flag overrides one value: -dc must not pull -cr off the config."""
        args = _parse("light-curve", "-c", "c.yaml", "-dc", "AST_DATA")
        config = {"data": {"data_col": "TAB_RES_DATA", "corr": "yy"}}
        assert self._mod().resolve_data_col(args, config) == "AST_DATA"
        assert self._mod().resolve_corr(args, config) == "yy"

    def test_the_correlation_flag_does_not_carry_the_column_with_it(self):
        args = _parse("light-curve", "-c", "c.yaml", "-cr", "xy")
        config = {"data": {"data_col": "TAB_RES_DATA", "corr": "yy"}}
        assert self._mod().resolve_data_col(args, config) == "TAB_RES_DATA"
        assert self._mod().resolve_corr(args, config) == "xy"


class _FrozenClock:
    """``datetime`` with a stopped clock, for the same-second collision test."""

    @staticmethod
    def now():
        return datetime(2026, 1, 2, 3, 4, 5)


class TestRunLogPath:
    """Where a run's log is written: in the plot directory it fills anyway.

    It used to be opened in the process's working directory and copied into the
    plot directory once the run had finished. Two runs started in the same
    second from one directory then shared that filename, and the first to finish
    removed it -- killing the second inside the copy with all of its compute
    already done.
    """

    @staticmethod
    def _resolve(impl, tmp_path, suffix=""):
        config = {"data": {"out_dir": str(tmp_path / "obs")}, "model": {}}
        return impl._resolve_paths(config, None, None, suffix, None)

    @staticmethod
    def _assert_log_is_in_the_plot_dir(paths):
        # Normalised: a run with no suffix leaves plot_dir with a trailing
        # separator, which dirname() of the log path does not have.
        assert os.path.normpath(os.path.dirname(paths.log_path)) == os.path.normpath(
            paths.plot_dir
        )

    def test_the_log_is_written_in_the_plot_directory(self, impl, tmp_path):
        paths = self._resolve(impl, tmp_path)

        assert os.path.isabs(paths.log_path)  # never relative to the CWD
        self._assert_log_is_in_the_plot_dir(paths)
        assert os.path.basename(paths.log_path) == f"log_tab_{paths.run_id}.txt"

    def test_the_suffix_names_the_log(self, impl, tmp_path):
        """Which run a log belongs to, without opening it."""
        paths = self._resolve(impl, tmp_path, suffix="runA")

        self._assert_log_is_in_the_plot_dir(paths)
        assert paths.plot_dir.endswith(os.path.join("plots", "runA"))
        assert os.path.basename(paths.log_path) == f"log_tab_runA_{paths.run_id}.txt"

    def test_the_directory_is_there_before_the_log_is_opened(
        self, impl, tmp_path, monkeypatch
    ):
        """``_stdout_logger`` opens the path for writing; nothing creates it later.

        And the directory the run was launched from is left alone.
        """
        monkeypatch.chdir(tmp_path)
        paths = self._resolve(impl, tmp_path)

        with impl._stdout_logger(paths.log_path, True):
            print("logged")

        assert Path(paths.log_path).read_text().strip() == "logged"
        assert list(Path.cwd().glob("log_tab*")) == []

    def test_runs_in_the_same_second_are_told_apart_by_their_suffix(
        self, impl, tmp_path, monkeypatch
    ):
        """The suffix separates concurrent runs; the timestamp alone cannot.

        ``run_id`` has 1-second resolution, so two runs launched together with
        the *same* suffix still resolve to one path -- asserted here so what is
        left of the collision is on the record rather than assumed gone.
        """
        monkeypatch.setattr(impl, "datetime", _FrozenClock)

        run_a = self._resolve(impl, tmp_path, "runA")
        run_b = self._resolve(impl, tmp_path, "runB")
        run_a_again = self._resolve(impl, tmp_path, "runA")

        assert run_a.run_id == run_b.run_id == run_a_again.run_id
        assert run_a.log_path != run_b.log_path
        assert run_a.log_path == run_a_again.log_path


class TestResolveMSPath:
    """Which Measurement Set a run reads: the flag, the config, the sim layout.

    Issue #207. ``data.ms_path`` was read from the config and then written over
    by ``<out_dir>/<basename(out_dir)>.ms``, the layout ``sim-vis`` leaves
    behind. On a simulation the two agree and nothing shows; on real data,
    where the MS's name has no relation to its parent directory's, a config
    that named its MS failed reporting a table the user had never mentioned.
    """

    @staticmethod
    def _resolve(impl, tmp_path, ms_path=None, config_ms_path=None):
        config = {
            "data": {"out_dir": str(tmp_path / "obs"), "ms_path": config_ms_path},
            "model": {},
        }
        paths = impl._resolve_paths(config, None, ms_path, "", None)
        # The run reads the bundle; the config is written back so the archived
        # tab_config_<run_id>.yaml records the MS the run actually used. They
        # have to be the one path or that record is a fiction.
        assert config["data"]["ms_path"] == paths.ms_path
        return paths.ms_path

    def test_the_config_names_the_ms(self, impl, tmp_path):
        """A config with no ``-ms`` flag beside it, which used to be ignored."""
        ms = str(tmp_path / "elsewhere" / "mwa_large_ch28.ms")

        assert self._resolve(impl, tmp_path, config_ms_path=ms) == ms

    def test_the_flag_beats_the_config(self, impl, tmp_path):
        """``-ms`` is how a run is pointed at one MS of many without editing."""
        flag = str(tmp_path / "from_flag.ms")
        config = str(tmp_path / "from_config.ms")

        assert self._resolve(impl, tmp_path, flag, config) == flag

    def test_the_sim_layout_is_the_fallback(self, impl, tmp_path):
        """Neither given: the simulation layout, exactly as before."""
        resolved = self._resolve(impl, tmp_path)

        assert resolved == str(tmp_path / "obs" / "obs.ms")

    @pytest.mark.parametrize("empty", [None, ""])
    def test_an_unset_config_key_falls_through(self, impl, tmp_path, empty):
        """The base config ships ``ms_path`` as a bare key, so it arrives None."""
        resolved = self._resolve(impl, tmp_path, config_ms_path=empty)

        assert resolved == str(tmp_path / "obs" / "obs.ms")

    def test_a_missing_config_key_falls_through(self, impl, tmp_path):
        """A hand-written config need not carry every key of the base one."""
        config = {"data": {"out_dir": str(tmp_path / "obs")}, "model": {}}

        paths = impl._resolve_paths(config, None, None, "", None)

        assert paths.ms_path == str(tmp_path / "obs" / "obs.ms")
        # Recorded even when derived, so the archived tab_config_<run_id>.yaml
        # is a record of the run and not of the intent -- which is why
        # re-running one after moving the data needs -ms and not -s.
        assert config["data"]["ms_path"] == paths.ms_path

    def test_a_relative_config_path_is_made_absolute(self, impl, tmp_path, monkeypatch):
        """As the flag already was: a run must not depend on its working dir."""
        monkeypatch.chdir(tmp_path)

        resolved = self._resolve(impl, tmp_path, config_ms_path="obs.ms")

        assert os.path.isabs(resolved)
        assert resolved == str(tmp_path / "obs.ms")

    def test_the_out_dir_flag_does_not_move_the_ms(self, impl, tmp_path):
        """The branch's headline consequence, and the one worth a test.

        ``-s`` used to be the single "which dataset" knob, because the MS was
        re-derived from it. It now moves the outputs and the truth zarr while
        the MS stays where the config named it -- which is the whole point on
        real data, where the visibilities do not live in the output directory.
        ``-ms`` is how a run is moved onto other visibilities.
        """
        named_ms = str(tmp_path / "elsewhere" / "real.ms")
        config = {
            "data": {"out_dir": str(tmp_path / "obs"), "ms_path": named_ms},
            "model": {},
        }

        paths = impl._resolve_paths(config, str(tmp_path / "other"), None, "", None)

        assert paths.ms_path == named_ms
        assert config["data"]["out_dir"] == str(tmp_path / "other")
        assert paths.plot_dir.startswith(str(tmp_path / "other"))

    def test_the_out_dir_flag_beats_the_config(self, impl, tmp_path):
        """It still wins for everything out_dir does name."""
        config = {"data": {"out_dir": str(tmp_path / "from_config")}, "model": {}}

        paths = impl._resolve_paths(config, str(tmp_path / "from_flag"), None, "", None)

        assert paths.ms_path == str(tmp_path / "from_flag" / "from_flag.ms")
        assert config["data"]["zarr_path"] == str(
            tmp_path / "from_flag" / "from_flag.zarr"
        )

    def test_the_truth_zarr_defaults_beside_the_outputs(self, impl, tmp_path):
        """Named for the MS, which on a simulation is the directory's own name.

        ``ast.init: truth`` and ``plots.truth`` are the only readers, and a
        simulation is the only thing that has one -- ``sim-vis`` leaves it at
        ``<dir>/<dir name>.zarr`` beside the MS of the same stem, so deriving
        it from the MS's name reproduces that exactly.
        """
        config = {"data": {"out_dir": str(tmp_path / "obs")}, "model": {}}

        impl._resolve_paths(config, None, None, "", None)

        assert config["data"]["truth_zarr"] == str(tmp_path / "obs" / "obs.zarr")
        assert config["data"]["zarr_path"] == config["data"]["truth_zarr"]

    def test_an_explicit_truth_zarr_wins(self, impl, tmp_path):
        """The point of naming it: a simulation whose products go elsewhere.

        Deriving it from the output directory could only ever find the truth
        when the run wrote into the simulation, which is the assumption #207
        is about.
        """
        truth = str(tmp_path / "sim" / "pnt_src_obs.zarr")
        config = {
            "data": {
                "out_dir": str(tmp_path / "scratch"),
                "ms_path": str(tmp_path / "sim" / "pnt_src_obs.ms"),
                "truth_zarr": truth,
            },
            "model": {},
        }

        impl._resolve_paths(config, None, None, "", None)

        assert config["data"]["zarr_path"] == truth

    def test_a_real_data_run_derives_a_truth_zarr_that_simply_is_not_there(
        self, impl, tmp_path
    ):
        """And that is fine: truth loading guards on existence and says so.

        Worth pinning that it is derived rather than left unset, because
        ``truth.py`` reads the key and a missing one would be a KeyError where
        an absent file is a printed line.
        """
        config = {
            "data": {"ms_path": str(tmp_path / "mwa" / "mwa_large_ch28.ms")},
            "model": {},
        }

        impl._resolve_paths(config, None, None, "", None)

        assert config["data"]["zarr_path"] == str(
            tmp_path / "mwa" / "mwa_large_ch28.zarr"
        )
        assert not os.path.exists(config["data"]["zarr_path"])


class TestResolvePathsRefusesWhatItCannotUse:
    """A message naming the key, rather than a traceback out of ``posixpath``.

    Nothing type-checks the ``data`` section, and these two keys are the ones
    that reach path arithmetic. ``os.path.abspath`` reports only the type it was
    handed -- "expected str, bytes or os.PathLike object, not NoneType" names
    neither the key nor the config it came from.
    """

    @staticmethod
    def _resolve(impl, data):
        return impl._resolve_paths({"data": data, "model": {}}, None, None, "", None)

    @pytest.mark.parametrize("data", [{}, {"out_dir": None}, {"out_dir": ""}])
    def test_naming_neither_the_ms_nor_the_directory_is_reported(self, impl, data):
        """Each implies the other, so only naming *neither* is unanswerable.

        ``sim_dir`` used to reach ``abspath(None)`` here and report the type it
        was handed. Now a run with an MS writes beside it and a run with a
        directory reads the simulation layout out of it -- there is nothing
        left to guess from only when both are absent.
        """
        with pytest.raises(SystemExit, match=r"Nothing to run on and nowhere"):
            self._resolve(impl, data)

    def test_the_message_names_both_ways_out(self, impl):
        with pytest.raises(SystemExit) as excinfo:
            self._resolve(impl, {})

        message = str(excinfo.value)
        assert "-ms/--ms_path" in message and "-od/--out_dir" in message
        assert "plots" in message and "results" in message

    def test_an_ms_alone_writes_beside_itself(self, impl, tmp_path):
        """The case #207 was reported for: real data, no simulation directory.

        ``sim_dir`` was required, and named the MS as well, so an observation
        that had never been near ``sim-vis`` had to be given a directory whose
        name the run would then look for the visibilities under.
        """
        ms = tmp_path / "mwa" / "mwa_large_ch28.ms"

        paths = impl._resolve_paths(
            {"data": {"ms_path": str(ms)}, "model": {}}, None, None, "", None
        )

        assert paths.ms_path == str(ms)
        assert paths.plot_dir.startswith(str(tmp_path / "mwa"))
        assert paths.f_name == "mwa_large_ch28"  # named for the data, not the dir

    def test_a_directory_alone_is_read_as_a_simulation(self, impl, tmp_path):
        """What ``sim_dir`` meant, unchanged, for the runs that were using it."""
        paths = impl._resolve_paths(
            {"data": {"out_dir": str(tmp_path / "obs")}, "model": {}},
            None,
            None,
            "",
            None,
        )

        assert paths.ms_path == str(tmp_path / "obs" / "obs.ms")
        assert paths.f_name == "obs"

    @pytest.mark.parametrize("value", [1227000000, ["a.ms", "b.ms"], 3.5])
    def test_a_config_ms_path_that_is_not_a_path_names_itself(
        self, impl, tmp_path, value
    ):
        """An obs ID pasted in, or the list form ``data.gain_table`` accepts."""
        data = {"out_dir": str(tmp_path / "obs"), "ms_path": value}

        with pytest.raises(SystemExit, match=r"data\.ms_path.*is not a path"):
            self._resolve(impl, data)

    @pytest.mark.parametrize("value", [1227000000, ["a", "b"]])
    def test_a_config_out_dir_that_is_not_a_path_names_itself(self, impl, value):
        with pytest.raises(SystemExit, match=r"data\.out_dir.*is not a path"):
            self._resolve(impl, {"out_dir": value})

    @pytest.mark.parametrize("value", [0, [], False])
    def test_a_falsy_non_path_is_reported_rather_than_ignored(
        self, impl, tmp_path, value
    ):
        """Only ``null`` and ``""`` mean unset; the rest are values someone wrote.

        Gating the type check on truthiness would let these fall through to
        ``<out_dir>/<basename>.ms`` in silence -- the config naming one MS and
        the run reading another, which is the bug this branch exists to fix.
        """
        data = {"out_dir": str(tmp_path / "obs"), "ms_path": value}

        with pytest.raises(SystemExit, match=r"data\.ms_path.*is not a path"):
            self._resolve(impl, data)

    def test_the_type_message_quotes_the_value(self, impl, tmp_path):
        """So a reader can see the obs ID or list they wrote, not just the key."""
        data = {"out_dir": str(tmp_path / "obs"), "ms_path": 1227000000}

        with pytest.raises(SystemExit) as excinfo:
            self._resolve(impl, data)

        assert "1227000000" in str(excinfo.value)

    def test_a_path_like_object_is_accepted(self, impl, tmp_path):
        """``Path`` is what a caller building a config in Python would pass."""
        paths = self._resolve(
            impl, {"out_dir": tmp_path / "obs", "ms_path": tmp_path / "real.ms"}
        )

        assert paths.ms_path == str(tmp_path / "real.ms")


class TestRunHeaderNamesTheMS:
    """The log has to say which visibilities the run read.

    It did not have to before: the MS was ``<out_dir>/<basename>.ms``, so the
    directory in the header named the data. Now that ``data.ms_path`` can name
    one anywhere, a config carrying a stale ``ms_path`` and swept over several
    ``-s`` directories would write to each in turn while reading the same
    visibilities throughout, with nothing in the log to show it.
    """

    @staticmethod
    def _header(impl, ms_path):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            impl._print_run_header("Custom", "obs", ms_path, "2026-01-02T03:04:05")

        return buffer.getvalue()

    def test_the_ms_is_in_the_header(self, impl):
        assert "/data/mwa/real.ms" in self._header(impl, "/data/mwa/real.ms")

    def test_the_output_directory_is_still_named_too(self, impl):
        """Both, because on real data they are no longer the same place."""
        header = self._header(impl, "/data/mwa/real.ms")

        assert "obs" in header
        assert "Custom" in header


class TestRunReporting:
    """``run()`` reports peak memory however the run ends; timings only on success."""

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
        """One page, and no README: these assert on the whole scrape result."""
        (tmp_path / "page.md").write_text(text)
        return documented_commands(tmp_path, readme=None)

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
        """The docs themselves, not the README, which is scraped alongside."""
        assert documented_commands(readme=None), "no tabascal commands found in docs/"

    def test_the_readme_is_scraped_too(self, tmp_path):
        """Its run command was broken for a release without any test noticing."""
        from_readme = documented_commands(docs_dir=tmp_path)

        assert [page for page, _ in from_readme] == ["README.md"] * len(from_readme)
        assert any(argv[:2] == ["tabascal", "run"] for _, argv in from_readme)

    @staticmethod
    def _run_options(argv):
        """What ``tabascal run`` would actually receive, per argparse.

        Matching argv tokens meant re-implementing option parsing and getting
        it wrong twice: ``--ms_path=data.ms`` was not recognised as naming an
        MS, ``--out_dir=`` was accepted as naming a directory when it is the
        empty string _resolve_paths treats as unset, and argparse's own
        abbreviations (``--sim_d=out``) matched nothing. The parser knows.
        """

        return build_parser().parse_args(argv[1:])

    def test_every_documented_run_names_an_ms_or_a_directory(self):
        """Parsing is not enough: neither flag is required, but one of them is.

        Each implies the other -- an MS writes beside itself, a directory is
        read as a simulation -- so argparse cannot demand either, and a command
        naming neither parses and then aborts with "Nothing to run on and
        nowhere to write". That is how the README and two docs pages came to
        show `tabascal run -c ... -ms ...` against a `sim_dir` that was
        required: correct again now, but only a check can keep it so.
        """
        offenders = []
        for page, argv in documented_commands():
            if argv[:2] != ["tabascal", "run"] or "-h" in argv:
                continue
            options = self._run_options(argv)
            if not (options.ms_path or options.out_dir):
                offenders.append(f"{page}: {' '.join(argv)}")

        assert not offenders, (
            "documented `tabascal run` commands naming neither a Measurement "
            f"Set nor an output directory: {offenders}. If one of these "
            "deliberately leans on a config for data.ms_path or data.out_dir, "
            "say so on the page and exempt it here."
        )

    @pytest.mark.parametrize(
        "tail, flagged",
        [
            ("-ms x.ms", False),  # writes beside the MS
            ("-od out", False),  # read as a simulation directory
            ("-ms x.ms -od out", False),
            ("--ms_path=x.ms --out_dir=out", False),
            ("--ms_p=x.ms --out_d=out", False),  # argparse abbreviations
            ('-ms x.ms -od "output dir"', False),  # one argument, not two
            ("-c only.yaml", True),  # neither: the one unresolvable case
            ('--out_dir=""', True),  # the shell's empty is unset, not a path
        ],
    )
    def test_the_check_reads_the_options_not_the_tokens(self, tail, flagged):
        """The spellings a token matcher got wrong, pinned rather than tried."""
        argv = shlex.split(f"tabascal run -c c.yaml {tail}")
        options = self._run_options(argv)

        assert bool(not (options.ms_path or options.out_dir)) is flagged

    def test_every_documented_command_parses(self):
        for page, argv in documented_commands():
            argv = argv[1:]  # drop the program name
            if "-h" in argv or "--help" in argv:
                continue  # argparse exits on help; the flag itself is always valid
            try:
                build_parser().parse_args(argv)
            except SystemExit as e:  # pragma: no cover - failure path
                pytest.fail(f"{page}: `tabascal {' '.join(argv)}` does not parse ({e})")
