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
