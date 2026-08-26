"""The ``tabascal`` CLI argument surface.

Every command form the documentation shows a user must actually parse. These
tests only exercise the parser, so nothing is imported from the heavy JAX/NumPyro
run implementation and nothing is executed.
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
