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

    def test_extra_tle_dir(self):
        assert _parse("run", "-c", "c.yaml", "--extra-tle-dir", "d").extra_tle_dir == "d"

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


class TestDocumentedCommands:
    """Every ``tabascal ...`` invocation in the docs must parse."""

    def _documented(self):
        commands = []
        for page in sorted(_DOCS.glob("*.md")):
            for line in page.read_text().splitlines():
                line = line.strip()
                # `tabascal ...` as a command, not `tabascal/` in a directory tree.
                if not re.match(r"^tabascal(\s|$)", line):
                    continue
                # Strip trailing comments used to annotate help invocations.
                line = re.split(r"\s+#", line, maxsplit=1)[0]
                commands.append((page.name, line.split()))
        return commands

    def test_docs_contain_tabascal_commands(self):
        assert self._documented(), "no tabascal commands found in docs/"

    def test_every_documented_command_parses(self):
        for page, argv in self._documented():
            argv = argv[1:]  # drop the program name
            if "-h" in argv or "--help" in argv:
                continue  # argparse exits on help; the flag itself is always valid
            try:
                build_parser().parse_args(argv)
            except SystemExit as e:  # pragma: no cover - failure path
                pytest.fail(f"{page}: `tabascal {' '.join(argv)}` does not parse ({e})")
