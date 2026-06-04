"""Tests for tabascal.config YAML loading.

The config files use bare scientific notation (``1e6``, ``209e3``, ``3e3``) which
PyYAML's stock SafeLoader parses as *strings*. ``tabascal.config.yaml_load`` uses
a private loader subclass that parses them as floats, without polluting the global
``yaml.SafeLoader`` for the rest of the process.
"""

import yaml

from tabascal.config import yaml_load


def test_yaml_load_parses_bare_scientific_notation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 3e3\nb: 1e6\nc: 209e3\nd: 1e-2\ne: 1.227e9\nf: 1e0\n")

    cfg = yaml_load(p)

    assert cfg == {
        "a": 3000.0,
        "b": 1e6,
        "c": 209000.0,
        "d": 0.01,
        "e": 1.227e9,
        "f": 1.0,
    }
    assert all(isinstance(v, float) for v in cfg.values())


def test_yaml_load_handles_nested_and_list_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("rfi:\n  k0s: [1e0, 1e-2]\n  p0: 3e3\n")

    cfg = yaml_load(p)

    assert cfg["rfi"]["k0s"] == [1.0, 0.01]
    assert cfg["rfi"]["p0"] == 3000.0


def test_yaml_load_does_not_pollute_global_safeloader():
    """Importing/using tabascal.config must not reprogram the global SafeLoader.

    The float resolver is scoped to a private subclass, so a bare
    ``yaml.safe_load`` still parses scientific notation as a string. This locks in
    the fix and guards against a regression back to the global mutation.
    """
    import tabascal.config  # noqa: F401 - ensure the module's import side effects ran

    assert isinstance(yaml.safe_load("3e3"), str)
