"""Reading a path out of a config, for the entry points that take one.

``tabascal run`` and ``tabascal light-curve`` read the same ``data.sim_dir`` and
``data.ms_path`` keys with the same precedence, so they answer for them the same
way here rather than each in its own idiom. Kept free of imports beyond ``os``:
``rfi_estimate``, which implements the light-curve subcommand, holds its
JAX/NumPyro imports back so that parsing arguments costs nothing, and this is
reached from argument resolution.
"""

import os


def config_is_unset(value) -> bool:
    """Whether a config key was left blank: absent, ``null``, or empty.

    Falsiness is the wrong test for a path key. ``0`` and ``[]`` are values
    somebody wrote, and falling through to a default as though the key had never
    been set is the shape of the bug issue #207 reports -- the config names one
    thing and the run reads another without saying so.
    """

    return value is None or (isinstance(value, str) and value == "")


def config_path(value, key: str) -> str:
    """A path from the config, absolute, or a message naming the key it came from.

    ``os.path.abspath`` raises ``TypeError: expected str, bytes or os.PathLike
    object, not int`` on anything else, which names neither the key nor the
    config it was read from. Nothing validates the ``data`` section's types, and
    ``data.gain_table`` beside these two keys genuinely accepts a list, so a
    list here is a plausible slip rather than a perverse one.

    ``bytes`` is rejected with the rest: ``abspath`` would return ``bytes``,
    which then fails in the ``os.path.join`` downstream instead of here.
    """

    if not isinstance(value, (str, os.PathLike)):
        raise SystemExit(
            f"Config parameter ({key}: {value!r}) is not a path. Give it as a "
            "string naming a file or directory."
        )

    return os.path.abspath(value)
