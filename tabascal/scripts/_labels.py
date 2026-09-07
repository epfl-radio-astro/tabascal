"""Validation for the CLI options that name an output rather than locate one.

``-sx/--tag`` and ``-sx/--suffix`` are labels: they are interpolated into a
filename that the run then places itself. A label carrying a path takes that
placement away, silently and in whichever direction the path points --
``os.path.join`` discards everything before an absolute component, ``os.makedirs``
and zarr create whatever directories a relative one names, and ``..`` walks out
of the output directory altogether.

Both are reachable by accident rather than only by misuse: argparse matches a
short option by prefix, so a bare ``-s`` on either subcommand arrives here as
the label.

Windows names -- a colon's alternate data stream, a reserved device like ``NUL``
-- are not policed: tabascal ships for macOS and Linux only (its casacore
dependency has no Windows build), so a label cannot reach a filesystem that
reads them that way. The backslash is refused anyway, because it is never a
deliberate part of a label.

Imports nothing but ``os`` and ``ntpath``, both standard library: the parsers
are built before the JAX import, so that ``-h`` does not pay for it.
"""

import ntpath
import os

#: Every separator this platform understands, without the ``None`` posix uses
#: for ``altsep`` -- ``"" in text`` is true of every string and would reject
#: every label. Windows' separator is included on every platform: a backslash
#: is never a deliberate part of a label, and one that reaches a Windows run
#: would split a path there.
_SEPARATORS = tuple({c for c in (os.sep, os.altsep, ntpath.sep, ntpath.altsep) if c})


def label(value: str) -> str:
    """``value`` if it names a file, raising if it names a path.

    Used as an argparse ``type``, so the run stops at the command line rather
    than after the work whose output the label was going to name.
    """

    # argparse applies type() to a string default too, and "" is the default
    # suffix: no label at all, which is not a path.
    if not value:
        return value

    # splitdrive is the platform's own: `D:sim` carries no separator at all
    # and resolves against drive D's current directory, but only where drives
    # exist. On posix `a:b` is an ordinary filename and stays one.
    if (
        any(c in value for c in _SEPARATORS)
        or os.path.splitdrive(value)[0]
        or value in (os.curdir, os.pardir)
    ):
        import argparse

        raise argparse.ArgumentTypeError(
            f"{value!r} is a path, and this is a label: it names the output "
            "file, which the run places itself. Give a path with -o/--output "
            "(light-curve) or -od/--out_dir (run)."
        )

    return value
