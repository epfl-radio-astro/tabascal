# ---------------------------------------------------------------------------
# 'run' subcommand
# ---------------------------------------------------------------------------

def _run_cmd(args):
    # Multi-process bring-up must precede everything jax-related: the distributed
    # runtime has to exist before the device backend initializes, and the impl module
    # import pulls in the whole jax/numpyro stack. Memory-on-demand likewise has to be
    # set before the backend grabs the GPU (the impl module also sets it, but by then
    # only for the single-process path -- here it must land before init_distributed).
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from tabascal.distributed import init_distributed
    init_distributed()

    # Imported lazily so the lightweight subcommands (and --help) don't pay the
    # JAX import cost. The heavy implementation lives in a separate module.
    from tabascal.scripts._run_tabascal_impl import run
    run(args)


# ---------------------------------------------------------------------------
# 'light-curve' subcommand
# ---------------------------------------------------------------------------

def _light_curve_cmd(args):
    # Imported lazily so --help and the parser tests don't pay the JAX cost. The
    # parser itself comes from the same module and imports nothing heavy.
    from tabascal.scripts.rfi_estimate import run
    run(args)


# ---------------------------------------------------------------------------
# 'rfi-per-sat' subcommand
# ---------------------------------------------------------------------------

def _rfi_per_sat_cmd(args):
    # Lazy for the same reason: the export needs tabascal.write, which is JAX.
    from tabascal.scripts.rfi_per_sat_to_MS import run
    run(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    """Build the CLI parser.

    Separate from :func:`main` so the argument surface can be tested — every
    command shown in the documentation must parse — without running anything.
    """
    import argparse

    parser = argparse.ArgumentParser(description="tabascal CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_parser = subparsers.add_parser("run", help="Apply tabascal to a simulation.")
    run_parser.add_argument("-c", "--config", required=True, help="Path to the config file.")
    run_parser.add_argument("-s", "--sim_dir", help="Path to the directory of the simulation.")
    run_parser.add_argument("-ms", "--ms_path", help="Path to Measurement Set.")
    run_parser.add_argument(
        "-np", "--norad-path",
        dest="norad_path",
        default=None,
        metavar="FILE",
        help=(
            "Text file of NORAD IDs to include, one per line (blank lines and "
            "'#' comments ignored). Overrides satellites.norad_ids_path and "
            "satellites.norad_ids in the config file."
        ),
    )
    run_parser.add_argument("-sx", "--suffix", default="", help="Image name suffix.")
    run_parser.add_argument("-t", "--timings", action="store_true", help="Enable timing measurements.")
    run_parser.add_argument(
        "-nl", "--no-log",
        dest="log",
        action="store_false",
        default=True,
        help="Do not write the stdout output to a log file (enabled by default).",
    )
    run_parser.add_argument(
        "--extra-orbit-dir",
        dest="extra_orbit_dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory of local orbit files (TLE or OMM) searched, per NORAD "
            "ID, before the managed cache and SatChecker. (To relocate the "
            "managed cache instead, set the ORBIT_CACHE_DIR environment "
            "variable — that is storage, not an additional orbit source.)"
        ),
    )

    # -- light-curve --
    from tabascal.scripts.rfi_estimate import build_parser as _build_lc_parser

    lc_parser = subparsers.add_parser(
        "light-curve",
        help="Matched-filter RFI light-curve extraction from an MS column.",
    )
    _build_lc_parser(lc_parser)

    # -- rfi-per-sat --
    from tabascal.scripts.rfi_per_sat_to_MS import build_parser as _build_ps_parser

    ps_parser = subparsers.add_parser(
        "rfi-per-sat",
        help="Write each satellite's RFI visibility prediction to its own MS column.",
    )
    _build_ps_parser(ps_parser)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "run":
        _run_cmd(args)
    elif args.command == "light-curve":
        _light_curve_cmd(args)
    elif args.command == "rfi-per-sat":
        _rfi_per_sat_cmd(args)


if __name__ == "__main__":
    main()
