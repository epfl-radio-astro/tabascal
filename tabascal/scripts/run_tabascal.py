# ---------------------------------------------------------------------------
# 'run' subcommand
# ---------------------------------------------------------------------------

def _run_cmd(args):
    # Imported lazily so the lightweight subcommands (and --help) don't pay the
    # JAX import cost. The heavy implementation lives in a separate module.
    from tabascal.scripts._run_tabascal_impl import run
    run(args)


# ---------------------------------------------------------------------------
# 'light-curve' subcommand
# ---------------------------------------------------------------------------

def _light_curve_cmd(args):
    # Imported lazily so --help and other subcommands don't pay the JAX cost.
    from tabascal.scripts.rfi_estimate import run
    run(args)


# ---------------------------------------------------------------------------
# 'spacetrack-login' subcommand
# ---------------------------------------------------------------------------

def _spacetrack_login_cmd(args):
    import getpass
    from tabascal.tle import save_spacetrack_credentials, spacetrack_config_path

    username = args.username or input("Space-Track username (email): ")
    password = getpass.getpass("Space-Track password: ")
    path = save_spacetrack_credentials(username, password)
    print(f"Credentials saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="tabascal CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_parser = subparsers.add_parser("run", help="Apply tabascal to a simulation.")
    run_parser.add_argument("-c", "--config", required=True, help="Path to the config file.")
    run_parser.add_argument("-s", "--sim_dir", help="Path to the directory of the simulation.")
    run_parser.add_argument("-ms", "--ms_path", help="Path to Measurement Set.")
    run_parser.add_argument("-np", "--norad_path", help="Path to text file containing NORAD IDs to include.")
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
        "--extra-tle-dir",
        dest="extra_tle_dir",
        default=None,
        metavar="DIR",
        help="Extra directory searched for cached TLEs before the managed cache and Space-Track.",
    )

    # -- light-curve --
    from tabascal.scripts.rfi_estimate import build_parser as _build_lc_parser
    lc_parser = subparsers.add_parser(
        "light-curve",
        help="Matched-filter RFI light-curve extraction from an MS column.",
    )
    _build_lc_parser(lc_parser)

    # -- spacetrack-login --
    login_parser = subparsers.add_parser(
        "spacetrack-login",
        help="Save Space-Track credentials to the user config file.",
    )
    login_parser.add_argument(
        "-u", "--username",
        default=None,
        help="Space-Track username (email address). Prompted interactively if not given.",
    )

    args = parser.parse_args()

    if args.command == "run":
        _run_cmd(args)
    elif args.command == "light-curve":
        _light_curve_cmd(args)
    elif args.command == "spacetrack-login":
        _spacetrack_login_cmd(args)


if __name__ == "__main__":
    main()
