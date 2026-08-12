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
# 'validate-config' subcommand
# ---------------------------------------------------------------------------

def _validate_config_cmd(args):
    """Check a config file and report every problem, without running anything."""
    import sys
    from tabascal.config import load_config
    from tabascal.validation import ConfigError, validate_config

    try:
        config = load_config(args.config)
        # load_config already validated; re-run only to surface the inert-key note,
        # which is suppressed on the normal run path because the packaged defaults
        # carry several such keys.
        validate_config(config, args.config, report_inert=True)
    except ConfigError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nOK: {args.config}")


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

    # -- validate-config --
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Check a config file for unknown, missing or invalid entries.",
    )
    validate_parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )

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
    elif args.command == "validate-config":
        _validate_config_cmd(args)
    elif args.command == "spacetrack-login":
        _spacetrack_login_cmd(args)


if __name__ == "__main__":
    main()
