"""``tabascal light-curve``: matched-filter RFI light-curve extraction.

Two ways in, and they name the satellites differently:

* **A tabascal config** (``-c``). The run's own ``satellites``, ``data`` and
  ``rfi.min_elevation`` sections decide what is filtered for, the MS is read once
  through :class:`~tabascal.config.TabConfig`, and the curves come back ordered
  to match ``satellites.norad_ids``. This is the estimate ``rfi.init:
  matched-filter`` makes internally, written out.
* **An MS plus NORAD IDs** (``-ms`` with ``-n``/``-np``), for an observation
  tabascal has not been configured against.

Either way the output is the ``rfi.est`` interchange format, so it can seed a
later run unchanged.

``--fit-offset`` adds the along-track (time-offset) search of GitHub #190 on top
of either path: each named satellite's ``tau`` is measured against the
visibilities before the curves are extracted, the curves are then extracted at
the offset that was found, and the fit -- ``tau_best`` above all -- is written
into the output beside them. The flags that shape the scan are registered by
:func:`add_offset_fit_arguments`, which the identification search of #191 will
call in turn.

Nothing here imports JAX: the parser is built by the top-level ``tabascal``
parser, and ``tabascal -h`` must not pay for the run stack.
"""

import argparse
import os


def _parse_norad_ids(text):
    return [int(x) for x in text.replace(",", " ").split()]


def build_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Matched-filter RFI light-curve extraction from an MS column."
        )

    # Both name the satellites, and there is no rule for which would win.
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "-c", "--config", default=None,
        help="Path to a tabascal config file. Its satellites, data and "
        "rfi.min_elevation sections drive the extraction and the MS is read once.",
    )
    src.add_argument(
        "-n", "--norad-ids", dest="norad_ids", default=None,
        help="Comma/space separated NORAD IDs to matched-filter.",
    )
    src.add_argument(
        "-np", "--norad-path", dest="norad_path", default=None, metavar="FILE",
        help="Text file of NORAD IDs (one per line or comma/space separated).",
    )
    parser.add_argument(
        "-ms", "--ms_path", default=None,
        help="Path to the Measurement Set. Required without -c, which can take "
        "it from data.ms_path instead.",
    )
    parser.add_argument(
        "-s", "--sim_dir", default=None,
        help="Path to the directory of the simulation, as for `tabascal run`.",
    )
    parser.add_argument(
        "-z", "--zarr", default=None,
        help="Score a run straight from its results zarr (map_pred_*.zarr): the "
        "residual is data_col - zarr.vis_obs. Preferred over reading TAB_RES_DATA "
        "from the MS, which every tabascal run overwrites. With -z, -dc is the "
        "*reference* column the residual is formed against.",
    )
    # No parser default: a default is indistinguishable from a value the user
    # typed, and with -c that would silently overwrite the config's own
    # data.data_col / data.corr on every run. See resolve_data_col.
    parser.add_argument(
        "-dc", "--data-col", dest="data_col", default=None,
        help="MS data column to matched-filter, or (with -z) the reference "
        "column the residual is formed against. Defaults to the config's "
        "data.data_col with -c, and to DATA otherwise.",
    )
    # No `choices`: the correlations tabascal can read live in
    # tabascal.ms.CORR_TYPES, and importing that here to list them would pull
    # the whole JAX/dask stack into `tabascal -h`. The value is checked against
    # that table in resolve_corr instead, once the run is already paying for it.
    parser.add_argument(
        "-cr", "--corr", default=None, metavar="CORR",
        help="Correlation to read: a linear (xx/xy/yx/yy), circular "
        "(rr/rl/lr/ll) or Stokes (i/q/u/v) name, whichever the MS holds. "
        "Defaults to the config's data.corr with -c, and to xx otherwise.",
    )
    parser.add_argument(
        "-f", "--freq", type=float, default=None,
        help="Use only the single channel nearest this frequency (Hz).",
    )
    parser.add_argument(
        "-sx", "--tag", default=None,
        help="Run tag/suffix (e.g. the tabascal -sx). Names the output so "
        "residuals of different runs do not collide. Default: the column.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .npz path (default: <ms_dir>/light_curves/<tag-or-column>.npz).",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", default=False,
        help="Also save a per-source z-statistic (residual/floor) spectrogram.",
    )
    parser.add_argument(
        "-zc", "--z-crit", dest="z_crit", type=float, default=3.0,
        help="z (residual/floor) threshold for the coverage statistic and band "
        "(default: 3.0).",
    )
    # A cut and no cut at all: given together one has to lose silently, so
    # neither is allowed to.
    elevation = parser.add_mutually_exclusive_group()
    elevation.add_argument(
        "--min-elevation", dest="min_elevation", type=float, default=None,
        metavar="DEG",
        help="Elevation in degrees below which a satellite is not filtered for. "
        "Defaults to the config's rfi.min_elevation with -c, and to 0 (the "
        "geometric horizon) otherwise.",
    )
    elevation.add_argument(
        "--no-elevation-cut", dest="elevation_cut", action="store_false",
        default=True,
        help="Filter for every satellite at every timestep, however far below "
        "the horizon it is.",
    )
    parser.add_argument(
        "--include-autos", dest="exclude_autos", action="store_false", default=True,
        help="Include autocorrelation baselines in the beam-former.",
    )
    parser.add_argument(
        "--extra-orbit-dir", dest="extra_orbit_dir", default=None, metavar="DIR",
        help="Directory of local orbit files (TLE or OMM) searched, per NORAD "
        "ID, before the managed cache and SatChecker.",
    )
    parser.add_argument(
        "--max-mem-gb", dest="max_mem_gb", type=float, default=1.0,
        help="Memory budget for the matched-filter time-chunk loop (default: 1.0).",
    )

    add_offset_fit_arguments(parser)

    return parser


#: The offset-fit options and the values they take when nobody asks for them.
#: Compared against the parsed arguments so a scan flag given without
#: ``--fit-offset`` is refused rather than quietly ignored.
_OFFSET_FIT_DEFAULTS = {
    "tau_max": 4.0,
    "tau_step": 0.25,
    "n_fine": 40,
    "sigma_transverse": 300.0,
    "soft_weights": False,
    "null_draws": 200,
    "null_jitter": 50.0,
    "threshold": 5.0,
    "only_detections": False,
    "write_shifted_tle": None,
    "precision": None,
}

#: Most offset grid points a scan will build. The peak is a fraction of a second
#: wide, so a grid finer than that buys nothing and a step small enough to be a
#: typo is an allocation rather than a search.
_MAX_TAU_GRID_POINTS = 1_000_000

#: Offset-fit options that must be a finite number to mean anything at all, and
#: the flag each is spelled with. A nan or an inf types cleanly through argparse
#: and then poisons the statistic somewhere far from the flag that caused it.
_FINITE_OFFSET_FIT_OPTIONS = (
    ("tau_max", "--tau-max"),
    ("tau_step", "--tau-step"),
    ("sigma_transverse", "--sigma-transverse"),
    ("null_jitter", "--null-jitter"),
    ("threshold", "--threshold"),
)


def add_offset_fit_arguments(parser):
    """Register the along-track (time-offset) search options on *parser*.

    A shared builder rather than a list spelled out per subcommand: the
    identification search of #191 offers the same scan over many candidates and
    must not drift from this one on what ``--n-fine`` or ``--threshold`` mean.
    """
    group = parser.add_argument_group(
        "along-track offset fit",
        "Measure each satellite's along-track time offset -- the dominant term "
        "of a TLE's error -- before the curves are extracted.",
    )
    group.add_argument(
        "--fit-offset", dest="fit_offset", action="store_true", default=False,
        help="Scan the along-track offset for each named satellite, extract the "
        "curves at the offset found, and record the fit in the output.",
    )
    group.add_argument(
        "--tau-max", dest="tau_max", type=float, default=4.0, metavar="SEC",
        help="Half-width of the offset grid in seconds (default: 4.0), wide "
        "enough for a day-old Starlink TLE. A half-width that is not a whole "
        "number of steps is rounded down, so the grid never reaches past it.",
    )
    group.add_argument(
        "--tau-step", dest="tau_step", type=float, default=0.25, metavar="SEC",
        help="Offset grid step in seconds (default: 0.25). The grid is this "
        "step times the integers out to the half-width, so tau = 0 is always "
        "on it. The step must resolve the peak, whose half-width shrinks as the "
        "coherent array grows. A grid of more than a million points is refused: "
        "the peak is a fraction of a second wide, so a step that fine buys "
        "nothing, and the fix is a coarser step.",
    )
    group.add_argument(
        "--n-fine", dest="n_fine", type=int, default=40, metavar="N",
        help="Sub-steps per integration in the fringe model (default: 40). Too "
        "few and the model cannot follow the fringe inside a dump.",
    )
    group.add_argument(
        "--sigma-transverse", dest="sigma_transverse", type=float, default=300.0,
        metavar="M",
        help="Transverse orbit error the coherent-baseline cut is sized by, in "
        "metres (default: 300.0). Longer baselines than it can steer are "
        "dropped rather than added with a random phase.",
    )
    group.add_argument(
        "--soft-weights", dest="soft_weights", action="store_true", default=False,
        help="Taper the marginal baselines with a Gaussian on the coherence "
        "length instead of cutting at it.",
    )
    group.add_argument(
        "--null-draws", dest="null_draws", type=int, default=200, metavar="N",
        help="Decohered-antenna draws the significance is measured against "
        "(default: 200).",
    )
    group.add_argument(
        "--null-jitter", dest="null_jitter", type=float, default=50.0, metavar="M",
        help="Per-antenna path scramble in the null, in metres (default: 50.0) "
        "-- tens of wavelengths, so nothing coherent survives it.",
    )
    group.add_argument(
        "--threshold", dest="threshold", type=float, default=5.0, metavar="SIGMA",
        help="Significance above the decohered null at which a fit counts as a "
        "detection (default: 5.0). It carries no trials factor: the scan "
        "maximises over the grid and the channels while the null is drawn at the "
        "best offset, so it is a working cut rather than a false-alarm rate.",
    )
    group.add_argument(
        "--only-detections", dest="only_detections", action="store_true",
        default=False,
        help="Save only the satellites that clear the threshold. Every fit is "
        "still reported; a curve extracted at an offset that is not a detection "
        "is a curve extracted at noise.",
    )
    group.add_argument(
        "--write-shifted-tle", dest="write_shifted_tle", default=None, metavar="DIR",
        help="Write the detected satellites' orbit records with their epochs "
        "moved by -tau into DIR, for a later run's --extra-orbit-dir.",
    )
    group.add_argument(
        "--precision", dest="precision", default=None,
        choices=("single", "double"),
        help="JAX precision the scan runs in. With -c the config's "
        "model.precision decides unless this is given; without -c the default "
        "is single, which is tabascal's own and enough for a fringe model on a "
        "path difference of a few kilometres.",
    )

    return parser


# ---------------------------------------------------------------------------
# Resolving what argparse cannot express
# ---------------------------------------------------------------------------

def resolve_norad_ids(args):
    """The satellites to filter for, in the manual mode."""
    if args.norad_ids:
        return _parse_norad_ids(args.norad_ids)
    if args.norad_path:
        with open(args.norad_path) as fh:
            return _parse_norad_ids(fh.read())

    raise SystemExit(
        "No NORAD IDs given. Provide -n/--norad-ids, -np/--norad-path, or a "
        "-c/--config whose satellites section names them."
    )


def resolve_ms_path(args, config):
    """The MS to read: the flag, the config's ``data.ms_path``, or the sim dir.

    The same precedence ``tabascal run`` uses, so a config that runs points the
    extractor at the same visibilities without being told twice.
    """
    if args.ms_path:
        return os.path.abspath(args.ms_path)

    if config is not None:
        ms_path = config.get("data", {}).get("ms_path")
        if ms_path:
            return os.path.abspath(ms_path)

    sim_dir = args.sim_dir or (config or {}).get("data", {}).get("sim_dir")
    if sim_dir:
        sim_dir = os.path.abspath(sim_dir)
        return os.path.join(sim_dir, f"{os.path.basename(sim_dir)}.ms")

    raise SystemExit(
        "No Measurement Set given. Provide -ms/--ms_path, -s/--sim_dir, or a "
        "config with data.ms_path or data.sim_dir set."
    )


def resolve_data_col(args, config):
    """The column to filter: the flag, else the config's, else ``DATA``."""
    if args.data_col is not None:
        return args.data_col
    if config is not None:
        return config.get("data", {}).get("data_col") or "DATA"

    return "DATA"


def resolve_corr(args, config):
    """The correlation to read: the flag, else the config's, else ``xx``.

    Checked against :data:`tabascal.ms.CORR_TYPES` rather than against a list
    repeated here, so the command accepts exactly what the reader resolves --
    circular and Stokes MSs included -- and cannot drift from it. The import is
    deferred because it carries the JAX stack, which ``tabascal -h`` must not
    pay for; by the time this runs the command is committed to reading an MS
    anyway.
    """
    if args.corr is not None:
        corr = args.corr
    elif config is not None:
        corr = config.get("data", {}).get("corr") or "xx"
    else:
        corr = "xx"

    from tabascal.ms import CORR_TYPES

    if str(corr).lower() not in CORR_TYPES:
        raise SystemExit(
            f"Correlation {corr!r} is not one tabascal can read. Choose from "
            f"{', '.join(sorted(CORR_TYPES))} -- whichever the MS actually holds."
        )

    return corr


def resolve_min_elevation(args, config):
    """The elevation cut: the flag, else the config's, else the horizon."""
    if not args.elevation_cut:
        return None
    if args.min_elevation is not None:
        return float(args.min_elevation)
    if config is not None:
        return config.get("rfi", {}).get("min_elevation")

    return 0.0


def _tau_grid_steps(tau_max, tau_step):
    """Whole steps of ``tau_step`` that fit either side of zero.

    The scan measures a *correction*, so ``tau = 0`` -- the trajectory as the
    elements give it -- has to be one of the points: it is the reference the peak
    is read against. Marching from ``-tau_max`` in steps that do not divide it
    loses that point altogether (4 s in steps of 3 gives -4, -1, 2, 5) and puts a
    sample beyond the half-width the caller asked for. Counting whole steps out
    from zero keeps the grid symmetric, centred and inside its own bounds, at the
    cost of a shorter reach when the step does not divide -- which is the honest
    reading of both flags.
    """
    from math import floor

    return int(floor(float(tau_max) / float(tau_step)))


def _check_offset_fit_arguments(args):
    """Refuse a scan setting that cannot describe a scan, naming the flag.

    argparse types a value; it cannot say whether it means anything. Each of
    these produces something that is not a search -- an empty or infinite offset
    grid, a model with no sub-steps inside an integration, a null of one draw
    whose spread is zero so the significance comes out infinite, a nan that
    quietly spreads through every sum -- and each fails somewhere far from the
    flag that caused it. Saying so here, by name, is the difference between a
    typo and a mystery.

    The finiteness checks come first: a nan loses every comparison, so it would
    slip through the range tests below and be caught by nothing.

    ``--tau-max 0`` and ``--n-fine 1`` are left alone. The first is a grid of one
    point at tau = 0 -- the honest way to ask for the statistic without a scan --
    and the second is the unsmeared template the forward model itself uses.
    """
    from math import isfinite

    for name, flag in _FINITE_OFFSET_FIT_OPTIONS:
        if not isfinite(float(getattr(args, name))):
            raise SystemExit(
                f"{flag} is {getattr(args, name)}, which is not a number the "
                "scan can be run with."
            )

    if args.tau_step <= 0:
        raise SystemExit(
            f"--tau-step is {args.tau_step:g}; the offset grid steps forward, so "
            "it must be positive."
        )
    if args.tau_max < 0:
        raise SystemExit(
            f"--tau-max is {args.tau_max:g}; it is the half-width of a grid "
            "centred on zero, so it cannot be negative. Use 0 for a single "
            "point at tau = 0."
        )
    if args.n_fine < 1:
        raise SystemExit(
            f"--n-fine is {args.n_fine}; the fringe model needs at least one "
            "sub-step per integration."
        )
    if args.null_draws < 2:
        raise SystemExit(
            f"--null-draws is {args.null_draws}; the null needs at least two "
            "draws to have a spread, and a significance is measured in them."
        )
    if args.null_jitter <= 0:
        raise SystemExit(
            f"--null-jitter is {args.null_jitter:g}; a null that moves the "
            "antennas by nothing is the detection itself, not a null."
        )

    # Counted before the array exists, so an absurd step is caught at the flag
    # rather than at the allocation.
    n_points = 2 * _tau_grid_steps(args.tau_max, args.tau_step) + 1
    if n_points > _MAX_TAU_GRID_POINTS:
        raise SystemExit(
            f"--tau-step {args.tau_step:g} over +-{args.tau_max:g} s is "
            f"{n_points} offsets, past the {_MAX_TAU_GRID_POINTS} this scan "
            "will build. The peak is a fraction of a second wide, so a step "
            "that fine buys nothing; widen it."
        )


def resolve_offset_fit(args):
    """The scan settings, or ``None`` when no scan was asked for.

    Also the place a scan flag given *without* ``--fit-offset`` is caught.
    Ignoring one silently would report a default-grid fit as the one that was
    asked for, which is worse than not fitting at all.

    ``precision`` comes back with the rest, but it is not a fit argument: the
    drivers pop it and apply it before anything is read. See
    :func:`set_precision_for_scan`.
    """
    given = [
        name for name, default in _OFFSET_FIT_DEFAULTS.items()
        if getattr(args, name, default) != default
    ]

    if not getattr(args, "fit_offset", False):
        if given:
            flags = ", ".join("--" + name.replace("_", "-") for name in sorted(given))
            raise SystemExit(
                f"No along-track offset scan was asked for, so {flags} would be "
                "ignored. Add --fit-offset to run the scan."
            )
        return None

    _check_offset_fit_arguments(args)

    import numpy as np

    steps = _tau_grid_steps(args.tau_max, args.tau_step)

    return dict(
        precision=args.precision or "single",
        taus_s=args.tau_step * np.arange(-steps, steps + 1, dtype=np.float64),
        n_fine=args.n_fine,
        sigma_transverse_m=args.sigma_transverse,
        soft_weights=args.soft_weights,
        n_null=args.null_draws,
        null_jitter_m=args.null_jitter,
        threshold_sigma=args.threshold,
    )


def set_precision_for_scan(precision: str):
    """Put JAX into ``precision`` before anything numerical is built.

    Importing the estimator pulls in sgp4jax, which enables x64 at import time,
    so a standalone (``-ms``) scan would silently run in double whatever was
    asked for -- and so would the visibilities, since ``read_ms`` builds its
    arrays under the flag as it finds it. The ``-c`` path never reaches here:
    ``set_precision(config)`` has already run there, from the run's own
    ``model.precision``, and setting it a second time from a flag nobody gave
    would overrule the config on its own subject.

    Defers to :func:`tabascal.scripts._run_tabascal_impl.set_precision` rather
    than repeating its import ordering, which is what makes the toggle stick.
    """
    from tabascal.scripts._run_tabascal_impl import set_precision

    return set_precision({"model": {"precision": precision}})


def resolve_output(args, ms_path, data_col):
    """Where to write, defaulting beside the MS under ``light_curves/``.

    Named for the column actually filtered rather than for the flag, so a ``-c``
    run whose config selects ``TAB_RES_DATA`` does not write ``DATA.npz``.

    The suffix is normalised because ``np.savez`` appends ``.npz`` itself: told
    to write ``curves`` it writes ``curves.npz``, and a path reported without it
    names a file that is not there.
    """
    label = args.tag or data_col
    ms_dir = os.path.dirname(os.path.abspath(str(ms_path).rstrip("/")))
    path = args.output or os.path.join(ms_dir, "light_curves", f"{label}.npz")

    return path if path.endswith(".npz") else path + ".npz"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _print_coverage(result, z_crit):
    from tabascal.rfi_estimate import coverage_stats, has_noise_scale

    if not has_noise_scale(result):
        print(
            "\n  No noise floor, so no coverage statistic: the MS carried no "
            "usable\n  noise column and none was given, so the light curves are "
            "unscaled and\n  the z statistic is nan. The curves themselves are "
            "still in the output."
        )
        return

    cov = coverage_stats(result, z_crit=z_crit)
    o = cov["overall"]
    print(
        f"\n  Coverage of '{result['data_col']}' ({result['corr']}) within "
        f"{z_crit:g} sigma."
    )
    print("  cov/null/excess read Re(S_hat), which assumes that column is PHASE")
    print("  CALIBRATED. |S| is the same statistic on |S_hat|/error <= "
          f"{o['amp_crit']:.2f} (the")
    print("  Rayleigh cut enclosing the same probability); it survives a phase")
    print("  common to every baseline, which empties Re(S_hat) into the null.")
    print("  Neither survives an uncalibrated ANTENNA gain: those phases")
    print("  decorrelate the sum itself, shrinking the estimate, so on a raw")
    print("  column both numbers understate what is there. Judge cov against")
    print("  the NULL column, not the analytic value: the floor assumes")
    print("  independent baselines and is optimistic. null = the same statistic")
    print("  on Im(S_hat), a matched source-free null. excess = null - cov, the")
    print("  part attributable to a real residual.\n")
    print(
        f"    {'source':<12} {'cov':>7} {'null':>7} {'excess':>8} "
        f"{'|S|':>7} {'max|z|':>8}"
    )
    for p in cov["per_source"]:
        print(
            f"    {p['title']:<12} {p['coverage'] * 100:6.2f}% "
            f"{p['null_coverage'] * 100:6.2f}% {p['excess'] * 100:+7.2f}pp "
            f"{p['amp_coverage'] * 100:6.2f}% {p['max_z']:8.1f}"
        )
    print(
        f"    {'OVERALL':<12} {o['coverage'] * 100:6.2f}% "
        f"{o['null_coverage'] * 100:6.2f}% "
        f"{(o['null_coverage'] - o['coverage']) * 100:+7.2f}pp "
        f"{o['amp_coverage'] * 100:6.2f}%"
    )


def _from_config(args, offset_fit=None):
    """The in-process path: one MS read, through the run's own TabConfig.

    The config is the default for everything it names -- the column, the
    correlation, the elevation cut -- and a flag overrides it only when one was
    actually given.
    """
    from tabascal.config import TabConfig, load_config
    from tabascal.rfi_estimate import light_curves_from_config
    from tabascal.scripts._run_tabascal_impl import set_precision

    config = load_config(args.config)
    # The config owns the precision here; an explicit --precision overrides it
    # through the config itself, so there is still only one place it is set.
    if offset_fit is not None:
        precision = offset_fit.pop("precision")
        if args.precision is not None:
            config.setdefault("model", {})["precision"] = precision
    set_precision(config)

    ms_path = resolve_ms_path(args, config)
    config["data"]["ms_path"] = ms_path
    config["data"]["data_col"] = resolve_data_col(args, config)
    config["data"]["corr"] = resolve_corr(args, config)
    if args.freq is not None:
        config["data"]["freq"] = args.freq
    if args.extra_orbit_dir:
        config["satellites"]["extra_orbit_dir"] = args.extra_orbit_dir
    # The satellites are the config's own: -n/-np are refused alongside -c, since
    # both name them and there is no rule for which would win.
    config["rfi"]["min_elevation"] = resolve_min_elevation(args, config)

    # An MS with no usable noise column is not fatal here: the curves are still
    # measured, and come back unweighted and unscaled with nan errors, which is
    # what this command documents. Nor is a satellite that never rose: its curve
    # is zero, which is a measurement, and stopping would leave
    # --no-elevation-cut -- which drops the cut for every satellite -- as the
    # only way to measure the ones that were up. Inference keeps both strict
    # defaults: neither an unweighted likelihood nor a fully-masked satellite
    # has anything to fit.
    tab_config = TabConfig(
        config, ms_path, require_noise=False, require_in_view=False
    )

    vis = None
    if args.zarr:
        import numpy as np
        import xarray as xr

        from tabascal.rfi_estimate import (
            _check_zarr_identity,
            _model_on_ms_channels,
        )

        # The same checks and the same alignment the standalone residual goes
        # through: subtracting positionally would meet a full-band store with a
        # config narrowed by data.freq and difference two different channels,
        # and a store from another observation would be differenced at all.
        xds = xr.open_zarr(args.zarr)
        _check_zarr_identity(
            xds,
            args.zarr,
            len(np.asarray(tab_config.a1)),
            tab_config.times_mjd,
            config["data"]["corr"],
        )
        model = _model_on_ms_channels(
            xds,
            tab_config.freqs,
            args.zarr,
            getattr(tab_config, "chan_widths", None),
        )
        vis = np.asarray(tab_config.vis_obs) - model

    result = light_curves_from_config(
        tab_config,
        vis=vis,
        exclude_autos=args.exclude_autos,
        max_mem_gb=args.max_mem_gb,
        offset_fit=offset_fit,
    )
    if args.zarr:
        result["data_col"] = (
            f"{config['data']['data_col']} - "
            f"{os.path.basename(str(args.zarr).rstrip('/'))}"
        )

    return ms_path, result, config["data"]["data_col"]


def _from_ms(args, offset_fit=None):
    """The standalone path: an MS and an explicit satellite list.

    The scan's precision is set here, before ``read_ms`` builds any jnp array
    whose dtype follows the flag, and the process is put back as it was found
    afterwards: this module is importable and ``run`` is callable, so a global
    numerics flag left flipped would follow the caller out of the command.

    *Both* the flags :func:`set_precision_for_scan` touches are restored.
    ``jax_enable_x64`` is the obvious one; ``jax_default_matmul_precision`` is
    the one that would go unnoticed, and on Ampere+ GPUs it decides whether an
    f32 matmul is really f32 or TF32 -- so leaving it pinned would quietly
    rewrite the numerics of whatever the caller does next. The setter itself is
    inside the ``try``, since a failure part-way through it leaves exactly the
    half-applied state the restore exists for.

    The ``-c`` path does not restore anything, and should not: its precision is
    the run's own, taken from its config, and is meant to stand.
    """
    if offset_fit is None:
        return _extract_from_ms(args, None)

    import jax

    was_x64 = jax.config.read("jax_enable_x64")
    was_matmul = jax.config.jax_default_matmul_precision
    try:
        set_precision_for_scan(offset_fit.pop("precision"))
        return _extract_from_ms(args, offset_fit)
    finally:
        jax.config.update("jax_enable_x64", was_x64)
        jax.config.update("jax_default_matmul_precision", was_matmul)


def _extract_from_ms(args, offset_fit):
    """Read the MS and filter it, in whatever precision is already in force."""
    from tabascal.rfi_estimate import (
        extract_light_curves_from_ms,
        extract_light_curves_from_zarr,
    )

    ms_path = resolve_ms_path(args, None)
    norad_ids = resolve_norad_ids(args)
    data_col = resolve_data_col(args, None)

    common = dict(
        norad_ids=norad_ids,
        corr=resolve_corr(args, None),
        data_col=data_col,
        freq=args.freq,
        exclude_autos=args.exclude_autos,
        extra_orbit_dir=args.extra_orbit_dir,
        min_elevation=resolve_min_elevation(args, None),
        max_mem_gb=args.max_mem_gb,
        offset_fit=offset_fit,
    )
    if args.zarr:
        return (
            ms_path,
            extract_light_curves_from_zarr(ms_path, args.zarr, **common),
            data_col,
        )

    if data_col.startswith("TAB_"):
        print(
            f"  WARNING  : reading '{data_col}' from the MS. Those columns "
            "are overwritten by every tabascal run -- pass -z <map_pred_*.zarr> "
            "to score the run you actually mean."
        )

    return ms_path, extract_light_curves_from_ms(ms_path, **common), data_col


def _gate_on_detections(result):
    """Drop the sources that did not clear the threshold, and say so."""
    import numpy as np

    from tabascal.rfi_estimate import select_sources

    detected = np.asarray(result["detected"], dtype=bool)
    if not detected.any():
        print(
            "  No satellite cleared the detection threshold, so no light curve "
            "is saved for any of them; the fits themselves are still written."
        )

    return select_sources(result, detected)


def _write_shifted_orbits(result, directory):
    """Write the detected satellites' epoch-shifted records into ``directory``.

    The detected ones only, whether or not the saving was gated: an offset that
    is not a detection is noise, and a record shifted by it would seed a later
    run with a trajectory nothing measured.
    """
    import numpy as np

    from tabascal.rfi_estimate import write_shifted_orbits

    detected = np.flatnonzero(np.asarray(result["detected"], dtype=bool))
    if detected.size == 0:
        print("  Shifted  : nothing detected, so no orbit record is worth shifting.")
        return

    path = write_shifted_orbits(
        directory,
        [result["norad_ids"][int(i)] for i in detected],
        [result["orbit_records"][int(i)] for i in detected],
        [float(result["tau_best"][int(i)]) for i in detected],
    )
    print(f"  Shifted  : {path}")


def run(args):
    from tabascal.rfi_estimate import save_light_curves_npz

    offset_fit = resolve_offset_fit(args)

    # Printed after the resolution rather than before it: with -c the column and
    # correlation are the config's, and this is what was actually read.
    ms_path, result, data_col = (
        _from_config(args, offset_fit) if args.config else _from_ms(args, offset_fit)
    )
    print(
        f"Matched-filter light curves from column '{data_col}' ({result['corr']})"
    )

    if offset_fit is not None and args.only_detections:
        result = _gate_on_detections(result)

    out_path = resolve_output(args, ms_path, data_col)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_light_curves_npz(out_path, result)

    print(f"  MS       : {ms_path}")
    print(f"  Sources  : {result['titles']}")
    print(f"  Shape    : {result['light_curves'].shape} (n_src, n_freq, n_time)")
    print(f"  Saved    : {out_path}")

    if offset_fit is not None and args.write_shifted_tle:
        _write_shifted_orbits(result, args.write_shifted_tle)

    if len(result["titles"]) == 0:
        return

    _print_coverage(result, args.z_crit)

    if args.plot:
        from tabascal.rfi_estimate import plot_z_spectrograms

        png = os.path.splitext(out_path)[0] + "_z_spectrogram.png"
        plot_z_spectrograms(result, png, z_crit=args.z_crit)
        print(f"  Plot     : {png}")

        if offset_fit is not None:
            from tabascal.rfi_estimate import plot_offset_diagnostics

            stem = os.path.splitext(out_path)[0]
            for norad_id, fit in zip(result["norad_ids"], result["offset_fits"]):
                png = f"{stem}_offset_{norad_id}.png"
                plot_offset_diagnostics(fit, png, title=str(norad_id))
                print(f"  Offset   : {png}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
