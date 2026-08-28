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


def _from_config(args):
    """The in-process path: one MS read, through the run's own TabConfig.

    The config is the default for everything it names -- the column, the
    correlation, the elevation cut -- and a flag overrides it only when one was
    actually given.
    """
    from tabascal.config import TabConfig, load_config
    from tabascal.rfi_estimate import light_curves_from_config
    from tabascal.scripts._run_tabascal_impl import set_precision

    config = load_config(args.config)
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
    )
    if args.zarr:
        result["data_col"] = (
            f"{config['data']['data_col']} - "
            f"{os.path.basename(str(args.zarr).rstrip('/'))}"
        )

    return ms_path, result, config["data"]["data_col"]


def _from_ms(args):
    """The standalone path: an MS and an explicit satellite list."""
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


def run(args):
    from tabascal.rfi_estimate import save_light_curves_npz

    # Printed after the resolution rather than before it: with -c the column and
    # correlation are the config's, and this is what was actually read.
    ms_path, result, data_col = _from_config(args) if args.config else _from_ms(args)
    print(
        f"Matched-filter light curves from column '{data_col}' ({result['corr']})"
    )

    out_path = resolve_output(args, ms_path, data_col)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_light_curves_npz(out_path, result)

    print(f"  MS       : {ms_path}")
    print(f"  Sources  : {result['titles']}")
    print(f"  Shape    : {result['light_curves'].shape} (n_src, n_freq, n_time)")
    print(f"  Saved    : {out_path}")

    _print_coverage(result, args.z_crit)

    if args.plot:
        from tabascal.rfi_estimate import plot_z_spectrograms

        png = os.path.splitext(out_path)[0] + "_z_spectrogram.png"
        plot_z_spectrograms(result, png, z_crit=args.z_crit)
        print(f"  Plot     : {png}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
