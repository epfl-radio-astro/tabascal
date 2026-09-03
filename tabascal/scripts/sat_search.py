"""``tabascal search``: which satellite is contaminating this observation?

The user story is "I am told a satellite is in my data, but not which one". Given
a Measurement Set and a TLE snapshot -- a constellation export in a directory, or
an explicit list of NORAD IDs -- this command produces the
``satellites.norad_ids`` list a run needs, measured from the visibilities:

1. propagate every record over the observation and keep the ones that were above
   the horizon (:func:`~tabascal.rfi_estimate.enumerate_candidates`);
2. run the along-track matched-filter scan of GitHub #190 over all of them at
   once, ``vmap``\\ ped and jitted, on the coherent baselines of #189
   (:func:`~tabascal.rfi_estimate.search_candidates`);
3. rank by score, calibrate the top of the ranking against a decohered null, and
   name whatever clears the threshold
   (:func:`~tabascal.rfi_estimate.select_detections`);
4. emit the config fragment, the ranking table, and -- for the satellites it
   named -- the light curves, the epoch-shifted orbit records and the plots.

It is a separate subcommand rather than a flag on ``light-curve`` because
discovery has its own input (a snapshot, not a satellite), its own output (a
config fragment) and its own exit semantics: **0** when something was found,
**3** when the scan ran and nothing cleared the threshold, which is a meaningful,
scriptable result rather than a failure. There is no ``--fit-offset`` here: the
scan *is* the search, and at ``tau = 0`` the MWA Cen A case scored 0.045 against
a candidate median of 0.0446, which is no detection at all.

Saving is threshold-gated by default -- a search meets hundreds of candidates and
a curve extracted at an offset that is not a detection is a curve extracted at
noise -- and ``--save-all`` opens it. The ranking table is written either way: it
is the evidence for a negative.

Nothing here imports JAX: the parser is built by the top-level ``tabascal``
parser, and ``tabascal -h`` must not pay for the run stack.
"""

import argparse
import os

from tabascal.scripts.rfi_estimate import (
    _check_offset_fit_arguments,
    _tau_grid_steps,
    resolve_corr,
    resolve_norad_ids,
    set_precision_for_scan,
)


def build_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Search a satellite snapshot for the ones contaminating an MS."
        )

    parser.add_argument(
        "-ms", "--ms_path", required=True,
        help="Path to the Measurement Set to search.",
    )
    parser.add_argument(
        "-dc", "--data-col", dest="data_col", default="DATA",
        help="MS data column to search (default: DATA).",
    )
    # No `choices`: the correlations tabascal can read live in
    # tabascal.ms.CORR_TYPES, and importing that here would pull the whole
    # JAX/dask stack into `tabascal -h`. resolve_corr checks it once the command
    # is already paying for the read.
    parser.add_argument(
        "-cr", "--corr", default="xx", metavar="CORR",
        help="Correlation to read: a linear (xx/xy/yx/yy), circular (rr/rl/lr/ll) "
        "or Stokes (i/q/u/v) name, whichever the MS holds (default: xx).",
    )
    parser.add_argument(
        "-f", "--freq", type=float, default=None,
        help="Search only the single channel nearest this frequency (Hz).",
    )

    _add_candidate_arguments(parser)
    _add_scan_arguments(parser)
    _add_output_arguments(parser)

    return parser


def _add_candidate_arguments(parser):
    """Where the satellites to score come from, and which of them are up."""
    group = parser.add_argument_group(
        "candidates",
        "The satellites to score. A snapshot directory and an explicit list both "
        "name them and there is no rule for which would win, so exactly one is "
        "required.",
    )
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tle-dir", dest="tle_dir", default=None, metavar="DIR",
        help="Directory of orbit files (TLE or OMM) to search, e.g. a Space-Track "
        "constellation export. One record per satellite is used, the one whose "
        "epoch is nearest the observation.",
    )
    source.add_argument(
        "-n", "--norad-ids", dest="norad_ids", default=None,
        help="Comma/space separated NORAD IDs to score, resolved through the "
        "run's own orbit sources.",
    )
    source.add_argument(
        "-np", "--norad-path", dest="norad_path", default=None, metavar="FILE",
        help="Text file of NORAD IDs (one per line or comma/space separated).",
    )
    group.add_argument(
        "--name-filter", dest="name_filter", default=None, metavar="STR",
        help="Case-insensitive substring of OBJECT_NAME, e.g. STARLINK. Only with "
        "--tle-dir: with -n/-np the candidates are named one by one and a filter "
        "could only drop some of them silently.",
    )
    group.add_argument(
        "--extra-orbit-dir", dest="extra_orbit_dir", default=None, metavar="DIR",
        help="Directory of local orbit files searched, per NORAD ID, before the "
        "managed cache and SatChecker. Seeds an explicit -n/-np list.",
    )
    group.add_argument(
        "--min-elevation", dest="min_elevation", type=float, default=0.0,
        metavar="DEG",
        help="Elevation in degrees at or above which a satellite counts as up "
        "(default: 0.0, the geometric horizon). It screens the candidates and is "
        "the per-frame mask each of them is scored over.",
    )

    return parser


def _add_scan_arguments(parser):
    """The along-track scan and the null, as ``light-curve --fit-offset`` has them.

    Spelled here rather than taken from
    :func:`~tabascal.scripts.rfi_estimate.add_offset_fit_arguments` because two
    of that group's flags have no meaning for a search -- ``--fit-offset``, since
    the scan is the search, and ``--only-detections``, which is spelled the other
    way round as ``--save-all`` -- and the offset step has a different default.
    The *rules* are shared: the same
    :func:`~tabascal.scripts.rfi_estimate._check_offset_fit_arguments` refuses
    the same values by the same names, so one scan means one set of checks.
    """
    group = parser.add_argument_group(
        "along-track offset scan",
        "The scan is the search and is always on: a satellite whose TLE is a "
        "couple of seconds out scores at the candidate median at tau = 0.",
    )
    group.add_argument(
        "--tau-max", dest="tau_max", type=float, default=4.0, metavar="SEC",
        help="Half-width of the offset grid in seconds (default: 4.0), wide "
        "enough for a day-old Starlink TLE. A half-width that is not a whole "
        "number of steps is rounded down, so the grid never reaches past it.",
    )
    group.add_argument(
        "--tau-step", dest="tau_step", type=float, default=0.5, metavar="SEC",
        help="Offset grid step in seconds (default: 0.5). Coarser than "
        "light-curve's 0.25 because the scan is run once per candidate; halve it "
        "once the field is narrowed. The step must still resolve the peak, whose "
        "half-width shrinks as the coherent array grows.",
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
        "metres (default: 300.0). Longer baselines than a candidate can steer are "
        "weighted out rather than added with a random phase.",
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
        help="Per-antenna path scramble in the null, in metres (default: 50.0) -- "
        "tens of wavelengths, so nothing coherent survives it.",
    )
    group.add_argument(
        "--null-top", dest="null_top", type=int, default=5, metavar="N",
        help="How many of the ranked candidates get a null, and so a significance "
        "(default: 5). The null is a whole scan's worth of work per satellite, "
        "and the ones below the cut are still scored and still ranked.",
    )
    group.add_argument(
        "--threshold", dest="threshold", type=float, default=5.0, metavar="SIGMA",
        help="Significance above the decohered null at which a candidate is a "
        "detection (default: 5.0). It carries no trials factor: the scan "
        "maximises over the grid, the channels and every candidate while the null "
        "is drawn at one offset, so it is a working cut rather than a false-alarm "
        "rate.",
    )
    group.add_argument(
        "--runner-up-ratio", dest="runner_up_ratio", type=float, default=1.5,
        metavar="X",
        help="Warn when the second candidate scores within this factor of the "
        "first (default: 1.5). Satellites in the same train partially match each "
        "other's fringes.",
    )
    group.add_argument(
        "--batch-size", dest="batch_size", type=int, default=8, metavar="N",
        help="Most candidates to score per jitted call (default: 8). An "
        "efficiency knob and nothing else; the batch actually run is the smaller "
        "of this and what --max-mem-gb affords.",
    )
    group.add_argument(
        "--max-mem-gb", dest="max_mem_gb", type=float, default=4.0, metavar="GB",
        help="Memory budget for the batch, in gigabytes (default: 4.0). It "
        "counts the two arrays that dominate -- one candidate's fringe model "
        "(n_bl x n_freq x n_time x n_fine complex, one offset at a time) and its "
        "path differences over the whole grid -- which run to gigabytes apiece "
        "on a real array, so this and not --batch-size is usually what decides "
        "how many candidates are scored at once. The weights beside them are not "
        "counted, so it is a sizing heuristic rather than a cap.",
    )
    group.add_argument(
        "--precision", dest="precision", default=None, choices=("single", "double"),
        help="JAX precision the scan runs in (default: single, tabascal's own, "
        "and enough for a fringe model on a path difference of a few kilometres).",
    )

    return parser


def _add_output_arguments(parser):
    """What the search writes, and how much of it."""
    group = parser.add_argument_group("outputs")
    group.add_argument(
        "-o", "--output", default=None, metavar="STEM",
        help="Output path stem (default: <ms_dir>/sat_search/<data column>). "
        "Writes <stem>_ranking.npz and <stem>_config.yaml always, plus "
        "<stem>_light_curves.npz, <stem>_shifted_tles/ and, with -p, the plots.",
    )
    # A directory and no directory: given together one would have to lose
    # silently, so neither is allowed to.
    shifted = group.add_mutually_exclusive_group()
    shifted.add_argument(
        "--write-shifted-tle", dest="write_shifted_tle", default=None, metavar="DIR",
        help="Where to write the detected satellites' orbit records with their "
        "epochs moved by -tau (default: <stem>_shifted_tles). The config fragment "
        "points at them, so a later run reproduces the measured trajectories.",
    )
    shifted.add_argument(
        "--no-shifted-tle", dest="shifted_tle", action="store_false", default=True,
        help="Do not write the epoch-shifted orbit records.",
    )
    group.add_argument(
        "--save-all", dest="save_all", action="store_true", default=False,
        help="Extract light curves (and plots) for every candidate, not only the "
        "detections. The ranking table is written either way.",
    )
    group.add_argument(
        "-p", "--plot", action="store_true", default=False,
        help="Also save the candidate ranking chart and, per saved satellite, its "
        "offset diagnostics.",
    )
    group.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print the whole ranking rather than its top ten.",
    )

    return parser


# ---------------------------------------------------------------------------
# Resolving what argparse cannot express
# ---------------------------------------------------------------------------

def check_arguments(args):
    """Refuse a search that cannot be run, naming the flag, before the MS is read.

    The scan's rules are ``light-curve``'s and are checked by its own function,
    so a bad value means the same thing, and is refused by the same name, from
    either command. What the search adds is spelled out here.
    """
    from math import isfinite

    if args.name_filter is not None and not args.tle_dir:
        raise SystemExit(
            "--name-filter is a substring of the names in a snapshot, so it needs "
            "--tle-dir. With -n/-np the candidates are named one by one and a "
            "filter could only drop some of them silently."
        )
    if args.tle_dir and args.extra_orbit_dir:
        raise SystemExit(
            "--tle-dir and --extra-orbit-dir are two sources of the same thing. "
            "--extra-orbit-dir is where an explicit -n/-np list is resolved from; "
            "with --tle-dir the snapshot already names the candidates, so a "
            "second directory could only be read for satellites it never named. "
            "Give one of them."
        )

    _check_offset_fit_arguments(args)

    if args.batch_size < 1:
        raise SystemExit(
            f"--batch-size is {args.batch_size}; the sweep scores at least one "
            "candidate per call."
        )
    # Written as "not > 0" so a nan is refused too, having lost the comparison;
    # an infinite budget is refused rather than read as "no budget", since a
    # batch sized from it is one the device cannot hold.
    if not args.max_mem_gb > 0 or not isfinite(args.max_mem_gb):
        raise SystemExit(
            f"--max-mem-gb is {args.max_mem_gb:g}; it is a memory budget in "
            "gigabytes, so it must be a positive, finite number."
        )
    if args.null_top < 0:
        raise SystemExit(
            f"--null-top is {args.null_top}; it is how many of the ranked "
            "candidates are calibrated against a null, so it cannot be negative. "
            "Use 0 to rank without measuring any significance."
        )
    if not args.runner_up_ratio > 0:
        raise SystemExit(
            f"--runner-up-ratio is {args.runner_up_ratio:g}; it is the factor the "
            "runner-up is read against, so it must be positive."
        )


def resolve_tau_grid(args):
    """The offsets to scan: whole steps either side of zero, as light-curve's are."""
    import numpy as np

    steps = _tau_grid_steps(args.tau_max, args.tau_step)

    return args.tau_step * np.arange(-steps, steps + 1, dtype=np.float64)


def resolve_output_stem(args, ms_path):
    """Where to write, defaulting beside the MS under ``sat_search/``.

    A stem rather than a file: one search writes a ranking, a config fragment,
    the curves it found and their plots, and they should sit together under one
    name. Named for the column searched, so two columns of one MS do not collide.
    """
    if args.output:
        return args.output

    ms_dir = os.path.dirname(os.path.abspath(str(ms_path).rstrip("/")))

    return os.path.join(ms_dir, "sat_search", args.data_col)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _progress(done, total):
    """A live count of the sweep: a search over a constellation runs for minutes."""
    print(
        f"\r  Scored   : {done}/{total} candidates",
        end="" if done < total else "\n",
        flush=True,
    )


def _print_ranking(search, verbose):
    """The ranking table, which is the log's evidence for whatever is named."""
    table = search["table"]
    shown = table if verbose else table[:10]

    print(
        f"\n  {'rank':>4} {'norad':>7}  {'name':<20} {'max el':>7} {'z2':>9} "
        f"{'tau':>8} {'chan':>5} {'MHz':>10} {'sigma':>8}"
    )
    for row in shown:
        print(
            f"  {row['rank']:>4} {row['norad_id']:>7}  {row['name']:<20.20} "
            f"{row['max_elevation']:7.1f} {row['z2_best']:9.4f} "
            f"{row['tau_best']:+8.2f} {row['best_chan']:>5} "
            f"{row['best_freq'] / 1e6:10.4f} {row['significance']:8.1f}"
        )
    if len(shown) < len(table):
        print(f"  ... and {len(table) - len(shown)} more; -v prints them all.")
    print(
        f"\n  Median z2 {search['median_z2']:.4f} over {len(table)} candidates, "
        f"scored on {search['n_bl_used']} coherent baselines "
        f"(b <= {search['b_coh_max']:.0f} m), "
        f"{search['batch_size']} at a time."
    )


def _write_light_curves(args, ms, search, saved, stem, corr):
    """The saved satellites' curves, through the light-curve command's own writer.

    Extracted at the offset the search measured, not at ``tau = 0``, and carrying
    the fit into the ``.npz`` beside them -- so a successful search emits the
    curves directly and there is no second pass. The threshold gate is applied
    *before* the filter rather than after: over a constellation, beam-forming
    toward every candidate to then drop all but one is the expensive way round.
    """
    import numpy as np

    from tabascal.rfi_estimate import (
        _lc_result,
        _resolve_noise,
        _times_jd,
        attach_offset_fits,
        matched_filter_light_curves,
        rfi_phase_from_records,
        save_light_curves_npz,
    )
    from tabascal.time import to_utc_mjd

    records = [search["candidates"][row["rank"]]["record"] for row in saved]
    fits = [search["fits"][row["rank"]] for row in saved]
    in_view = np.stack([np.asarray(search["frames"][row["rank"]]) for row in saved])
    norad_ids = [row["norad_id"] for row in saved]
    freqs = np.asarray(ms["freqs"])
    times_jd = _times_jd(ms)
    ants_itrf = np.asarray(ms["ants_itrf"])
    flags = None if ms.get("flags") is None else np.asarray(ms["flags"])

    rfi_phase = rfi_phase_from_records(
        records, ants_itrf, times_jd,
        {"ra": float(ms["ra"]), "dec": float(ms["dec"])}, freqs,
        time_offsets_s=[row["tau_best"] for row in saved],
    )
    light_curves, error = matched_filter_light_curves(
        np.asarray(ms["vis_obs"]),
        rfi_phase,
        np.asarray(ms["a1"]),
        np.asarray(ms["a2"]),
        noise=_resolve_noise(ms.get("noise"), "the MS partition"),
        flags=flags,
        in_view=in_view,
    )

    result = _lc_result(
        light_curves, error, norad_ids, freqs,
        to_utc_mjd(ms["times_mjd"], ms["time_scale"]), args.data_col, corr,
        in_view=in_view,
    )
    result["orbit_records"] = records
    result = attach_offset_fits(result, fits, args.threshold)

    path = f"{stem}_light_curves.npz"
    save_light_curves_npz(path, result)

    return path


def _plot(search, selection, saved, stem):
    """The ranking chart, and the offset diagnostics of each saved satellite."""
    from tabascal.rfi_estimate import plot_candidate_ranking, plot_offset_diagnostics

    ranking = plot_candidate_ranking(search, selection, f"{stem}_ranking.png")
    print(f"  Plot     : {ranking}")
    for row in saved:
        png = f"{stem}_offset_{row['norad_id']}.png"
        plot_offset_diagnostics(
            search["fits"][row["rank"]], png,
            title=f"{row['norad_id']} ({row['name']})",
        )
        print(f"  Offset   : {png}")


def _search(args) -> int:
    """Read the MS, score every candidate, and write what was found."""
    import numpy as np

    from tabascal.rfi_estimate import (
        _read_ms,
        _times_jd,
        candidates_from_norad_ids,
        candidates_from_orbit_dir,
        enumerate_candidates,
        search_candidates,
        select_detections,
        write_config_fragment,
        write_search_results,
        write_shifted_orbits,
    )

    corr = resolve_corr(args, None)
    ms_path = os.path.abspath(args.ms_path)
    ms = _read_ms(ms_path, args.freq, corr, args.data_col)
    times_jd = _times_jd(ms)
    ants_itrf = np.asarray(ms["ants_itrf"])

    if args.tle_dir:
        records, names, _ = candidates_from_orbit_dir(
            args.tle_dir, times_jd, name_filter=args.name_filter
        )
    else:
        records, names, _ = candidates_from_norad_ids(
            resolve_norad_ids(args), times_jd, extra_orbit_dir=args.extra_orbit_dir
        )

    candidates = enumerate_candidates(
        records, names, times_jd, ants_itrf, min_elevation=args.min_elevation
    )
    print(
        f"Searching '{args.data_col}' ({corr}) of {ms_path}\n"
        f"  Snapshot : {len(records)} records, {len(candidates)} above "
        f"{args.min_elevation:g} deg elevation"
    )
    if not candidates:
        raise SystemExit(
            f"None of the {len(records)} candidate satellites reaches "
            f"{args.min_elevation:g} degrees elevation during this observation, so "
            "there is nothing above the horizon to search for. Check the snapshot "
            "covers this epoch, or lower --min-elevation."
        )
    # Lowering --min-elevation past the horizon gets a candidate through the
    # screen; it does not get one into the sum. A satellite on the far side of
    # the Earth has no baseline set it could be beam-formed over, so ranking it
    # would name a NORAD ID off geometry that means nothing.
    if not any(candidate["max_elevation"] >= 0.0 for candidate in candidates):
        raise SystemExit(
            f"None of the {len(candidates)} screened satellites rises above the "
            "geometric horizon during this observation, so none of them can "
            "fringe the array and there is nothing to search. Raise "
            "--min-elevation back to 0 or above, and check the snapshot covers "
            "this epoch."
        )

    search = search_candidates(
        np.asarray(ms["vis_obs"]),
        candidates,
        ants_itrf,
        times_jd,
        {"ra": float(ms["ra"]), "dec": float(ms["dec"])},
        np.asarray(ms["freqs"]),
        np.asarray(ms["a1"]),
        np.asarray(ms["a2"]),
        float(ms["int_time"]),
        noise=ms.get("noise"),
        flags=None if ms.get("flags") is None else np.asarray(ms["flags"]),
        taus_s=resolve_tau_grid(args),
        n_fine=args.n_fine,
        sigma_transverse_m=args.sigma_transverse,
        soft_weights=args.soft_weights,
        batch_size=args.batch_size,
        max_mem_gb=args.max_mem_gb,
        n_null=args.null_draws,
        null_jitter_m=args.null_jitter,
        n_null_candidates=args.null_top,
        progress=_progress,
    )
    selection = select_detections(search, args.threshold, args.runner_up_ratio)
    detected = selection["detected"]

    _print_ranking(search, args.verbose)
    for warning in selection["warnings"]:
        print(f"\n  WARNING  : {warning}")

    stem = resolve_output_stem(args, ms_path)
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

    ranking = write_search_results(
        f"{stem}_ranking.npz", search, selection, args.threshold
    )
    print(f"\n  Ranking  : {ranking}")

    # The detected ones only, whether or not the saving is gated: an offset that
    # is not a detection is noise, and a record shifted by it would seed a later
    # run with a trajectory nothing measured.
    shifted_dir = None
    if detected and args.shifted_tle:
        # Absolute, because the config fragment names it: a relative path there
        # would mean whatever the directory the later run is started from means.
        shifted_dir = os.path.abspath(
            args.write_shifted_tle or f"{stem}_shifted_tles"
        )
        shifted = write_shifted_orbits(
            shifted_dir,
            [row["norad_id"] for row in detected],
            [search["candidates"][row["rank"]]["record"] for row in detected],
            [row["tau_best"] for row in detected],
        )
        print(f"  Shifted  : {shifted}")

    config = write_config_fragment(
        f"{stem}_config.yaml", selection, shifted_orbit_dir=shifted_dir
    )
    print(f"  Config   : {config}")

    saved = search["table"] if args.save_all else detected
    if saved:
        curves = _write_light_curves(args, ms, search, saved, stem, corr)
        print(f"  Curves   : {curves}")
    if args.plot:
        _plot(search, selection, saved, stem)

    print("")
    for row in detected:
        print(
            f"  DETECTED {row['norad_id']} ({row['name']}) at tau "
            f"{row['tau_best']:+.2f} s on channel {row['best_chan']} "
            f"({row['best_freq'] / 1e6:.4f} MHz), z2 {row['z2_best']:.4f}, "
            f"{row['significance']:.1f} sigma."
        )
    if detected:
        return 0

    print(
        "  No candidate cleared the threshold of "
        f"{args.threshold:g} sigma. The ranking is written either way -- it is "
        "the evidence for the negative -- but no curve is extracted at an offset "
        "that is not a detection."
    )

    return 3


def run(args) -> int:
    """Search an MS for the satellites in it. 0 found, 3 nothing above threshold.

    The scan's precision is set before the MS is read, since ``read_ms`` builds
    its arrays under the flag as it finds it, and the process is put back as it
    was found afterwards: this module is importable and ``run`` is callable, so a
    global numerics flag left flipped would follow the caller out of the command.
    Both flags the setter touches are restored -- ``jax_enable_x64`` and
    ``jax_default_matmul_precision``, the second of which decides on Ampere+ GPUs
    whether an f32 matmul is really f32 or TF32, and would go unnoticed.
    """
    check_arguments(args)

    import jax

    was_x64 = jax.config.read("jax_enable_x64")
    was_matmul = jax.config.jax_default_matmul_precision
    try:
        set_precision_for_scan(args.precision or "single")
        return _search(args)
    finally:
        jax.config.update("jax_enable_x64", was_x64)
        jax.config.update("jax_default_matmul_precision", was_matmul)


def main():
    import sys

    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
