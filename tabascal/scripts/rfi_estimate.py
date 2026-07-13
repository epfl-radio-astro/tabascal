"""CLI for matched-filter RFI light-curve extraction from any MS column.

Runnable as ``tabascal light-curve ...`` (subcommand) or standalone via
``python -m tabascal.scripts.rfi_estimate ...``.

Examples
--------
Fetch TLEs for three satellites and matched-filter the flux-calibrated column::

    python -m tabascal.scripts.rfi_estimate \\
        -ms cosmos_polxx.ms -dc REAL_DATA_FLUXCAL \\
        -n 27868,57865,60093 -o cosmos_polxx_mf_light_curves.npz

Take the NORAD IDs from a tabascal config's ``satellites.norad_ids`` instead::

    python -m tabascal.scripts.rfi_estimate -ms cosmos_polxx.ms -c tab_cosmos.yaml \\
        -dc TAB_RES_DATA -o residual_mf_light_curves.npz
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
    parser.add_argument("-ms", "--ms_path", required=True, help="Path to the Measurement Set.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "-n", "--norad-ids", dest="norad_ids", default=None,
        help="Comma/space separated NORAD IDs to matched-filter.",
    )
    src.add_argument(
        "-np", "--norad-path", dest="norad_path", default=None,
        help="Text file of NORAD IDs (one per line or comma/space separated).",
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="Optional tabascal config (.yaml); NORAD IDs read from "
        "satellites.norad_ids if -n/-np not given.",
    )
    parser.add_argument(
        "-z", "--zarr", default=None,
        help="Score a tabascal run straight from its results zarr (map_pred_*.zarr): "
        "the residual is data_col - zarr.vis_obs and the gain template is taken from "
        "the run's own fitted gains. PREFERRED over reading TAB_RES_DATA from the MS, "
        "which every tabascal run overwrites. With -z, -dc is the *reference* column "
        "(e.g. REAL_DATA_FLUXCAL), not the residual column.",
    )
    parser.add_argument(
        "-dc", "--data-col", dest="data_col", default="DATA",
        help="MS data column to matched-filter, or (with -z) the reference data "
        "column the residual is formed against (default: DATA).",
    )
    parser.add_argument(
        "-cr", "--corr", default="xx", choices=["xx", "xy", "yx", "yy"],
        help="Correlation to read (default: xx).",
    )
    parser.add_argument(
        "-f", "--freq", type=float, default=None,
        help="Use only the single channel nearest this frequency (Hz).",
    )
    parser.add_argument(
        "-sx", "--tag", default=None,
        help="Run tag/suffix (e.g. the tabascal -sx). Names the output and its "
        "subdir so residuals of different runs don't collide. Default: the column.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .npz path (default: <ms_dir>/mf/<tag-or-col>/"
        "<ms_stem>_<tag-or-col>_mf_light_curves.npz).",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", default=False,
        help="Also save a per-source z-statistic (residual/floor) spectrogram.",
    )
    parser.add_argument(
        "-g", "--ant-gain", dest="ant_gain", default=None,
        help="Path to an .npz ('gain', shape (n_ant,)) with the complex per-antenna "
        "gain. Included in the matched-filter template, which down-weights low-gain "
        "baselines (the correct way to apply a gain -- dividing the data by it "
        "up-weights the noisiest baselines instead).",
    )
    parser.add_argument(
        "-zc", "--z-crit", dest="z_crit", type=float, default=3.0,
        help="z (residual/floor) threshold for the coverage statistic and band. "
        "Default 3.0: a fully subtracted source has z ~ N(0,1), so the null "
        "coverage within 3 sigma is 99.73%%.",
    )
    parser.add_argument(
        "--include-autos", dest="exclude_autos", action="store_false", default=True,
        help="Include autocorrelation baselines in the beam-former.",
    )
    parser.add_argument(
        "--extra-tle-dir", dest="extra_tle_dir", default=None,
        help="Extra local directory searched for cached TLEs before Space-Track.",
    )
    parser.add_argument(
        "--max-mem-gb", dest="max_mem_gb", type=float, default=1.0,
        help="Memory budget for the matched-filter time-chunk loop (default: 1.0).",
    )
    return parser


def _resolve_norad_ids(args):
    if args.norad_ids:
        return _parse_norad_ids(args.norad_ids)
    if args.norad_path:
        with open(args.norad_path) as fh:
            return _parse_norad_ids(fh.read())
    if args.config:
        from tabascal.config import load_config
        cfg = load_config(args.config)
        ids = cfg.get("satellites", {}).get("norad_ids")
        if ids:
            return [int(x) for x in ids]
    raise SystemExit(
        "No NORAD IDs given. Provide -n/--norad-ids, -np/--norad-path, or a "
        "-c/--config with satellites.norad_ids."
    )


def run(args):
    from tabascal.rfi_estimate import (
        extract_light_curves_from_ms, extract_light_curves_from_zarr,
        save_light_curves_npz, coverage_stats, plot_z_spectrograms, load_ant_gain,
    )

    try:
        norad_ids = _resolve_norad_ids(args)
    except SystemExit:
        if not args.zarr:
            raise
        norad_ids = None   # -z: the zarr carries the run's own TLEs and NORAD ids

    label = args.tag or args.data_col
    stem = os.path.splitext(os.path.basename(args.ms_path.rstrip("/")))[0]
    if args.output:
        out_path = args.output
        out_dir = os.path.dirname(os.path.abspath(out_path))
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ms_path)), "mf", label)
        out_path = os.path.join(out_dir, f"{stem}_{label}_mf_light_curves.npz")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Matched-filter light curves from column '{args.data_col}' ({args.corr})")
    print(f"  MS       : {args.ms_path}")
    print(f"  NORAD IDs: {norad_ids}")

    gain = load_ant_gain(args.ant_gain) if args.ant_gain else None

    common = dict(
        norad_ids=norad_ids, corr=args.corr, data_col=args.data_col, freq=args.freq,
        exclude_autos=args.exclude_autos, extra_tle_dir=args.extra_tle_dir,
        max_mem_gb=args.max_mem_gb, ant_gain=gain,
    )
    if args.zarr:
        print(f"  Residual : {args.data_col} - {args.zarr}")
        result = extract_light_curves_from_zarr(args.ms_path, args.zarr, **common)
    else:
        if args.data_col.startswith("TAB_"):
            print(f"  WARNING  : reading '{args.data_col}' from the MS. Those columns are "
                  f"overwritten by every tabascal run -- pass -z <map_pred_*.zarr> to "
                  f"score the run you actually mean.")
        result = extract_light_curves_from_ms(args.ms_path, **common)
    save_light_curves_npz(out_path, result)
    print(f"  Sources  : {result['titles']}")
    print(f"  Shape    : {result['light_curves'].shape} (n_src, n_freq, n_time)")
    print(f"  Saved    : {out_path}")

    # Coverage statistic (fraction of freq-time cells within |z| <= z_crit).
    cov = coverage_stats(result, z_crit=args.z_crit)
    print(f"\n  Coverage within {args.z_crit:g} sigma. Judge against the NULL column, not the")
    print(f"  analytic 99.73%: the floor assumes independent baselines and is optimistic.")
    print(f"  null = the same statistic on Im(S_hat), a matched source-free null.")
    print(f"  excess = null - cov, the part attributable to a real residual.\n")
    print(f"    {'source':<12} {'cov':>7} {'null':>7} {'excess':>8} {'max|z|':>8}")
    for p in cov["per_source"]:
        print(f"    {p['title']:<12} {p['coverage']*100:6.2f}% {p['null_coverage']*100:6.2f}% "
              f"{p['excess']*100:+7.2f}pp {p['max_z']:8.1f}")
    o = cov["overall"]
    print(f"    {'OVERALL':<12} {o['coverage']*100:6.2f}% {o['null_coverage']*100:6.2f}% "
          f"{(o['null_coverage']-o['coverage'])*100:+7.2f}pp")

    if args.plot:
        png = os.path.join(out_dir, f"{stem}_{label}_z_spectrogram.png")
        plot_z_spectrograms(result, png, z_crit=args.z_crit)
        print(f"  Plot     : {png}")


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
