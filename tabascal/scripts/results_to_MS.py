#!/usr/bin/env python
"""``tab2MS``: a tabascal results zarr into the MS it was fitted from.

The parser is built separately from the run, as in the other export scripts, so
the argument surface can be tested without importing ``tabascal.write`` -- and
with it the whole JAX stack -- to do it.
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Copy recovered data from a tabascal run and save in a MS file under the column named 'TAB_DATA'."
    )

    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "-z",
        "--results_zarr_path",
        required=True,
        help="File path to the zarr file containing results.",
    )
    parser.add_argument(
        "-d", "--data_col", default="DATA", help="Data column name. Default is DATA"
    )
    parser.add_argument(
        "-c",
        "--corr",
        default=None,
        help=(
            "Override the correlation the results were fitted on, e.g. 'xx'. "
            "Not normally needed: the zarr records it. Required only for a zarr "
            "written before it was recorded, and then only if the MS holds more "
            "than one correlation."
        ),
    )
    parser.add_argument(
        "-gt",
        "--gain_table",
        action="append",
        default=None,
        help=(
            "CASA calibration table the run was fitted with (data.gain_table). "
            "Required if one was used: the MS's data column is still raw, so "
            "without it every column is written a calibration layer away from "
            "the frame the models are in. Repeat the flag, or give a "
            "comma-separated list, to pass several -- in the same order as the "
            "config, since the tables compose in order."
        ),
    )
    parser.add_argument(
        "-o",
        "--caltable-path",
        dest="caltable_path",
        default=None,
        metavar="PATH",
        help=(
            "Where to write the exported calibration table. Defaults to the "
            "results path with a '.B' extension, beside the zarr. The path "
            "given is the only one the export uses: it writes there, and a "
            "table an earlier run left there is what a run with nothing to "
            "export removes."
        ),
    )

    return parser


def run(args):
    # Imported here rather than at module level so building the parser costs
    # nothing; tabascal.write pulls in the whole JAX stack.
    from tabascal.write import write_results_ms

    write_results_ms(
        ms_path=args.ms_path,
        results_zarr_path=args.results_zarr_path,
        data_col=args.data_col,
        corr=args.corr,
        gain_table=_gain_tables(args.gain_table),
        caltable_path=args.caltable_path,
    )


def _gain_tables(values):
    """``-gt`` occurrences, comma lists included, as one ordered list.

    Both spellings mirror the config's ordered list, so a user copying a
    ``data.gain_table`` value straight onto the command line gets the order the
    run composed the tables in.
    """

    if not values:
        return None

    return [
        path.strip()
        for value in values
        for path in value.split(",")
        if path.strip()
    ]


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
