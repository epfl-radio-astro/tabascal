#!/usr/bin/env python
"""``tabascal rfi-per-sat``: per-satellite RFI predictions into MS columns.

Reads ``rfi_vis_src`` from a results zarr -- written by a run with
``data.save_rfi_per_sat: true`` -- and writes one MS column per satellite,
``TAB_RFI_<NORAD id>``. Standalone and re-runnable: the zarr and the MS are all
it needs, so the decomposition can be exported long after the fit.

Image one column to inspect a single satellite's modelled RFI: a real satellite
is a clean streak in exactly one per-source image, while a feature appearing in
several is sky flux the RFI model has split across satellites.

Nothing here imports JAX: the parser is built by the top-level ``tabascal``
parser, and ``tabascal -h`` must not pay for the run stack.
"""

import argparse

#: Mirrors ``tabascal.write.RFI_PER_SAT_PREFIX``, spelled out rather than
#: imported: that module pulls in JAX, and building this parser must not.
DEFAULT_PREFIX = "TAB_RFI_"


def build_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Write each satellite's RFI visibility prediction to "
            "its own MS column, from a tabascal results zarr."
        )

    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "-z", "--results_zarr_path", required=True,
        help="File path to the results zarr (map_pred_*.zarr). It must hold "
        "'rfi_vis_src', which a run writes when data.save_rfi_per_sat is true.",
    )
    parser.add_argument(
        "-p", "--prefix", default=DEFAULT_PREFIX,
        help=f"Column name prefix; the NORAD id is appended. Default "
        f"'{DEFAULT_PREFIX}', giving {DEFAULT_PREFIX}58126.",
    )
    parser.add_argument(
        "-c", "--corr", default=None,
        help="Override the correlation the results were fitted on, e.g. 'xx'. "
        "Not normally needed: the zarr records it. Required only for a zarr "
        "written before it was recorded, and then only if the MS holds more "
        "than one correlation.",
    )

    return parser


def run(args):
    # Imported here rather than at module level so building the parser -- which
    # the top-level `tabascal -h` does -- costs nothing.
    from tabascal.write import write_per_sat_rfi_ms

    write_per_sat_rfi_ms(
        ms_path=args.ms_path,
        results_zarr_path=args.results_zarr_path,
        prefix=args.prefix,
        corr=args.corr,
    )


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
