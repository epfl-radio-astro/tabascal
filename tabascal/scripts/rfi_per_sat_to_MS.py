#!/usr/bin/env python
"""``tabascal rfi-per-sat``: per-satellite RFI predictions into MS columns.

Reads ``rfi_vis_src`` from a run's results -- written when the run had
``data.save_rfi_per_sat: true`` -- and writes one MS column per satellite,
``TAB_RFI_<NORAD id>``. Takes the run directory, as ``tab2MS`` does, and the
same results: ``map_pred`` if the run optimised, ``init_pred`` if not.
Standalone and re-runnable, so the decomposition can be exported long after the
fit.

Image one column to inspect a single satellite's modelled RFI: a real satellite
is a clean streak in exactly one per-source image, while a feature appearing in
several is sky flux the RFI model has split across satellites.

Nothing here imports JAX: the parser is built by the top-level ``tabascal``
parser, and ``tabascal -h`` must not pay for the run stack.
"""

import argparse

from tabascal.scripts._labels import removed as _removed

#: Mirrors ``tabascal.write.RFI_PER_SAT_PREFIX``, spelled out rather than
#: imported: that module pulls in JAX, and building this parser must not.
DEFAULT_PREFIX = "TAB_RFI_"


def build_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Write each satellite's RFI visibility prediction to "
            "its own MS column, from a tabascal run."
        )

    parser.add_argument(
        "run_dir",
        help="The run's directory, <out_dir>/runs/<run>; <out_dir>/runs/latest "
        "names the most recently started one. Its results must hold "
        "'rfi_vis_src', which a run writes when data.save_rfi_per_sat is true.",
    )
    parser.add_argument(
        "-m", "--ms_path", default=None,
        help="Use this Measurement Set rather than the one the run recorded, "
        "e.g. after moving it. Warns when it differs.",
    )
    parser.add_argument(
        "-z", "--results_zarr_path", action=_removed("give the run directory instead: "
                              "tabascal rfi-per-sat <out_dir>/runs/<run>"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-p", "--prefix", default=DEFAULT_PREFIX,
        help=f"Column name prefix; the NORAD id is appended. Default "
        f"'{DEFAULT_PREFIX}', giving {DEFAULT_PREFIX}58126.",
    )
    parser.add_argument(
        "-c", "--corr", default=None,
        help="Override the correlation the results were fitted on, e.g. 'xx'. "
        "Not normally needed: the results record it.",
    )

    return parser


def run(args):
    # Imported here rather than at module level so building the parser -- which
    # the top-level `tabascal -h` does -- costs nothing.
    from tabascal.run_dir import check_inputs, exported_results, load_run, override
    from tabascal.write import write_per_sat_rfi_ms

    run_dir, config, record = load_run(args.run_dir)
    ms_path = override("data.ms_path", args.ms_path, config["data"]["ms_path"])
    check_inputs(record, {"data.ms_path": ms_path})

    write_per_sat_rfi_ms(
        ms_path=ms_path,
        results_zarr_path=exported_results(run_dir, config),
        prefix=args.prefix,
        corr=args.corr,
    )


def main():
    # `tabascal rfi-per-sat` and this command are the same tool under two
    # names, so they have to claim the device the same way; see
    # scripts/_device_memory.
    from tabascal.scripts._device_memory import default_memory_on_demand

    default_memory_on_demand()

    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
