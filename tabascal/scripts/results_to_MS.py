#!/usr/bin/env python
"""``tab2MS``: a tabascal run's results into the MS it was fitted from.

Takes the run directory and reads the rest from what the run recorded there:
the Measurement Set, its data column, the gain tables, and which results to
export -- ``map_pred`` if the run optimised, ``init_pred`` if it did not. The
calibration table goes into the run's own ``results/``.

The parser is built separately from the run, as in the other export scripts, so
the argument surface can be tested without importing ``tabascal.write`` -- and
with it the whole JAX stack -- to do it.
"""

import argparse

from tabascal.scripts._labels import removed as _removed

_NEW_COMMAND = "give the run directory instead: tab2MS <out_dir>/runs/<run>"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Write a tabascal run's recovered visibilities into the "
        "Measurement Set it was fitted from."
    )

    parser.add_argument(
        "run_dir",
        help="The run's directory, <out_dir>/runs/<run>; <out_dir>/runs/latest "
        "names the most recently started one.",
    )
    parser.add_argument(
        "-m", "--ms_path", default=None,
        help="Use this Measurement Set rather than the one the run recorded, "
        "e.g. after moving it. Warns when it differs.",
    )
    parser.add_argument(
        "-c",
        "--corr",
        default=None,
        help=(
            "Override the correlation the results were fitted on, e.g. 'xx'. "
            "Not normally needed: the results record it."
        ),
    )
    parser.add_argument(
        "-gt",
        "--gain_table",
        action="append",
        default=None,
        help=(
            "Use these CASA calibration tables rather than the data.gain_table "
            "the run recorded, e.g. after moving them. Repeat the flag, or give "
            "a comma-separated list, in the same order as the config, since the "
            "tables compose in order. Warns when they differ."
        ),
    )
    for flags in (("-z", "--results_zarr_path"), ("-d", "--data_col"), ("-o", "--caltable-path")):
        parser.add_argument(*flags, action=_removed(_NEW_COMMAND), help=argparse.SUPPRESS)

    return parser


def run(args):
    # Imported here rather than at module level so building the parser costs
    # nothing; tabascal.write pulls in the whole JAX stack.
    from tabascal.run_dir import check_inputs, exported_results, load_run, override
    from tabascal.write import write_results_ms

    run_dir, config, record = load_run(args.run_dir)
    data = config["data"]

    ms_path = override("data.ms_path", args.ms_path, data["ms_path"])
    # A list, as the run records it, so -gt and the record line up by position.
    recorded = data.get("gain_table")
    recorded = [recorded] if isinstance(recorded, str) else (recorded or None)
    gain_table = override("data.gain_table", _gain_tables(args.gain_table), recorded)
    overrides = {"data.ms_path": ms_path}
    overrides.update(
        {f"data.gain_table[{i}]": path for i, path in enumerate(gain_table or [])}
    )
    check_inputs(record, overrides)

    results = exported_results(run_dir, config)
    print(f"Copying tabascal results to MS file from {results}")
    write_results_ms(
        ms_path=ms_path,
        results_zarr_path=results,
        data_col=data["data_col"],
        corr=args.corr,
        gain_table=gain_table,
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
    # A standalone entry point, so nothing above has done this: copying a
    # results zarr into an MS pulls in tabascal.write and with it JAX, and
    # taking three quarters of the device to do it is not on. See
    # scripts/_device_memory.
    from tabascal.scripts._device_memory import default_memory_on_demand

    default_memory_on_demand()

    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
