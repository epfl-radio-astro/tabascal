#!/usr/bin/env python
from tabascal.write import write_results_ms

import argparse

def main():

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
        "-gt",
        "--gain_table",
        default=None,
        help="The CASA calibration table the run was fit with (data.gain_table). Required "
        "if one was used: data_col in the MS is raw, so without it the models are "
        "subtracted in the wrong frame.",
    )

    args = parser.parse_args()
    write_results_ms(
        ms_path=args.ms_path,
        results_zarr_path=args.results_zarr_path,
        data_col=args.data_col,
        gain_table=args.gain_table,
    )


if __name__ == "__main__":
    main()
