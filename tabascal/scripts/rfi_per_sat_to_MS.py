#!/usr/bin/env python
"""Export per-satellite RFI visibility predictions from a tabascal run to MS columns.

Reads ``rfi_vis_src`` from a results zarr (produced when tabascal is run with
``data.save_rfi_per_sat: true``) and writes one MS column per satellite, named
``<prefix><NORAD_ID>`` (default prefix ``TAB_RFI_``). Standalone and re-runnable:
needs only the zarr and the MS, no re-fit. Image a column with ``nufft-gif`` to
inspect a single satellite's modelled RFI for astronomical-signal contamination.
"""
from tabascal.write import write_per_sat_rfi_ms

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Write each satellite's RFI visibility prediction to its own MS "
        "column (TAB_RFI_<NORAD_ID>) from a tabascal results zarr."
    )
    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "-z",
        "--results_zarr_path",
        required=True,
        help="File path to the map_pred results zarr (must contain 'rfi_vis_src').",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="TAB_RFI_",
        help="Column name prefix; the NORAD id is appended. Default 'TAB_RFI_'.",
    )

    args = parser.parse_args()
    write_per_sat_rfi_ms(
        ms_path=args.ms_path,
        results_zarr_path=args.results_zarr_path,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
