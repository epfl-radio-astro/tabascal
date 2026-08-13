"""Select the N satellites with the highest passes during an observation.

``tabsim``'s own ``max_n_sat`` truncation is ``np.unique(ids)[:max_n_sat]``, i.e. the
N *lowest NORAD IDs* among the visible satellites -- effectively "oldest launched"
rather than "most relevant". For an RFI benchmark we want the satellites that
actually dominate the observation, so this script ranks the visible passes by peak
elevation and writes out the top N NORAD IDs to paste into the sim config's
``norad_ids``.

Only satellites that also carry an entry in the ``norad_spec_model`` spectral model
file are eligible: tabsim silently drops the rest when building RFI sources, so
selecting one without a spectrum would quietly shrink the source count.

Usage
-----
    python select_top_satellites.py --sim-config sim_128A_zenith.yaml -n 32
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from tabsim.config import MJD0, load_config
from tabsim.tle import (
    check_satellite_visibilibities,
    get_tles_by_name,
    load_spacetrack_credentials,
)


def observation_times(sim_config: dict, step_minutes: float) -> Time:
    """Reconstruct the observation time grid from the sim config.

    Mirrors ``tabsim.config``'s start-time resolution: ``start_time_lha`` is in
    *degrees* of local hour angle, and the epoch is pinned to ``MJD0`` (the JD at
    which GMSA = 0), so the RA and LHA together choose the time of day.
    """
    obs = sim_config["observation"]
    tel = sim_config["telescope"]

    gsa = obs["start_time_lha"] - tel["longitude"] + obs["ra"]
    start_mjd = MJD0 + gsa / 360

    duration_days = obs["n_time"] * obs["int_time"] / 86400
    step_days = step_minutes / (24 * 60)
    return Time(
        np.arange(start_mjd, start_mjd + duration_days + step_days, step_days),
        format="mjd",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-config", required=True, help="tabsim simulation config")
    parser.add_argument("-n", "--n-sat", type=int, default=32)
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Satellite name search terms (default: the config's sat_names).",
    )
    parser.add_argument("--step-minutes", type=float, default=1.0)
    parser.add_argument("--tle-dir", default=None)
    parser.add_argument("-o", "--output", default="selected_norad_ids.txt")
    args = parser.parse_args()

    sim_config = load_config(args.sim_config, config_type="sim")
    sat = sim_config["rfi_sources"]["tle_satellite"]
    obs = sim_config["observation"]
    tel = sim_config["telescope"]

    names = args.names if args.names else sat["sat_names"]
    times = observation_times(sim_config, args.step_minutes)
    print(f"Observation window: {times[0].isot} -> {times[-1].isot} UT1")
    print(f"Searching for satellites matching {names}")

    username, password = load_spacetrack_credentials(args.tle_dir)
    if username is None:
        raise SystemExit(
            "No Space-Track credentials found. Place spacetrack_login.yaml in the "
            "TLE dir, ~/.credentials/, or the working directory."
        )

    tles = get_tles_by_name(
        username, password, names, float(np.mean(times.jd)), tle_dir=args.tle_dir
    )
    print(f"Retrieved {len(tles)} TLEs")
    if len(tles) == 0:
        raise SystemExit("No TLEs retrieved for the requested names.")

    # Search the whole visible sky: rank on elevation rather than pre-filtering, so
    # "highest pass" means highest, not "highest among an arbitrary cut".
    windows = check_satellite_visibilibities(
        tles["NORAD_CAT_ID"].values,
        tles["TLE_LINE1"].values,
        tles["TLE_LINE2"].values,
        times,
        tel["latitude"],
        tel["longitude"],
        tel["elevation"],
        obs["ra"],
        obs["dec"],
        max_ang_sep=180.0,
        min_elev=0.0,
    )
    if len(windows) == 0:
        raise SystemExit("No satellites were above the horizon during the window.")

    # One satellite can have several passes; rank each satellite by its best pass.
    per_sat = (
        windows.groupby("norad_id")
        .agg(
            max_elevation=("max_elevation", "max"),
            min_ang_sep=("min_ang_sep", "min"),
            visible_period=("visible_period", "max"),
        )
        .reset_index()
    )

    spec = pd.read_csv(sat["norad_spec_model"])
    have_spectrum = set(spec["norad_id"].unique())
    eligible = per_sat[per_sat["norad_id"].isin(have_spectrum)]
    print(
        f"{len(per_sat)} satellites above the horizon; "
        f"{len(eligible)} of them have a spectral model"
    )

    top = eligible.sort_values("max_elevation", ascending=False).head(args.n_sat)
    if len(top) < args.n_sat:
        print(
            f"WARNING: only {len(top)} satellites available, fewer than the "
            f"{args.n_sat} requested."
        )

    print(f"\nTop {len(top)} by peak elevation:")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    ids = sorted(int(i) for i in top["norad_id"])
    Path(args.output).write_text("\n".join(str(i) for i in ids) + "\n")
    print(f"\nWrote {len(ids)} NORAD IDs to {args.output}")
    print(f"norad_ids: [{', '.join(str(i) for i in ids)}]")


if __name__ == "__main__":
    main()
