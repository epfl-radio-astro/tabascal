"""Calibrate ``power_scale`` so a satellite crossing the main beam is ~1000 Jy.

tabsim builds satellite RFI as::

    I          = Pv_to_Sv(Pv, distances)              # apparent flux in Jy
    rfi_A_app  = sqrt(|I|) * airy_beam(ang_sep, ...)  # beam-attenuated amplitude

so "crossing the main beam" is exactly ``airy_beam -> 1``, and the on-axis apparent
flux is ``I``. ``Pv = power_scale * spectrum``, and ``Pv_to_Sv`` is linear in ``Pv``,
so ``I`` is linear in ``power_scale`` and a single evaluation fixes the scale
exactly -- no trial simulations needed.

This computes, for each selected satellite, its on-axis apparent flux at closest
approach, and reports the ``power_scale`` that puts the median of those at the
target flux. The median is used rather than the max so one unusually low pass does
not set the normalisation for the whole population; min/median/max are all printed
so the choice can be revisited.

Usage
-----
    python calibrate_power_scale.py --sim-config sim_128A_zenith.yaml \
        --ids selected_norad_ids.txt --target-jy 1000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from skyfield.api import load, wgs84

from tabsim.config import MJD0, generate_spectra, load_config
from tabsim.tle import get_tles_by_id, load_spacetrack_credentials

# Pv_to_Sv is Sv = Pv / (4 pi d^2) converted to Jy (1 Jy = 1e-26 W/m^2/Hz).
JY = 1e-26


def observation_times(sim_config: dict) -> Time:
    """The observation time grid, matching tabsim's start-time resolution."""
    obs, tel = sim_config["observation"], sim_config["telescope"]
    gsa = obs["start_time_lha"] - tel["longitude"] + obs["ra"]
    start_mjd = MJD0 + gsa / 360
    offsets = np.arange(obs["n_time"]) * obs["int_time"] / 86400
    return Time(start_mjd + offsets, format="mjd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--ids", default="selected_norad_ids.txt")
    parser.add_argument("--target-jy", type=float, default=1000.0)
    parser.add_argument("--tle-dir", default=None)
    args = parser.parse_args()

    sim_config = load_config(args.sim_config, config_type="sim")
    obs, tel = sim_config["observation"], sim_config["telescope"]
    sat = sim_config["rfi_sources"]["tle_satellite"]

    norad_ids = [int(i) for i in Path(args.ids).read_text().split()]
    times = observation_times(sim_config)
    freqs = obs["start_freq"] + obs["chan_width"] * np.arange(obs["n_freq"])

    username, password = load_spacetrack_credentials(args.tle_dir)
    tles = get_tles_by_id(
        username, password, norad_ids, float(np.mean(times.jd)), tle_dir=args.tle_dir
    )

    # Satellite and observer positions in the same frame; the array is a few km
    # across against a ~550 km range, so the array centre stands in for all antennas.
    from tabsim.tle import get_satellite_positions

    sat_xyz = get_satellite_positions(
        tles[["TLE_LINE1", "TLE_LINE2"]].values, times.jd
    )  # (n_sat, n_time, 3) in metres
    ts = load.timescale()
    obs_xyz = (
        wgs84.latlon(tel["latitude"], tel["longitude"], tel["elevation"])
        .at(ts.ut1_jd(times.jd))
        .position.m.T
    )  # (n_time, 3)

    distances = np.linalg.norm(sat_xyz - obs_xyz[None], axis=-1)  # (n_sat, n_time)

    spec = pd.read_csv(sat["norad_spec_model"])
    ids, spectra = generate_spectra(
        spec[spec["norad_id"].isin(tles["NORAD_CAT_ID"].values)], freqs, "norad_id"
    )
    spectra = np.asarray(spectra)

    current_scale = float(sat["power_scale"])
    rows = []
    for i, nid in enumerate(tles["NORAD_CAT_ID"].values):
        sel = ids == nid
        if not sel.any():
            continue
        # Band-centre spectral power for this satellite, at the current scale.
        pv = current_scale * spectra[sel].sum(axis=0)[obs["n_freq"] // 2]
        d_min = distances[i].min()
        flux = pv / (4 * np.pi * d_min**2) / JY  # Jy, on-axis
        rows.append(
            {
                "norad_id": int(nid),
                "min_range_km": d_min / 1e3,
                "on_axis_Jy": flux,
            }
        )

    df = pd.DataFrame(rows).sort_values("on_axis_Jy", ascending=False)
    print(f"\nOn-axis apparent flux at closest approach (power_scale={current_scale:g}):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    median = df["on_axis_Jy"].median()
    new_scale = current_scale * args.target_jy / median
    print(
        f"\nmin={df['on_axis_Jy'].min():.4g} Jy  "
        f"median={median:.4g} Jy  max={df['on_axis_Jy'].max():.4g} Jy"
    )
    print(f"\nTo put the median at {args.target_jy:g} Jy:")
    print(f"  power_scale: {new_scale:.4g}")
    print(
        f"  -> min {df['on_axis_Jy'].min() * new_scale / current_scale:.4g} Jy, "
        f"max {df['on_axis_Jy'].max() * new_scale / current_scale:.4g} Jy"
    )


if __name__ == "__main__":
    main()
