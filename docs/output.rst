Output Files
============

tab-sim outputs a simulation directory containing all input and output data.

Directory Structure
-------------------

.. code-block:: text

   sim_name/
   ├── sim_name.zarr/
   ├── sim_name.ms/
   ├── AngularSeps.png
   ├── SourceAltitude.png
   ├── UV.png
   ├── log_sim_*.txt
   └── input_data/
       ├── MeerKAT.itrf.txt
       ├── norad_ids.yaml
       ├── norad_satellite.rfimodel
       └── sim_config.yaml

Zarr Output (.zarr)
-------------------

Use `xarray` to open `.zarr` files:

.. code-block:: python

   import xarray as xr
   xds = xr.open_zarr("path/to/sim_name.zarr/")
   print(xds)

Includes:
- Coordinates: `ant`, `freq`, `time`, `uvw`, `radec`, etc.
- Data: `vis_obs`, `vis_rfi`, `vis_ast`, `noise_data`, `rfi_tle_sat_xyz`, etc.
- Attributes: observation metadata and simulation parameters

Measurement Set (.ms)
---------------------

Standard columns:
- `DATA`, `CORRECTED_DATA`, `MODEL_DATA`

Extended columns:
- `CAL_DATA`, `AST_MODEL_DATA`, `RFI_MODEL_DATA`, `NOISE_DATA`

Each contains different subsets or processed versions of the simulated data, useful for analysis and calibration testing.
