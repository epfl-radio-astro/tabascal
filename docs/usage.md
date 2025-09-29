# Usage Guide

## Installation

TABASCAL interacts with Measurement Sets and therefore depends on `python-casacore`. As such, on Mac OS we highly recommend using conda environments to install `python-casacore` first. 

### Create conda environment with `python-casacore` installed

```bash
conda create -n tab-env -c conda-forge python=3.11 python-casacore
conda activate tab-env
```

Currently TABASCAL is not on PyPI so you need to clone the repository:

```bash
git clone https://github.com/epfl-radio-astro/tabascal.git
```

or if you have repository access

```bash
git clone git@github.com:epfl-radio-astro/tabascal.git
```

### Install via pip (CPU-only):

From here you can install TABASCAL with pip using either

```bash
pip install -e ./tabascal/
```

### Or with GPU support:

```bash
pip install -e ./tabascal/[gpu]
```

## Space-Track Login

To get historical orbital elements to predict the positions of satellites in an observation, a [Space-Track](https://www.space-track.org/auth/login) account is needed. Currently, Space-Track login details are required both for simulating a dataset and for applying TABASCAL to a dataset.

Your login details should then be saved in a YAML file that is formatted as follows:

```yaml
username: user@email.com
password: password1234
```

The rest of the example below assumes you have a Space-Track account with your login details saved in a file named `spacetrack_login.yaml`.

# Example Simulation and RFI Subtraction

Assuming you have cloned the repository, navigate to the `tabascal/examples` directory in the root of the repository. It contains 

``` 
tabascal/
    ├── examples/
|       └── ex_spacetrack_login.yaml    # Space-Track login details template
|       └── sim_target_8A.yaml          # Simulation configuration file
|       └── tab_target.yaml             # TABASCAL configuration file
```

## Running Simulations

Simulations are defined by YAML config files and can be launched using:

```bash
sim-vis -c sim_target_8A.yaml -st spacetrack_login.yaml
```

The output of this command will show you a number of simulation details and finally end with some lines that looks like

```text
Mean RFI Amp.  : 8.48 Jy
Mean AST Amp.  : 1.56 Jy
Vis Noise Amp. : 0.66 Jy
Flag Rate      : 79.3 %

Total simulation time : 0:00:15.483440

2025-09-25 07:57:01.423957
(<tabsim.dask.observation.Observation object at 0x1400cac90>, 'data/pnt_src_obs_08A_120T-0000-0238_1025I_001F-1.227e+09-1.227e+09_050PAST_000GAST_000EAST_3SAT_0GRD_1.0e+00RFI')
```

The path printed at the end, `data/pnt_src_obs_08A_120T-0000-0238_1025I_001F-1.227e+09-1.227e+09_050PAST_000GAST_000EAST_3SAT_0GRD_1.0e+00RFI` is the path to the simulation directory which contains the simulated dataset and many other simulation details. The structure of this directory and its contents are described in the [tab-sim documentation](https://tab-sim.readthedocs.io/en/latest/output.html).

`sim-vis` has a help prompt wich can be accessed with

```bash
sim-vis -h
```

## Subtracting Satellite-based RFI

RFI subtraction (TABASCAL) runs are also defined by YAML configuration files and can be run in much the same way. Given the simulation dataset created in the previous step, we can run TABASCAL on it using

```bash
tabascal -c tab_target.yaml -s data/pnt_src_obs_08A_120T-0000-0238_1025I_001F-1.227e+09-1.227e+09_050PAST_000GAST_000EAST_3SAT_0GRD_1.0e+00RFI -st spacetrack_login.yaml
```

The output of a successful run with TABASCAL will show lines like

```text
Copying tabascal results to MS file from data/pnt_src_obs_08A_120T-0000-0238_1025I_001F-1.227e+09-1.227e+09_050PAST_000GAST_000EAST_3SAT_0GRD_1.0e+00RFI/results/map_pred_Custom.zarr
Writing tabascal results to ['TAB_AST_DATA', 'TAB_RFI_DATA', 'TAB_AST_RES', 'TAB_RFI_RES', 'TAB_RES_DATA'] columns in MS file.
Data type: 24, SORT_COLUMNSnot handled
Data type: 24, SORT_ORDERnot handled
```

The results of the TABASCAL run are saved in a `.zarr` file and then transferred into the Measurement Set.

If you have a Measurement Set from another source you can run TABASCAL on that directly with

```bash
tabascal -c path/to/config.yaml -ms path/to/ms/file.ms -st spacetrack_login.yaml
```

The `tabascal` script also has a help context whcih can be accessed with

```bash
tabascal -h
```