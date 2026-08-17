# Usage Guide

## Installation

TABASCAL is pure Python — no compiler or CUDA toolkit is needed to install it —
and requires **Python 3.10–3.13**. Python 3.14 has not been validated against
the pinned jax/jaxlib versions.

### Create a conda environment with `python-casacore` installed

TABASCAL interacts with Measurement Sets and therefore depends on
`python-casacore`, which is *not* a pip dependency of TABASCAL. Install it with
conda first:

```bash
conda create -n tab-env -c conda-forge "python>=3.10,<3.14" python-casacore
conda activate tab-env
```

`python-casacore` is pip-installable on **linux-x86_64**, so this step can be
skipped there. On **macOS and linux-aarch64** the conda route is strongly
recommended, as `python-casacore` is difficult to build from source on those
platforms.

### Install via pip (CPU-only)

TABASCAL is not on PyPI yet, so install it from the repository:

```bash
pip install git+https://github.com/epfl-radio-astro/tabascal.git
```

### Or with GPU support

For NVIDIA GPUs on Linux, use the `cuda12` extra (or `cuda13` for CUDA 13):

```bash
pip install "tabascal[cuda12] @ git+https://github.com/epfl-radio-astro/tabascal.git"
```

The GPU extras also pull in the compiled GPU kernel used by the FFI
`rfi_vis` components — see [RFI-visibility kernels](kernels.md).

### Installing from a clone

To work from a checkout instead, clone the repository and install it in place.
This is also what you want for the example below, which uses the config files
that ship in `examples/`:

```bash
git clone https://github.com/epfl-radio-astro/tabascal.git
pip install -e ./tabascal/
```

or if you have repository access

```bash
git clone git@github.com:epfl-radio-astro/tabascal.git
```

If you intend to develop TABASCAL, use the pixi-based setup described in
[Developer install](installation.md) instead.

## Satellite orbital elements

TABASCAL retrieves the orbital elements needed to predict satellite positions
from the [IAU CPS SatChecker](https://satchecker.cps.iau.org/) service. **No
account or credentials are required** — records are fetched automatically for
the requested NORAD IDs and cached locally for reuse.

SatChecker serves two formats: TLEs for epochs up to 2026-07-11, and OMM
(Orbit Mean-Elements Message) records from 2026-07-12 onwards. TABASCAL asks
whichever archive your observation epoch falls in and falls back to the other if
that one has nothing usable, so this is not something you configure or need to
think about.

Every configured satellite must resolve to an acceptable record. TABASCAL checks
this during preflight — before the visibilities are read — and stops with an
error naming each failing satellite rather than quietly subtracting an
incomplete RFI model. The remedies are to supply the missing records via
`--extra-orbit-dir`, to change `satellites.remote_max_age_days` deliberately,
or to remove the satellite from `satellites.norad_ids`.

Every run also saves the records it actually used to
`<sim_dir>/results/used_orbits_<name>.json`; passing that file's directory back
via `--extra-orbit-dir` reproduces the run's trajectory priors exactly. For the
two archives and the handover between them, the full caching behaviour, the age
policies, what validation each format does and does not give you, and how to
supply records manually (e.g. from Space-Track) when SatChecker cannot provide
them, see [Satellite orbit records](orbits.md).

Note: generating a simulation with `sim-vis` (part of tab-sim) still uses
Space-Track and requires a `spacetrack_login.yaml`. That requirement applies only
to the simulation step below, not to running TABASCAL.

# Example Simulation and RFI Subtraction

Assuming you have cloned the repository, navigate to the `tabascal/examples` directory in the root of the repository. It contains 

``` 
tabascal/
    ├── examples/
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
tabascal run -c tab_target.yaml -s data/pnt_src_obs_08A_120T-0000-0238_1025I_001F-1.227e+09-1.227e+09_050PAST_000GAST_000EAST_3SAT_0GRD_1.0e+00RFI
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
tabascal run -c path/to/config.yaml -ms path/to/ms/file.ms
```

## Checking a configuration file

The configuration file is validated at the start of every run, before the
Measurement Set is read or any orbital records are fetched, and every problem in
it is reported at once. You can run just that check, without running anything
else, with

```bash
tabascal check-config -c tab_target.yaml
```

It prints the model it resolved and the configuration TABASCAL would actually
use, with every default filled in — which is the quickest way to see what a
parameter you did not set is going to be. If the file is not usable it prints the
same report the run would have, and exits non-zero:

```text
Error: invalid configuration in tab_target.yaml: 2 problems found

  gains.corr_time: unknown key (did you mean 'gains.amp_corr_time'?)
  opt.max_iter: expected an integer >= 0, got 'many'
```

Which parameters exist is determined by the components in `model.components`:
each one declares what it reads, so a key that belongs to a component you have
not selected is reported as such rather than silently ignored. See
[the configuration file](config.md).

The `tabascal` script also has a help context which can be accessed with

```bash
tabascal -h        # top-level: lists the subcommands
tabascal run -h    # every option of the run subcommand
```