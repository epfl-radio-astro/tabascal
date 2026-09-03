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
think about. The client itself is the
[satchecker-client](https://satchecker-client.readthedocs.io/) package; how
TABASCAL uses it — source precedence, record age limits, and coverage
enforcement — is documented in [Satellite orbit records](orbits.md).

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
them, see [Satellite orbit records](orbits.md). `tabascal light-curve
--fit-offset --write-shifted-tle DIR` writes records of the same kind with a
fitted along-track offset folded into their epochs, consumed the same way
through `--extra-orbit-dir` — see
[Records with a fitted time offset](orbits.md#records-with-a-fitted-time-offset).

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
Writing tabascal results to ['CORRECTED_DATA', 'TAB_AST_DATA', 'TAB_RFI_DATA', 'TAB_AST_RES', 'TAB_RFI_RES', 'TAB_RES_DATA', 'WEIGHT_SPECTRUM', 'WEIGHT'] columns in MS file.
Data type: 24, SORT_COLUMNSnot handled
Data type: 24, SORT_ORDERnot handled
```

The results of the TABASCAL run are saved in a `.zarr` file and then transferred into the Measurement Set.

If you have a Measurement Set from another source you can run TABASCAL on that directly with

```bash
tabascal run -c path/to/config.yaml -ms path/to/ms/file.ms
```

## Extracting RFI light curves

`tabascal light-curve` measures each satellite's apparent flux over time and
frequency directly from the visibilities, by matched-filtering them against the
known satellite trajectory phase. No imaging is involved. It is the same
estimate `rfi.init: matched-filter` makes inside a run — see
[Estimating the light curves from the data](config.md#estimating-the-light-curves-from-the-data)
— written out in the `rfi.est` interchange format, so it can seed a later run
unchanged.

Given a config, the satellites, the data column, the correlation and the
elevation cut all come from it, and the Measurement Set is read once. Any of
`-dc`, `-cr` and `--min-elevation` overrides the config for that one value; give
none of them and the config decides:

```bash
tabascal light-curve -c tab_target.yaml -ms path/to/ms/file.ms
```

A satellite in the config that never rises above the cut is not an error here,
as it is for a run: this command measures rather than fits, a satellite that
never rose has a zero curve, and stopping would leave `--no-elevation-cut` —
which drops the cut for *every* satellite — as the only way to measure the ones
that were up. The command names it and carries on.

For an observation TABASCAL has not been configured against, name the
satellites yourself:

```bash
tabascal light-curve -ms path/to/ms/file.ms -n 27868,57865,60093 -dc DATA
```

The output goes to `<ms_dir>/light_curves/<tag or column>.npz` unless `-o` says
otherwise. `light_curves` is the magnitude `|S_hat|`, which is what the format
requires — a complex array there is rejected on read rather than truncated to
its real part. `times` is written as **UTC** MJD, whatever scale the MS declares
in its `TIME` column: the format states one scale so the curves stay
interpretable away from the MS they were measured on, and a run seeding from
them samples on the same one. That scale is stamped into the file as
`time_scale`, so nothing downstream has to assume it — and a file written before
the stamp existed, which may have been on a declared scale, is read as UTC with
a warning rather than silently. Alongside the four names the format requires, the
output carries the noise floor (`error`), the significance
`z = Re(S_hat) / error`, the native complex estimate (`light_curves_complex`)
and the in-view mask; readers of the format ignore the extras. `-p` also writes
a per-source spectrogram of `z`.

To score a run, filter its *residual* rather than a data column. Point `-z` at
the run's results zarr and `-dc` at the reference column the residual is formed
against: the residual is then `data_col - zarr.vis_obs`, which cannot be
invalidated by a later run overwriting the MS's `TAB_*` columns.

```bash
tabascal light-curve -c tab_target.yaml -z path/to/results/map_pred_Custom.zarr -dc DATA
```

The store is checked against the visibilities before anything is subtracted —
its baseline and timestep counts, the cadence of its time axis, and the
correlation it was fitted on — so a results zarr from another run or another
correlation is refused rather than differenced. Frequencies are matched
channel by channel; the counts and cadence are what catch a store that lines up
by accident.

A fully subtracted satellite has `|z| <= 3` almost everywhere. Judge that
against the `null` column the command prints, not against the analytic 99.73%:
the noise floor assumes the de-rotated per-baseline samples are independent and
residual sky is not, so the floor is optimistic. The null is the same statistic
on `Im(S_hat)`, which after de-rotation carries the same noise and no source.

**`cov`, `null` and `excess` assume the column they scored is phase
calibrated.** They read `Re(S_hat)`, which is the whole of a de-rotated real
source only once the antenna gain phases are out of the data. The command names
the column it scored in the heading for that reason.

`|S|` is the same statistic on `|S_hat|/error`, against a Rayleigh threshold
enclosing the same probability (3.44 for 3 sigma). It survives a phase **common
to every baseline** — an overall offset, or a stable phase on the source itself —
which would otherwise empty `Re(S_hat)` and push the source into the imaginary
null that `cov` is judged against, moving both halves of that comparison the
wrong way.

**Neither survives an uncalibrated antenna gain.** A gain multiplies each
baseline *before* the average, `S_hat = S · Σ w gₚ gq* / Σ w`, so
antenna-dependent phases decorrelate the coherent sum itself: the estimate
shrinks, and the magnitude shrinks with it. On a raw column both numbers
understate what is there — they are a lower bound on the residual, not a
detection threshold. The optimistic-floor caveat applies to both.

The floor comes from the MS's own noise column, so an MS carrying none — and no
`data.noise` to supply one — has no floor to quote. The light curves are still
measured and written, but `error` and `z` are NaN and the coverage table is
replaced by a line saying so: `1/sqrt(N_bl)` would be quoting a noise of 1 Jy
that nobody stated, and a z built on it would look like a detection at any flux.

An MS with no usable noise column is not an error here, unlike a `tabascal run`
that has to weight a likelihood by it: the curves are still measured, and come
back unweighted with NaN errors and no coverage table, as above.

### Fitting the along-track time offset

A TLE's dominant error is along-track — kilometres to tens of kilometres of drag
mismodelling and unannounced manoeuvres — and along the track an error is very
nearly a pure *time offset*. `--fit-offset` measures it: for each satellite it
scans `tau`, evaluating the orbit at `t + tau`, builds the near-field fringe
model on a fine grid inside each integration, and coherently correlates it
against the data over the baselines the orbit is accurate enough to steer. Frames
are combined into a per-channel score `z²`, the best cell over offset and channel
is the answer, and its significance is measured against a null in which every
antenna's path is scrambled by tens of metres — an empirical null, on these data,
with their own weights, flagging and residual sky.

```bash
tabascal light-curve -ms path/to/ms/file.ms -n 46344 --fit-offset --only-detections --write-shifted-tle path/to/shifted_orbits
```

One line is printed per satellite — the best `tau`, the best channel, `z²`, the
null's mean and spread, the significance, and `DETECTED` or `not detected`
against `--threshold` (default 5 sigma). **That threshold carries no trials
factor**: the scan maximises over the whole grid and every channel while the null
is drawn at the best offset only, so it is a working cut calibrated on the MWA
Cen A case rather than a false-alarm rate. The grid is `--tau-max` ±4 s in
`--tau-step` 0.25 s steps by default — the step times the integers out to the
half-width, so `tau = 0` is always on it and a half-width that is not a whole
number of steps is rounded down rather than overshot. A grid of more than a million points is
refused with a message naming `--tau-step`: the peak is a fraction of a second
wide, so a step that fine buys nothing, and the remedy is a coarser step. The step has to resolve
the peak, which narrows as the coherent array grows, so a longer array needs a
finer step rather than a wider grid.

The scan runs in single precision by default — a fringe model on a path
difference of a few kilometres needs no more, and with `-ms` there is no config
to ask — which `--precision {single,double}` overrides; with `-c` the config's
`model.precision` decides unless the flag is given.

The curves are then extracted at the offset that was measured, not at `tau = 0`,
and the fit travels with them into the `.npz`: `tau_best`, `tau_grid`, `z2_tau`,
`z2_best`, `best_chan`, `significance`, `null_mean`, `null_std`, `detected` and
the frame-by-channel `r_best` spectrogram. Recording `tau_best` is the point —
without it a later run cannot reproduce the trajectory the curves were measured
on. `--only-detections` drops the satellites that did not clear the threshold
from the saved curves, though every fit is still reported: a curve extracted at
an offset that is not a detection is a curve extracted at noise.
`--write-shifted-tle DIR` writes the **detected** satellites' orbit records with
their epochs moved by `-tau` into `DIR`, which a later run picks up with
`--extra-orbit-dir` and so reproduces the measured trajectory with no further
configuration. With `-p`, each saved satellite also gets a
`<output>_offset_<norad>.png`: the `|r|` spectrogram with the elevation curve
over it, the per-channel `z²` against the null band, and the scan curve itself.

To read one channel instead of the whole band, pass `-f` with a frequency in Hz.
The nearest channel is used, and the request must land inside it — a frequency
more than half a channel outside the band is an error naming the band, rather
than a silent read of the nearest edge channel. With `-z` the model is matched
to the channels read by frequency, not by position, each within half of *its
own* width, so a non-uniform spectral window is matched channel by channel.

## Searching for the contaminating satellite

Everything above assumes you know which satellite is in the data. `tabascal
search` is for when you do not: given the visibilities and a TLE snapshot — a
constellation export in a directory, or an explicit list of NORAD IDs — it
produces the `satellites.norad_ids` list itself, with the along-track offset of
each satellite it names, from the data and nothing else.

The pipeline is four steps, each auditable on its own:

1. **Enumerate.** Every record in the snapshot is propagated over the
   observation and the ones that never came above `--min-elevation` are dropped.
   That is also what keeps the geometry honest: the coherence ceiling grows with
   slant range, so a satellite on the far side of the Earth would tolerate
   kilometres of baseline and let every long one back into the sum.
2. **Score.** The along-track scan of the previous section is run for every
   surviving candidate — the same statistic, `vmap`ped over a candidate axis and
   jitted once for the whole sweep, so a snapshot compiles a single program and
   then runs device-resident. Each candidate's horizon mask and coherence cut are
   applied *inside* the statistic as weights, because the batched shapes cannot
   be sliced: a satellite that sets mid-observation contributes only its own
   frames, and a nearer one is summed over only the baselines it can steer.
3. **Rank and calibrate.** Candidates are ranked by the best `z²` over the offset
   grid and the channels, and the top `--null-top` (5 by default) are measured
   against the decohered null. The null is a whole scan's work per satellite;
   over a constellation, drawing one for every candidate is the search twice
   over, spent on satellites nothing will be reported for.
4. **Select and emit.** Whatever clears `--threshold` is named, and the config
   fragment, the light curves and the shifted orbit records are written for it.

```bash
tabascal search -ms path/to/ms/file.ms --tle-dir path/to/tle_snapshot --name-filter STARLINK -p
```

That is how the MWA Cen A observation was identified: 2551 Starlink records, 128
of them above the horizon during the 56 s observation, and STARLINK-1765 / 46344
alone at `z² = 0.0995` against a runner-up of 0.0523 and a candidate median of
0.0446 — with its best channel at 175.0 MHz, the known UEMR downlink band.
What *names* a satellite is its significance against the decohered null, measured
against `--threshold`. The winner's separation from the candidate field — its
`z²` beside the median, and the runner-up warning — is the diagnostic to read
alongside it: `z²` is not calibrated between datasets, so its absolute value says
little on its own, but a winner that does not stand clear of the field is one to
look at twice however significant it is.

The batch is sized against a memory budget as well as against `--batch-size`. It
has to be: once candidates come near the horizon the shared coherent set grows
with them, and on the MWA case it reaches 7704 of the array's 9180 baselines
(`b_coh_max` 2880 m), which puts one candidate over 24 channels at some 2.1 GB —
so a default batch of eight would ask for 17 GB. `--max-mem-gb` (4.0 by default)
caps that; the run prints the batch it actually used. It counts the fringe model
and the path differences, not the weights beside them, so it is a sizing
heuristic rather than a guarantee: halve it before blaming the flag.

Everything is written under `-o`'s stem, which defaults to
`<ms_dir>/sat_search/<data column>`:

| file | when |
| --- | --- |
| `<stem>_ranking.npz` | always — the ranking table, every candidate |
| `<stem>_config.yaml` | always — the `satellites` section, ready to merge |
| `<stem>_light_curves.npz` | for the satellites that were named |
| `<stem>_shifted_tles/` | for the satellites that were named |
| `<stem>_ranking.png`, `<stem>_offset_<norad>.png` | with `-p` |

**Saving is threshold-gated by default.** A search meets hundreds of candidates,
and a curve extracted at an offset that is not a detection is a curve extracted
at noise, so only the detections get curves and plots; `--save-all` opens it for
the run where the negatives are the point. The ranking table is written either
way — it is the evidence for a negative. The config fragment names the shifted
records with no age ceiling, so `tabascal run --extra-orbit-dir` reproduces the
trajectories the search measured, whatever SatChecker serves by then.

**The exit status is part of the answer.** `0` when at least one satellite
cleared the threshold, `3` when the scan ran and nothing did — a meaningful,
scriptable result rather than a failure — and an error when nothing in the
snapshot was above the horizon at all, which is a statement about the snapshot
rather than about the data.

Two warnings are printed where the ranking cannot be read at face value. A
**close runner-up** — a second candidate scoring within `--runner-up-ratio`
(1.5) of the winner — is usually two satellites of the same train partially
matching each other's fringes, and is a result to look at twice rather than a
satellite to name. **`tau` at the scan edge** means a named satellite's best
offset is the first or last point of the grid, so the peak may lie past it and
the offset reported is a floor; widen `--tau-max` and run it again.

The scan flags are the light-curve command's, with the same meanings and the same
refusals, except that `--tau-step` defaults to the coarser 0.5 s because the scan
runs once per candidate — halve it once the field is narrowed. There is no
`--fit-offset` here: the scan *is* the search, and at `tau = 0` the MWA case
scored 0.045 against a candidate median of 0.0446, which is no detection at all.

The `tabascal` script also has a help context which can be accessed with

```bash
tabascal -h                # top-level: lists the subcommands
tabascal run -h            # every option of the run subcommand
tabascal light-curve -h    # every option of the light-curve subcommand
tabascal search -h         # every option of the search subcommand
```