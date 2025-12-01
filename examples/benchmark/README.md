# Benchmarking Configs for SKA-Low Problem Size (up to 512 antennas)

For this benchmarking we are not interested in getting the correct results but rather the speed of the optimisation loop. As such, the simulation config provided will simulate 32 satellites for a 15 minute observation with 10 second integration time per time sample. For speed of simulation, only 1 integration sample is used.

## Simulate a 32 satellite observation with `N` number of antennas

```bash
sim-vis -c sim_target_ska-low.yaml -st /path/to/spacetrack_login.yaml -a N
```

The number of antennas used affects the dataset name. The naming convention is `pnt_src_obs_0NA_*`.


## Run tabascal on a simulation with 8 antennas

All 32 satellites are included in the `tab_target.yaml` file. To set the number of satellites included in the model fit, comment out the appropriate number of NORAD IDs under the `satellites: norad_ids:` section.  

To set the number of integration samples used in the reduction on the time axis, adapt the `rfi: n_int_time:` section. 

```bash
tabascal -c tab_target.yaml -s data/pnt_src_obs_08A_090T-0000-0890_001I_001F-1.500e+08-1.500e+08_050PAST_000GAST_000EAST_32SAT_0GRD_1.0e+00RFI/ -st /path/to/spacetrack_login.yaml
```
