# Custom kernel
## Compilation

The CPU and GPU shared libraries are built by CMake via scikit-build-core as
part of installing tabascal (`pip install -e .`, or `pixi install`). See the
top-level `CMakeLists.txt` and the `[tool.scikit-build]` section of
`pyproject.toml`.

To rebuild after editing the kernels:
```
pixi run -e dev build-ffi                   # CPU (libtabascal.so)
pixi run -e cuda12-compile build-ffi-cuda   # NVIDIA GPU (libtabascal_cuda.so)
```

GPU builds are enabled by the `TABASCAL_CUDA=1` environment variable (set by the
`build-ffi-cuda` task); `nvcc` must be available on the `PATH`. The targeted GPU
architectures are configured via `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

## Usage

The configuration file should include the following in the components section:
```
- rfi_vis: RiemannVisTimeFreqCalculationFFI
```
