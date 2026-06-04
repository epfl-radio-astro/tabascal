# Custom kernel
## Compilation

The CPU shared library (`libtabascal.so`) is built by CMake via scikit-build-core
automatically as part of installing tabascal (`pip install -e .`, or
`pixi install`). The GPU library (`libtabascal_cuda.so`) is **not** built by a
plain install — it requires `TABASCAL_CUDA=1` (see below). See the top-level
`CMakeLists.txt` and the `[tool.scikit-build]` section of `pyproject.toml`.

To rebuild after editing the kernels:
```
pixi run -e dev build-ffi                   # CPU (libtabascal.so)
pixi run -e cuda12 build-ffi-cuda           # NVIDIA GPU (libtabascal_cuda.so)
```

GPU builds are enabled by the `TABASCAL_CUDA=1` environment variable (set by the
`build-ffi-cuda` task); `nvcc` must be available on the `PATH`. The targeted GPU
architectures are configured via `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

## Usage

The configuration file should include the following in the components section:
```
- rfi_vis: RiemannVisTimeFreqCalculationFFI
```
