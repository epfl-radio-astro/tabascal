# Custom kernel
## Compilation

Compilation requires a shell environment, where JAX is available from python.
For GPU support, the `nvcc` compiler must be avaible or the environment variable `CUDACXX` must be set to point to the location of `nvcc`.

By default, both CPU and GPU libraries are compiled.
To compile for CPU only:
```
make cpu
```

To compile for GPU only:
```
make gpu
```

Note: The targeted GPU architecture can be changed inside the makefile at the `CUDAFLAGS` section. By default, architectures from sm_70 to sm_90 are supported.

## Usage

The configuration file should include the following in the components section:
```
- rfi_vis: RiemannVisTimeFreqCalculationFFI
```
