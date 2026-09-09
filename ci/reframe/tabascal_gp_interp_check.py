"""ReFrame runtime and memory checks for the data-grid RFI route.

The same pipeline as :mod:`tabascal_perf_check`, on the same 96-antenna
simulation, with the RFI signal and phase carried on the data grid and the
fine grid rebuilt inside ``rfi_vis:GPInterpVis`` -- see
``docs/components/rfi_vis.md``. Two variants:

* ``GPInterp``: every time step at once, so the fine grids are formed whole
  inside the forward, as ``RiemannVis`` forms them, and only their absence
  from the model state differs.
* ``GPInterpBlocked``: ``rfi.time_block_size`` set, so the fine grids exist
  for one block of time steps at a time under a checkpointed scan. This is
  the memory case.

Kept out of ``ci/cscs.yml``'s perf job, which names ``tabascal_perf_check.py``
alone: that job is serial under a 30-minute limit with no room for four more
runs. Run by hand on a GH200 node beside the ``Riemann`` and ``RiemannFFI``
variants of the same commit, which is the comparison that means something::

    reframe -C ci/reframe/settings.py \
        -c ci/reframe/tabascal_perf_check.py -c ci/reframe/tabascal_gp_interp_check.py \
        --system=daint:gpu --exec-policy=serial --run --performance-report \
        -S strict_check=0

No references: the metrics are reported, not asserted.

Measured on Daint nid005984 (commit 29ed2fd, JAX 0.10.2 from the pixi
``cuda12-dev`` environment), all sixteen runs in one job, so the fine-grid
columns are the same commit and build rather than the older references in
``tabascal_perf_check.py``. Peak memory in GB, runtimes in seconds::

    single GPU                     memory        total runtime    optimizer
                                 single double   single double   single double
    Riemann                       0.751  1.463    155.3  157.3    120.1  129.2
    RiemannFFI                    0.586  1.167     61.0   67.7     35.7   43.1
    GPInterp                      0.532  1.063    149.6  151.2    123.7  122.4
    GPInterpBlocked (10 steps)    0.258  0.391    238.3  157.5    213.4  128.2

    all 4 GPUs
    Riemann                       0.295  0.574     99.5   94.2     70.6   65.0
    RiemannFFI                    0.265  0.534     68.9   72.8     40.0   44.2
    GPInterp                      0.241  0.473    121.5  113.0     75.7   70.1
    GPInterpBlocked (10 steps)    0.191  0.306    120.5   94.1     90.2   63.3

The blocked route peaks at a third of ``Riemann`` (2.9x / 3.7x less in single
/ double on one GPU) and below the FFI kernel, at ``Riemann``'s optimiser time
in double. Two things stand out and are not yet understood: the unblocked
route saves almost no memory over ``Riemann`` -- the fine grids are no longer
model state but are still formed whole inside the forward, so the peak is
the same term -- and the blocked route is 1.7x slower in single precision
than in double on one GPU (1.4x on four), where every other variant runs the
two precisions at the same speed.
"""

import os
import sys

import reframe as rfm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabascal_perf_check import TabascalMultiGpuPerfCheck, TabascalPerfCheck  # noqa: E402

_DATA_GRID = [
    "trajectory:FixedOrbitCoarse",
    "rfi_signal:ComplexRFIVarAntCoarse",
    "rfi_vis:GPInterpVis",
    "ast_vis:GPVisAst",
    "gains:UnitaryGains",
]

_COMPONENTS = {"GPInterp": _DATA_GRID, "GPInterpBlocked": _DATA_GRID}

# The 96A simulation has 90 time steps: 9 blocks of 10.
_OVERRIDES = {"GPInterpBlocked": {"rfi": {"time_block_size": 10}}}


@rfm.simple_test
class TabascalGPInterpPerfCheck(TabascalPerfCheck):
    """The data-grid route on a single GPU."""

    descr = "tabascal data-grid pipeline performance (single GPU)"
    variant = parameter(["GPInterp", "GPInterpBlocked"])
    _components_map = _COMPONENTS
    _config_overrides_map = _OVERRIDES
    _reference_by_variant: dict = {}


@rfm.simple_test
class TabascalGPInterpMultiGpuPerfCheck(TabascalMultiGpuPerfCheck):
    """The data-grid route across all GPUs of the node."""

    descr = "tabascal data-grid pipeline performance (all GPUs)"
    variant = parameter(["GPInterp", "GPInterpBlocked"])
    _components_map = _COMPONENTS
    _config_overrides_map = _OVERRIDES
    _reference_by_variant: dict = {}
