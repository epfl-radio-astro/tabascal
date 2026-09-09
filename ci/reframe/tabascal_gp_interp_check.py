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
