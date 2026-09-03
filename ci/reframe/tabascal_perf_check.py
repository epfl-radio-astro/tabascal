"""ReFrame performance regression checks for the tabascal pipeline."""

import os
import re
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn

_src_root = str(Path(__file__).resolve().parents[2])

# Printed by the pipeline once the optimizer has converged.
_CHI2_PATTERN = r"Reduced Chi\^2 @ opt params : [\d.eE+-]+"

# Printed by _run_tabascal_impl.tabascal_subtraction just above the device table,
# depending on whether more than one global JAX device is visible, i.e. whether
# the RFI axis is sharded across devices.
_SHARDING_PATTERN = r"^Sharding RFI sources over (?P<n>\d+) devices"
_SINGLE_DEVICE_PATTERN = r"^Running on single device:"


def _device_row_pattern(kind=None):
    """Match one row of the device table printed by _run_tabascal_impl.print_devices.

    The table looks like::

        Device  Kind                Process
        ------  ------------------  -------
        cuda:0  NVIDIA GH200 120GB  0

    With ``kind`` given, only rows whose Kind column contains it are matched.
    The trailing integer is the process index, which is what keeps this from also
    matching the memory-usage table (its last two columns are fixed-point numbers
    or "n/a", never a bare integer preceded by whitespace).
    """
    kind_re = rf"\S[^\n]*{re.escape(kind)}[^\n]*?" if kind else r"\S.*?"
    return rf"^cuda:\d+\s+(?P<kind>{kind_re})\s+\d+\s*$"


class _TabascalPerfCheckBase(rfm.RunOnlyRegressionTest):
    """Shared setup for the tabascal pipeline performance checks."""

    variant = parameter(["Riemann", "RiemannFFI"])
    precision = parameter(["single", "double"])

    # Ends up in the ReFrame report and is used by reframe_to_bmf.py to keep the
    # single- and multi-GPU benchmark series apart on bencher.dev. Anything other
    # than "single" is appended to the benchmark name, so the existing
    # single-GPU series keep their historical names.
    gpu_mode = variable(str, value="single")

    valid_systems = ["daint:gpu", "generic:default"]
    valid_prog_environs = ["builtin"]
    time_limit = "30m"

    # The reference numbers below only mean anything on the node type they were
    # measured on, so sanity requires the run to have used exactly this many GPUs
    # of exactly this kind; anything else fails rather than being compared against
    # timings it cannot be compared against.
    #
    # The references below were re-measured on Daint after #103 swapped the
    # RFI-signal component in `_components_map` from the real-space
    # `ComplexRFI` to the scanned Fourier `ComplexRFIVarAnt`. That is a
    # deliberate accuracy-for-runtime trade, not a bug: see issue #107. Do not
    # "restore" the older, faster numbers -- they belong to a component that no
    # longer exists.
    _expected_gpus = 1
    _expected_device_kind = "GH200"

    # Keyed by (variant, precision); metrics without an entry are reported but
    # not checked by ReFrame.
    _reference_by_variant: dict = {}

    _components_map = {
        "Riemann": [
            "trajectory:FixedOrbit",
            "rfi_signal:ComplexRFIVarAnt",
            "rfi_vis:RiemannVis",
            "ast_vis:GPVisAst",
            "gains:UnitaryGains",
        ],
        "RiemannFFI": [
            "trajectory:FixedOrbit",
            "rfi_signal:ComplexRFIVarAnt",
            "rfi_vis:RiemannVisFFI",
            "ast_vis:GPVisAst",
            "gains:UnitaryGains",
        ],
    }

    def gpu_setup_cmds(self):
        """Commands selecting the GPUs the run may use, prepended to prerun_cmds."""
        return []

    def assert_devices(self):
        """Assert the pipeline ran on exactly the expected GPUs.

        Both the total row count of the device table and the count of rows naming
        the expected device kind have to match, which is what rules out a node
        mixing device kinds.
        """
        n_expected = self._expected_gpus
        kind = self._expected_device_kind

        n_devices = sn.count(
            sn.extractall(_device_row_pattern(), self.stdout, "kind")
        )
        n_kind = sn.count(
            sn.extractall(_device_row_pattern(kind), self.stdout, "kind")
        )

        return [
            sn.assert_eq(
                n_devices,
                n_expected,
                msg=f"pipeline listed {{0}} devices, expected {n_expected}",
            ),
            sn.assert_eq(
                n_kind,
                n_expected,
                msg=f"only {{0}} of the devices are {kind}, expected {n_expected}",
            ),
        ]

    @run_before("performance")
    def set_reference(self):
        self.reference = self._reference_by_variant.get(
            (self.variant, self.precision), {}
        )

    @run_before("run")
    def prepare_run(self):
        components = self._components_map[self.variant]
        components_str = ",".join(components)
        workdir = os.path.join(self.stagedir, "workdir")

        self.prerun_cmds = ["set -e"]

        if self.current_partition.fullname == "daint:gpu":
            self.prerun_cmds += [
                ". /opt/conda/etc/profile.d/conda.sh",
                "conda activate tab",
            ]

        self.prerun_cmds += self.gpu_setup_cmds()

        self.prerun_cmds.append(
            f"python {_src_root}/ci/reframe/prepare_data.py"
            f" --components '{components_str}'"
            f" --workdir {workdir}"
            f" --src-root {_src_root}"
            f" --precision {self.precision}"
        )

        self.executable = "python"
        self.executable_opts = [
            f"{_src_root}/tabascal/scripts/run_tabascal.py",
            "run",
            "-c",
            f"{workdir}/tab_target.yaml",
            "-s",
            f"$(cat {workdir}/sim_dir.txt)",
            "--extra-orbit-dir",
            f"{_src_root}/tabascal/data/tles",
            "-t",
        ]

    @performance_function("s")
    def total_runtime(self):
        # Extract the Mean column from the runtime statistics table.
        # Columns: Function | Calls | Total | Glob (%) | Rel (%) | Mean | Std
        return sn.extractsingle(
            r"^tabascal_subtraction\s+\d+\s+[\d.]+\s+\S+\s+[\d.]+%\s+[\d.]+%\s+(?P<val>[\d.]+)\s+s\s",
            self.stdout,
            "val",
            float,
        )

    @performance_function("s")
    def optimizer_runtime(self):
        # Extract the Mean column from the runtime statistics table.
        # Columns: Function | Calls | Total | Glob (%) | Rel (%) | Mean | Std
        return sn.extractsingle(
            r"^\s{2}run_opt\s+\d+\s+[\d.]+\s+\S+\s+[\d.]+%\s+[\d.]+%\s+(?P<val>[\d.]+)\s+s\s",
            self.stdout,
            "val",
            float,
        )


# Parse the per-device peak from the "Memory usage" table printed by
# print_memory_usage at the end of the run, e.g.:
#   Device  Peak (GB)  Limit (GB)
#   ------  ---------  ----------
#   cuda:0  32.798     102.005
# A cuda:N prefix is not enough to pin down that row any more: the device
# overview printed by print_devices at the start of the run uses the same
# first column, e.g. "cuda:0  NVIDIA GH200 120GB  0", and comes first, so
# it would be the one extractsingle returns. Both remaining columns are
# therefore required to be fixed-point numbers (print_memory_usage formats
# them with .3f) up to the end of the line, which neither the Kind column
# nor the integer Process column of the device table can satisfy. The limit
# may be "n/a" on backends that report no bytes_limit. Peak is in GB.
_MEMORY_PATTERN = r"^(?P<device>cuda:\d+)\s+(?P<val>\d+\.\d+)\s+(?:\d+\.\d+|n/a)\s*$"


@rfm.simple_test
class TabascalPerfCheck(_TabascalPerfCheckBase):
    """Verify tabascal pipeline performance on a single GPU."""

    descr = "tabascal pipeline performance (single GPU)"

    # References are keyed by (variant, precision).
    #
    # Re-measured on Daint nid005989 (commit 46b6153, JAX 0.6.0) after #103
    # replaced the real-space `rfi_signal:ComplexRFI` with the scanned Fourier
    # `ComplexRFIVarAnt`. The previous values described the deleted component,
    # so they are superseded rather than adjusted. Tolerances are unchanged.
    #
    # The optimiser got slower where the RFI-signal component is a large share
    # of the step (RiemannFFI: +58% single, +26% double) and slightly faster
    # where RFI-vis still dominates (Riemann: -10% single, -6% double). Setup
    # time is unchanged, so total_runtime moves with optimizer_runtime alone.
    # Memory is flat. See issue #107 for why the slower component is preferred.
    # The two Riemann memory references are the measurements from before the
    # baseline scan of #114, kept as the ceiling the scanned kernel has to stay
    # under. Their lower bound is open because the scan is expected to cut the
    # peak by roughly an order of magnitude and nothing below has been
    # re-measured on Daint: re-record the value and close the bound again from
    # the first CSCS run that includes the scan.
    #
    # The runtime references are left as they are, which is a bet rather than a
    # measurement: the scan costs nothing end to end on CPU, and a GPU cost of
    # the order #108 paid for the same treatment (~5%) would still sit inside
    # the existing band. If the first Daint run says otherwise, they move too.
    # The RiemannFFI references are untouched throughout, since the kernel
    # bounds the same term inside itself.
    _reference_by_variant = {
        ("Riemann", "single"): {
            "daint:gpu": {
                "total_runtime": (79.91, -0.25, 0.15, "s"),
                "optimizer_runtime": (66.28, -0.20, 0.15, "s"),
                "memory_usage": (8.679, None, 0.1, "GB"),
            },
        },
        ("Riemann", "double"): {
            "daint:gpu": {
                "total_runtime": (80.66, -0.25, 0.15, "s"),
                "optimizer_runtime": (66.60, -0.20, 0.15, "s"),
                "memory_usage": (16.887, None, 0.1, "GB"),
            },
        },
        ("RiemannFFI", "single"): {
            "daint:gpu": {
                "total_runtime": (31.33, -0.25, 0.15, "s"),
                "optimizer_runtime": (19.16, -0.20, 0.15, "s"),
                "memory_usage": (0.606, -0.15, 0.15, "GB"),
            },
        },
        ("RiemannFFI", "double"): {
            "daint:gpu": {
                "total_runtime": (41.13, -0.25, 0.15, "s"),
                "optimizer_runtime": (27.84, -0.20, 0.15, "s"),
                "memory_usage": (1.239, -0.15, 0.15, "GB"),
            },
        },
    }

    def gpu_setup_cmds(self):
        # The CI job exports CUDA_VISIBLE_DEVICES=0 (ci/cscs.yml), but pin the
        # single GPU here as well so the test is independent of the environment
        # it is launched from -- and so it stays single-GPU next to the
        # multi-GPU check below.
        return ["export CUDA_VISIBLE_DEVICES=0"]

    @sanity_function
    def validate(self):
        # No nvidia-smi cross-check here (unlike the multi-GPU test): this test
        # pins one GPU of a node that has several, so the device table the
        # pipeline prints -- one GH200 row, under the single-device header -- is
        # the only thing that says what the run actually used.
        return sn.all(
            [
                sn.assert_found(
                    _SINGLE_DEVICE_PATTERN,
                    self.stdout,
                    msg="pipeline did not report running on a single device",
                ),
                *self.assert_devices(),
                sn.assert_found(_CHI2_PATTERN, self.stdout),
            ]
        )

    # Only the first GPU device is considered (extractsingle takes the first
    # match), which is the only one this test runs on.
    @performance_function("GB")
    def memory_usage(self):
        return sn.extractsingle(_MEMORY_PATTERN, self.stdout, "val", float)


@rfm.simple_test
class TabascalMultiGpuPerfCheck(_TabascalPerfCheckBase):
    """Verify tabascal pipeline performance across all GPUs of the node.

    The pipeline shards the RFI-source axis over every *global* JAX device
    (tabascal.distributed), so making all GPUs of the node visible to a single
    process is enough to exercise the sharded path -- no launcher change needed.
    """

    descr = "tabascal pipeline performance (all GPUs)"
    gpu_mode = "all"

    _expected_gpus = 4

    # Re-measured on Daint nid005989 (commit 46b6153, JAX 0.6.0) alongside the
    # single-GPU references above, for the same reason: these described the
    # deleted `ComplexRFI`. The previous set came from PR #92's final CSCS run
    # (commit 7a41e4c, node nid005553). Tolerances are unchanged.
    #
    # The optimiser regression is largest here (+146% / +132% on RiemannFFI):
    # sharding the RFI-source axis over 4 GPUs divides the RFI-vis work but the
    # per-source signal transform stays on each device, so the component under
    # test is the dominant remaining term. Memory rises 0.238 -> 0.306 GB
    # (single) and 0.477 -> 0.614 GB (double); both are sub-GB totals where the
    # signal component's absolute cost is a large fraction.
    # The two Riemann memory references are the measurements from before the
    # baseline scan of #114, kept as the ceiling the scanned kernel has to stay
    # under. Their lower bound is open because the scan is expected to cut the
    # peak by roughly an order of magnitude and nothing below has been
    # re-measured on Daint: re-record the value and close the bound again from
    # the first CSCS run that includes the scan.
    #
    # The runtime references are left as they are, which is a bet rather than a
    # measurement: the scan costs nothing end to end on CPU, and a GPU cost of
    # the order #108 paid for the same treatment (~5%) would still sit inside
    # the existing band. If the first Daint run says otherwise, they move too.
    # The RiemannFFI references are untouched throughout, since the kernel
    # bounds the same term inside itself.
    _reference_by_variant: dict = {
        ("Riemann", "single"): {
            "daint:gpu": {
                "total_runtime": (64.59, -0.25, 0.15, "s"),
                "optimizer_runtime": (41.69, -0.20, 0.15, "s"),
                "memory_usage": (3.294, None, 0.15, "GB"),
            },
        },
        ("Riemann", "double"): {
            "daint:gpu": {
                "total_runtime": (70.65, -0.25, 0.15, "s"),
                "optimizer_runtime": (46.88, -0.20, 0.15, "s"),
                "memory_usage": (7.058, None, 0.15, "GB"),
            },
        },
        ("RiemannFFI", "single"): {
            "daint:gpu": {
                "total_runtime": (37.19, -0.25, 0.15, "s"),
                "optimizer_runtime": (22.26, -0.20, 0.15, "s"),
                "memory_usage": (0.306, -0.15, 0.15, "GB"),
            },
        },
        ("RiemannFFI", "double"): {
            "daint:gpu": {
                "total_runtime": (43.55, -0.25, 0.15, "s"),
                "optimizer_runtime": (27.61, -0.20, 0.15, "s"),
                "memory_usage": (0.614, -0.15, 0.15, "GB"),
            },
        },
    }

    def gpu_setup_cmds(self):
        # The CI job pins CUDA_VISIBLE_DEVICES=0 (ci/cscs.yml) for the
        # single-GPU checks; drop that pin so JAX sees every GPU of the node.
        # nvidia-smi is queried after the unset (NVML honours the variable in
        # recent drivers) and its count is checked against the device count the
        # pipeline reports, which is what makes this an "all GPUs" test rather
        # than a "more than one GPU" one. Failures of nvidia-smi are tolerated
        # under `set -e` and surface as a sanity failure with GPU count 0.
        return [
            "unset CUDA_VISIBLE_DEVICES",
            'echo "Available GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"',
        ]

    @sanity_function
    def validate(self):
        n_expected = self._expected_gpus

        n_avail = sn.extractsingle(
            r"^Available GPUs:\s+(?P<n>\d+)\s*$", self.stdout, "n", int
        )
        n_sharded = sn.extractsingle(_SHARDING_PATTERN, self.stdout, "n", int)

        return sn.all(
            [
                sn.assert_eq(
                    n_avail,
                    n_expected,
                    msg=f"expected a node with exactly {n_expected} GPUs, found {{0}}",
                ),
                sn.assert_eq(
                    n_sharded,
                    n_expected,
                    msg=f"pipeline sharded over {{0}} devices, expected {n_expected}",
                ),
                *self.assert_devices(),
                sn.assert_found(_CHI2_PATTERN, self.stdout),
            ]
        )

    # Peak over all local devices: with the RFI axis split across GPUs the
    # devices do not carry identical loads (n_rfi is padded up to a multiple of
    # the device count), so the largest peak is the meaningful number.
    @performance_function("GB")
    def memory_usage(self):
        return sn.max(sn.extractall(_MEMORY_PATTERN, self.stdout, "val", float))
