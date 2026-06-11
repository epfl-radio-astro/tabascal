"""ReFrame performance regression check for the tabascal pipeline."""

import os
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn

_src_root = str(Path(__file__).resolve().parents[2])


class _TabascalRunBase(rfm.RunOnlyRegressionTest):
    """Common configuration and run setup for the tabascal checks."""

    valid_systems = ["daint:gpu", "generic:default"]
    valid_prog_environs = ["builtin"]
    time_limit = "30m"

    # Subclasses must define ``variant`` and ``precision`` (as parameters or
    # plain attributes). ``measure_memory`` enables GPU memory profiling.
    measure_memory = False

    _components_map = {
        "Riemann": [
            "trajectory:FixedOrbit",
            "rfi_signal:ComplexRFI",
            "rfi_vis:RiemannVisTimeFreqCalculation",
            "ast_vis:FourierTimeFreqGPAst",
            "gains:UnitaryGains",
        ],
        "RiemannFFI": [
            "trajectory:FixedOrbit",
            "rfi_signal:ComplexRFI",
            "rfi_vis:RiemannVisTimeFreqCalculationFFI",
            "ast_vis:FourierTimeFreqGPAst",
            "gains:UnitaryGains",
        ],
    }

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

        # Memory checks only need a few optimizer iterations to reach peak
        # usage, while timing checks run the full optimization.
        max_iter = 10 if self.measure_memory else 1000

        self.prerun_cmds.append(
            f"python {_src_root}/ci/reframe/prepare_data.py"
            f" --components '{components_str}'"
            f" --workdir {workdir}"
            f" --src-root {_src_root}"
            f" --precision {self.precision}"
            f" --max-iter {max_iter}"
        )

        run_cmd = [
            "python",
            f"{_src_root}/tabascal/scripts/run_tabascal.py",
            "run",
            "-c",
            f"{workdir}/tab_target.yaml",
            "-s",
            f"$(cat {workdir}/sim_dir.txt)",
            "--extra-tle-dir",
            f"{_src_root}/tabascal/data/tles",
            "-t",
        ]

        if self.measure_memory:
            # Use the platform allocator so JAX releases memory on demand,
            # giving an accurate peak GPU memory reading via gpu_stats. This
            # disables the caching BFC allocator and slows execution, so this
            # path is kept separate from the timing references.
            self.env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
            # Wrap the run with gpu_stats (https://github.com/AdhocMan/gpu_stats),
            # which must be available on PATH on the target system.
            self.executable = "gpu_stats"
            self.executable_opts = run_cmd
        else:
            self.executable = run_cmd[0]
            self.executable_opts = run_cmd[1:]

    @sanity_function
    def validate(self):
        return sn.assert_found(
            r"Reduced Chi\^2 @ opt params : [\d.eE+-]+", self.stdout
        )


@rfm.simple_test
class TabascalPerfCheck(_TabascalRunBase):
    """Verify tabascal pipeline performance on GPU."""

    variant = parameter(["Riemann", "RiemannFFI"])
    precision = parameter(["single", "double"])

    # References are keyed by (variant, precision). Single precision timings
    # are placeholders to be measured and updated later.
    _reference_by_variant = {
        ("Riemann", "single"): {
            "daint:gpu": {
                "total_runtime": (86.0, -0.25, 0.15, "s"),
                "optimizer_runtime": (74.0, -0.20, 0.15, "s"),
            },
        },
        ("Riemann", "double"): {
            "daint:gpu": {
                "total_runtime": (84.0, -0.25, 0.15, "s"),
                "optimizer_runtime": (71.0, -0.20, 0.15, "s"),
            },
        },
        ("RiemannFFI", "single"): {
            "daint:gpu": {
                "total_runtime": (24.1, -0.25, 0.15, "s"),
                "optimizer_runtime": (12.1, -0.20, 0.15, "s"),
            },
        },
        ("RiemannFFI", "double"): {
            "daint:gpu": {
                "total_runtime": (35.4, -0.25, 0.15, "s"),
                "optimizer_runtime": (22.1, -0.20, 0.15, "s"),
            },
        },
    }

    @run_before("performance")
    def set_reference(self):
        self.reference = self._reference_by_variant[(self.variant, self.precision)]

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


@rfm.simple_test
class TabascalMemCheck(_TabascalRunBase):
    """Verify peak GPU memory usage of the RiemannFFI run."""

    variant = "RiemannFFI"
    precision = parameter(["single", "double"])
    measure_memory = True

    # Peak memory references keyed by precision. The single precision value is
    # a placeholder to be measured and updated later.
    _reference_by_precision = {
        "single": {
            "daint:gpu": {
                "max_memory": (2024, -0.20, 0.10, "MB"),
            },
        },
        "double": {
            "daint:gpu": {
                "max_memory": (4047, -0.20, 0.10, "MB"),
            },
        },
    }

    @run_before("performance")
    def set_reference(self):
        self.reference = self._reference_by_precision[self.precision]

    @performance_function("MB")
    def max_memory(self):
        # gpu_stats prints a per-process summary line to stdout:
        #   total memory: mean <X> MB, max <Y> MB
        # Take the largest reported peak across any reported process.
        return sn.max(
            sn.extractall(
                r"total memory:\s+mean\s+[\d.]+\s+MB,\s+max\s+(?P<val>[\d.]+)\s+MB",
                self.stdout,
                "val",
                float,
            )
        )
