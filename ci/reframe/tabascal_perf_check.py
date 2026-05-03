"""ReFrame performance regression check for the tabascal pipeline."""

import os
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn

_src_root = str(Path(__file__).resolve().parents[2])


@rfm.simple_test
class TabascalPerfCheck(rfm.RunOnlyRegressionTest):
    """Verify tabascal pipeline performance on GPU."""

    variant = parameter(["Riemann", "RiemannFFI"])

    valid_systems = ["daint:gpu", "generic:default"]
    valid_prog_environs = ["builtin"]
    time_limit = "30m"

    _reference_by_variant = {
        "Riemann": {
            "daint:gpu": {
                "total_runtime": (34.0, -0.25, 0.15, "s"),
                "optimizer_runtime": (10.0, -0.20, 0.15, "s"),
            },
        },
        "RiemannFFI": {
            "daint:gpu": {
                "total_runtime": (23, -0.25, 0.15, "s"),
                "optimizer_runtime": (5.2, -0.20, 0.15, "s"),
            },
        },
    }

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

    @run_before("performance")
    def set_reference(self):
        self.reference = self._reference_by_variant[self.variant]

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

        self.prerun_cmds.append(
            f"python {_src_root}/ci/reframe/prepare_data.py"
            f" --components '{components_str}'"
            f" --workdir {workdir}"
            f" --src-root {_src_root}"
        )

        self.executable = "python"
        self.executable_opts = [
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

    @sanity_function
    def validate(self):
        return sn.assert_found(
            r"Reduced Chi\^2 @ opt params : [\d.eE+-]+", self.stdout
        )

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
