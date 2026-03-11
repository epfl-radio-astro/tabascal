"""ReFrame performance regression check for the tabascal pipeline."""

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TabascalPerfCheck(rfm.RunOnlyRegressionTest):
    """Verify tabascal pipeline performance on GPU."""

    variant = parameter(["Riemann", "RiemannFFI"])

    valid_systems = ["daint:gpu"]
    valid_prog_environs = ["builtin"]
    time_limit = "30m"

    _reference_by_variant = {
        "Riemann": {
            "daint:gpu": {
                "total_runtime": (35.0, -0.15, 0.15, "s"),
                "optimizer_runtime": (52.0, -0.15, 0.15, "s"),
            },
        },
        "RiemannFFI": {
            "daint:gpu": {
                "total_runtime": (35.0, -0.15, 0.15, "s"),
                "optimizer_runtime": (52.0, -0.15, 0.15, "s"),
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

        self.prerun_cmds = [
            "set -e",
            ". /opt/conda/etc/profile.d/conda.sh",
            "conda activate tab",
            "cd /tabascal/src",
            (
                f"python ci/reframe/prepare_data.py "
                f"--components '{components_str}' "
                f"--workdir /tmp/rfm_workdir"
            ),
        ]

        self.executable = "python"
        self.executable_opts = [
            "/tabascal/src/tabascal/scripts/run_tabascal.py",
            "-c",
            "/tmp/rfm_workdir/tab_target.yaml",
            "-s",
            "$(cat /tmp/rfm_workdir/sim_dir.txt)",
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
