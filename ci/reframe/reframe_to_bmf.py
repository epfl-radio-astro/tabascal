"""Convert a ReFrame --report-file JSON to Bencher Metric Format (BMF)."""

import argparse
import json


def reframe_to_bmf(report_path: str, jax_version: str | None = None) -> dict:
    with open(report_path) as f:
        report = json.load(f)

    bmf: dict = {}
    for run in report["runs"]:
        for tc in run["testcases"]:
            if tc["result"] != "pass":
                continue
            variant = tc.get("variant", tc["name"])
            precision = tc.get("precision")
            # Set by the checks in tabascal_perf_check.py. Only non-"single"
            # modes add a name segment, so the single-GPU series keep the
            # benchmark names they have been tracked under so far -- without it
            # the multi-GPU cases would collide with them and overwrite them in
            # this dict.
            gpu_mode = tc.get("gpu_mode")
            for key, values in tc["perfvalues"].items():
                # key format: "system:partition:metric_name"
                metric = key.rsplit(":", 1)[-1]
                measured = values[0]  # first element is the value
                unit = values[4] if len(values) > 4 else "s"
                parts = [variant]
                if precision:
                    parts.append(precision)
                if gpu_mode and gpu_mode != "single":
                    parts.append(f"gpus-{gpu_mode}")
                if jax_version:
                    parts.append(f"jax-{jax_version}")
                parts.append(metric)
                bench_name = "/".join(parts)
                bmf[bench_name] = {
                    "latency": {
                        "value": measured,
                        "lower_value": None,
                        "upper_value": None,
                        "unit": unit,
                    }
                }
    return bmf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to ReFrame JSON report file")
    parser.add_argument("--jax-version", help="JAX version to include in benchmark names")
    args = parser.parse_args()
    print(json.dumps(reframe_to_bmf(args.report, args.jax_version), indent=2))
