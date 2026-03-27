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
            for key, values in tc["perfvalues"].items():
                # key format: "system:partition:metric_name"
                metric = key.rsplit(":", 1)[-1]
                measured = values[0]  # first element is the value
                parts = [variant]
                if jax_version:
                    parts.append(f"jax-{jax_version}")
                parts.append(metric)
                bench_name = "/".join(parts)
                bmf[bench_name] = {
                    "latency": {
                        "value": measured,
                        "lower_value": None,
                        "upper_value": None,
                    }
                }
    return bmf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to ReFrame JSON report file")
    parser.add_argument("--jax-version", help="JAX version to include in benchmark names")
    args = parser.parse_args()
    print(json.dumps(reframe_to_bmf(args.report, args.jax_version), indent=2))
