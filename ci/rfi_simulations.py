"""Publish tabsim simulations of the test configs to HuggingFace.

The pipeline tests (tests/test_tabascal_pipeline.py) and the ReFrame benchmark
(ci/reframe/prepare_data.py) download their input simulations from the
epfl-radio-astro/rfi-simulations dataset, from the branch ``tabsim_v<version>``
of the installed tabsim, and look up each simulation under the SHA256 of its
config file. Without a matching snapshot they fall back to running tabsim
themselves, which needs Space-Track credentials.

``check`` reports which configs have no snapshot on the branch of the latest
tabsim release. ``publish`` simulates those configs with the installed tabsim
and uploads them in a single commit. Configs are checked one by one rather than
by branch existence, so an edited config, or a failed upload that left the
branch behind, is filled in on the next run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

REPO_ID = "epfl-radio-astro/rfi-simulations"
SRC_ROOT = Path(__file__).resolve().parent.parent
CONFIG_GLOBS = ["tests/data/sim_target*.yaml", "ci/reframe/data/sim_target*.yaml"]


def compute_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sim_configs() -> dict[str, Path]:
    """Map the config hash the consumers look up to its config file."""
    configs = sorted(p for g in CONFIG_GLOBS for p in SRC_ROOT.glob(g))
    return {compute_sha256(p): p for p in configs}


def latest_tabsim_version() -> str:
    with urllib.request.urlopen("https://pypi.org/pypi/tabsim/json") as r:
        return json.load(r)["info"]["version"]


def missing_configs(api: HfApi, branch: str) -> dict[str, Path]:
    """Return the configs with no simulation directory on ``branch``."""
    configs = sim_configs()
    try:
        entries = api.list_repo_tree(REPO_ID, repo_type="dataset", revision=branch)
        present = {e.path for e in entries}
    except RevisionNotFoundError:
        present = set()
    return {h: p for h, p in configs.items() if h not in present}


def write_spacetrack_credentials(target_dir: Path) -> Path:
    """tabsim resolves the satellites' TLEs through Space-Track."""
    username = os.environ.get("SPACETRACK_LOGIN")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set SPACETRACK_LOGIN and SPACETRACK_PASSWORD: tabsim needs Space-Track "
            "to simulate the satellite RFI."
        )
    path = target_dir / "spacetrack_login.yaml"
    path.write_text(yaml.safe_dump({"username": username, "password": password}))
    return path


def check(args: argparse.Namespace) -> None:
    version = latest_tabsim_version()
    branch = f"tabsim_v{version}"
    missing = missing_configs(HfApi(), branch)
    for h, p in missing.items():
        print(f"{branch} is missing {p.relative_to(SRC_ROOT)} ({h})")
    if not missing:
        print(f"{branch} has every simulation.")

    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a") as f:
            f.write(f"version={version}\n")
            f.write(f"missing={'true' if missing else 'false'}\n")


def publish(args: argparse.Namespace) -> None:
    import tabsim

    version = tabsim.__version__
    if args.expect_version and version != args.expect_version:
        raise RuntimeError(
            f"Installed tabsim is {version}, expected {args.expect_version}."
        )
    branch = f"tabsim_v{version}"

    api = HfApi()
    missing = missing_configs(api, branch)
    if not missing:
        print(f"{branch} has every simulation; nothing to publish.")
        return

    sim_script = Path(tabsim.__file__).parent / "scripts" / "sim_vis.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        credentials = write_spacetrack_credentials(tmp)
        upload_dir = tmp / "upload"

        for h, config in missing.items():
            print(f"Simulating {config.relative_to(SRC_ROOT)} into {h}", flush=True)
            # Paths inside the config (itrf_path, ...) resolve against the
            # config's own directory, so only the output and login move here.
            subprocess.run(
                [
                    sys.executable,
                    str(sim_script),
                    "-c",
                    str(config),
                    "-sp",
                    str(upload_dir / h),
                    "-st",
                    str(credentials),
                ],
                cwd=tmp,
                check=True,
            )

        names = ", ".join(str(p.relative_to(SRC_ROOT)) for p in missing.values())
        api.create_branch(REPO_ID, repo_type="dataset", branch=branch, exist_ok=True)
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=branch,
            folder_path=upload_dir,
            commit_message=f"Simulate {names} with tabsim {version}",
        )
    print(f"Uploaded {len(missing)} simulation(s) to {REPO_ID}@{branch}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser(
        "check", help="Report whether the latest tabsim release lacks simulations."
    )
    p.set_defaults(func=check)

    p = sub.add_parser(
        "publish", help="Simulate the missing configs and upload them."
    )
    p.add_argument(
        "--expect-version",
        help="Fail unless the installed tabsim has this version.",
    )
    p.set_defaults(func=publish)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
