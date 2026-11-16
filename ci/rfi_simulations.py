"""Publish tabsim simulations of the test configs to HuggingFace.

The pipeline tests (tests/test_tabascal_pipeline.py) and the ReFrame benchmark
(ci/reframe/prepare_data.py) download their input simulations from the
epfl-radio-astro/rfi-simulations dataset, from the branch ``tabsim_v<version>``
of the installed tabsim, and look up each simulation under the SHA256 of its
config file. Each downloads only the simulations of its own configs
(download_simulations), so the pipeline tests never fetch the benchmark's.
Without a matching snapshot they fall back to running tabsim themselves.

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

from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError
from packaging.version import Version
import yaml

REPO_ID = "epfl-radio-astro/rfi-simulations"
SRC_ROOT = Path(__file__).resolve().parent.parent
CONFIG_GLOBS = ["tests/data/sim_target*.yaml", "ci/reframe/data/sim_target*.yaml"]


#: The tabsim config keys that name an input file or directory.
_INPUT_KEYS = {
    "path", "itrf_path", "enu_path", "norad_ids_path", "norad_spec_model",
    "extra_orbit_dir", "replay_orbit_dir", "circ_path", "spec_model", "geo_path",
}
#: Inputs tabsim 0.1.0 resolves against the working directory, not the config's,
#: so a relative one names a different file depending on where the run starts.
_CWD_RELATIVE = {
    ("satellite", "circ_path"), ("satellite", "spec_model"),
    ("stationary", "geo_path"), ("stationary", "spec_model"),
}


def _referenced_inputs(value, base: Path, parent: str = ""):
    """(reference, path) for every input a tabsim config names that exists."""
    if isinstance(value, list):
        for item in value:
            yield from _referenced_inputs(item, base, parent)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "output":  # where the simulation writes, not what it reads
                continue
            if key in _INPUT_KEYS and isinstance(item, str) and item:
                path = Path(item)  # tabsim does not expand "~", so neither does the key
                if not path.is_absolute():
                    if (parent, key) in _CWD_RELATIVE:
                        raise ValueError(
                            f"{parent}.{key}: {item!r} is resolved against the working "
                            "directory, so it cannot key a published simulation; give "
                            "it as an absolute path"
                        )
                    path = base / path
                if path.exists():
                    yield item, path
            else:
                yield from _referenced_inputs(item, base, key)


def compute_sha256(file_path: Path) -> str:
    """The key a simulation is stored and looked up under.

    The config's bytes, then the reference and contents of every input file or
    directory it names that exists (an antenna layout, say), resolved against the
    config's directory as tabsim resolves them. Changing a referenced input then
    gives a new key rather than reusing a simulation of the old one.
    ci/reframe/prepare_data.py and tests/test_tabascal_pipeline.py load this
    function, so the publisher and its consumers always agree.
    """
    file_path = Path(file_path)
    sha256_hash = hashlib.sha256(file_path.read_bytes())
    config = yaml.safe_load(file_path.read_text()) or {}
    for reference, path in sorted(set(_referenced_inputs(config, file_path.parent))):
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for f in files:
            name = reference if f == path else f"{reference}/{f.relative_to(path).as_posix()}"
            sha256_hash.update(b"\0" + name.encode() + b"\0")
            sha256_hash.update(f.read_bytes())
    return sha256_hash.hexdigest()


def download_simulations(configs, revision: str) -> Path:
    """Download the published simulations of ``configs`` from ``revision``.

    Only those simulations, not the whole branch: it holds every config's, and
    the ReFrame benchmark's are far larger than anything the pipeline tests need.
    Returns the snapshot root, under which each simulation is stored by its
    config's hash; a config the branch has no simulation for has no directory.
    Raises whatever huggingface_hub does if the branch cannot be reached.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{compute_sha256(c)}/*" for c in configs],
        )
    )


def sim_configs() -> dict[str, Path]:
    """Map the config hash the consumers look up to its config file."""
    configs = sorted(p for g in CONFIG_GLOBS for p in SRC_ROOT.glob(g))
    return {compute_sha256(p): p for p in configs}


def latest_tabsim_version() -> str:
    """The newest final, unyanked tabsim release."""
    with urllib.request.urlopen("https://pypi.org/pypi/tabsim/json") as r:
        releases = json.load(r)["releases"]
    versions = [
        Version(v) for v, files in releases.items()
        if files and not all(f.get("yanked") for f in files)
    ]
    return str(max(v for v in versions if not v.is_prerelease))


def missing_configs(api: HfApi, branch: str) -> dict[str, Path]:
    """Return the configs with no simulation directory on ``branch``."""
    configs = sim_configs()
    try:
        entries = api.list_repo_tree(REPO_ID, repo_type="dataset", revision=branch)
        present = {e.path for e in entries}
    except RevisionNotFoundError:
        present = set()
    return {h: p for h, p in configs.items() if h not in present}


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
        upload_dir = tmp / "upload"

        for h, config in missing.items():
            print(f"Simulating {config.relative_to(SRC_ROOT)} into {h}", flush=True)
            # Paths inside the config (itrf_path, ...) resolve against the
            # config's own directory, so only the output moves here.
            subprocess.run(
                [
                    sys.executable,
                    str(sim_script),
                    "-c",
                    str(config),
                    "-sp",
                    str(upload_dir / h),
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
