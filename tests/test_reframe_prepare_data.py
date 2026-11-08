"""The ReFrame data preparation hands tabsim a single GPU."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("tabsim")
_spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).parents[1] / "ci" / "reframe" / "prepare_data.py"
)
prepare_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_data)


@pytest.mark.parametrize(
    "environ, gpu",
    [({}, "0"), ({"CUDA_VISIBLE_DEVICES": ""}, ""), ({"CUDA_VISIBLE_DEVICES": "2,3"}, "2"),
     ({"CUDA_VISIBLE_DEVICES": "GPU-5c1a,GPU-77ab"}, "GPU-5c1a")],
)
def test_simulation_gpu(environ, gpu):
    assert prepare_data.simulation_gpu(environ) == gpu


@pytest.mark.parametrize("reference", ["array.itrf.txt", "layouts/array.itrf.txt", "../array.itrf.txt"])
def test_simulation_key_covers_the_files_a_config_references(tmp_path, reference):
    config_dir = tmp_path / "configs"
    layout = (config_dir / reference).resolve()
    layout.parent.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    config = config_dir / "sim_target.yaml"
    config.write_text(f"telescope:\n  itrf_path: {reference}\noutput:\n  path: ./data\n")
    (config_dir / "tab_target.yaml").write_text("a: 1\n")
    layout.write_text("0 0 0\n")
    key = prepare_data.compute_sha256(config)
    (config_dir / "tab_target.yaml").write_text("a: 2\n")
    (config_dir / "data").mkdir()
    assert prepare_data.compute_sha256(config) == key  # neither is an input
    layout.write_text("100 0 0\n")
    assert prepare_data.compute_sha256(config) != key
