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


def test_simulation_key_covers_the_files_a_config_references(tmp_path):
    config = tmp_path / "sim_target.yaml"
    config.write_text("telescope: {}\n")
    (tmp_path / "tab_target.yaml").write_text("a: 1\n")
    bare = prepare_data.compute_sha256(config)
    (tmp_path / "tab_target.yaml").write_text("a: 2\n")
    assert prepare_data.compute_sha256(config) == bare  # a sibling config is not an input
    (tmp_path / "array.itrf.txt").write_text("0 0 0\n")
    with_layout = prepare_data.compute_sha256(config)
    (tmp_path / "array.itrf.txt").write_text("100 0 0\n")
    assert len({bare, with_layout, prepare_data.compute_sha256(config)}) == 3
