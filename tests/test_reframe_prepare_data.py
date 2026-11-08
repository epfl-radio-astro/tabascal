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
