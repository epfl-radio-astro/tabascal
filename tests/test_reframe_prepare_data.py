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
    config.write_text(f"telescope:\n  itrf_path: {reference}\nobservation:\n  target_name: data\n"
                      "output:\n  path: ./data\n")
    (config_dir / "tab_target.yaml").write_text("a: 1\n")
    layout.write_text("0 0 0\n")
    key = prepare_data.compute_sha256(config)
    (config_dir / "tab_target.yaml").write_text("a: 2\n")
    (config_dir / "data").mkdir()
    assert prepare_data.compute_sha256(config) == key  # none of these is an input
    layout.write_text("100 0 0\n")
    assert prepare_data.compute_sha256(config) != key


def test_simulation_key_refuses_inputs_resolved_against_the_working_directory(tmp_path):
    config = tmp_path / "sim_target.yaml"
    config.write_text("rfi_sources:\n  stationary:\n    geo_path: locations.loc\n")
    with pytest.raises(ValueError, match="working directory"):
        prepare_data.compute_sha256(config)


def test_the_simulation_directory_is_found_by_its_configs_prefix(tmp_path):
    config = tmp_path / "sim_target.yaml"
    config.write_text("output:\n  path: ./data\n  prefix: ska_low\n")
    assert prepare_data.simulation_prefix(config) == "ska_low_obs_"


def test_the_run_config_keeps_the_templates_settings_and_finds_its_inputs(tmp_path):
    """Only the settings given are replaced; inputs beside the template are made absolute.

    The run starts in ReFrame's stage directory, where tabascal would look for a
    relative input, so one left relative would not be found.
    """
    (tmp_path / "ids.txt").write_text("45399\n")
    template = tmp_path / "tab_target.yaml"
    template.write_text(
        "model:\n  precision: single\n  components: [rfi_vis:AnalyticVis]\n"
        "opt:\n  max_iter: 500\n  epsilon: 1e-3\n"
        "satellites:\n  norad_ids_path: ids.txt\n"
        "gains:\n  init: prior\n"
    )

    config = prepare_data.prepare_config(template)
    assert config["model"] == {"precision": "single", "components": ["rfi_vis:AnalyticVis"]}
    assert config["opt"] == {"max_iter": 500, "epsilon": 1e-3}
    assert "rfi" not in config
    assert config["satellites"]["norad_ids_path"] == str((tmp_path / "ids.txt").resolve())
    assert config["gains"]["init"] == "prior"  # not a file beside the template

    config = prepare_data.prepare_config(
        template, components=["rfi_vis:RiemannVisFine"], precision="double",
        rfi_backend="jax", max_iter=3,
    )
    assert config["model"] == {"precision": "double", "components": ["rfi_vis:RiemannVisFine"]}
    assert config["opt"]["max_iter"] == 3
    assert config["rfi"] == {"backend": "jax"}
