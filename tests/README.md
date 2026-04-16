# Tests

## Running tests

```bash
# All tests (from tabascal/)
pytest

# Single file
pytest tests/components/test_gains.py

# Single test
pytest tests/components/test_gains.py::TestGPGains::test_forward_output_shapes
```

SGP4 component tests use a bundled TLE cache (`tabascal/data/tles/`) and run without Space-Track credentials. Pipeline tests that use `SGP4LEONoDragOrbit` or `SGP4LEOOrbit` require credentials and are skipped automatically when none are found.

## Test inventory

`TEST_INVENTORY.md` is a generated file — do not edit it by hand.

To regenerate it after adding or modifying tests:

```bash
python tests/generate_inventory.py
```

The script collects test IDs via `pytest --collect-only` and reads descriptions from each test function's docstring (first line only). The output is a Markdown table per test class, with each test name linked to its source line.

## Writing tests compatible with the generator

**Add a one-line docstring to every test function.** This becomes the Description column in the inventory. Without a docstring the entry shows `*(no docstring)*`.

```python
def test_forward_output_shape(self):
    """Forward pass produces rfi_phase with shape (n_rfi, n_ant, n_freq_fine, n_time_fine)."""
    ...
```

**Parametrized tests** are expanded automatically — all variants share the function's docstring. Write the docstring to describe what the whole family tests, not a specific parameter combination.

```python
@pytest.mark.parametrize("n_ant,n_freq,n_time", [(2, 1, 4), (5, 3, 12)])
def test_setup_and_forward_various_sizes(self, n_ant, n_freq, n_time):
    """Setup and forward succeed end-to-end for the given (n_ant, n_freq, n_time)."""
    ...
```

**New test files** must be added to `FILE_ORDER` in `generate_inventory.py` to appear in the inventory. If the file or any of its test classes warrant a prose note (e.g. credential requirements, scope), add an entry to `FILE_NOTES` or `CLASS_NOTES` in the same script.
