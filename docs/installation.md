# Developer Install

```bash
git clone git@github.com:epfl-radio-astro/tabascal.git
pip install -e ./tabascal[dev]
```

# Building Documentation

Navigate to the base directory of the tabascal repository and run the `sphinx-build` command.

```bash
sphinx-build -b html docs docs/_build/html
```