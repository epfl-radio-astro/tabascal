# tabascal
New home for tabascal with all private code included.


# Build the documentation

Inside the base directory of tabascal run the follwoing commands.

## Install docs env

```bash
pip install -e "./[docs]"
```

## Build the docs

```bash
sphinx-build -b html docs docs/_build/html
```

After this you can `open docs/_build/html/index.html`.
