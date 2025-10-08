# Developer guide

## Environment setup

1. Install Python 3.11 or newer.
2. Create a virtual environment and install the project in editable mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. Install the companion ``lrgsglib`` library if it is not already available:

   ```bash
   pip install ./lrgsglib
   ```

## Coding standards

* Follow [PEP 8](https://peps.python.org/pep-0008/) and keep lines under 88
  characters.  The project ships with ``black`` and ``isort`` configured in
  ``pyproject.toml``.
* All functions must include type hints.  Enable strict mypy checks locally by
  running ``mypy src``.
* Prefer explicit imports (``import module`` or ``from module import name``) and
  avoid ``from module import *``.
* Add docstrings describing the intent of each public function or class.

## Testing and quality gates

Run the quality suite before submitting changes:

```bash
pytest
black --check src
isort --check src
flake8 src
mypy src
```

Continuous integration should replicate the same checks.  The default pytest
configuration expects tests in the ``tests/`` directory.

## Documentation

* Update ``README.md`` whenever a user-facing feature changes.
* ``docs/overview.md`` explains the architecture; expand it when new modules are
  introduced.
* Add usage examples and edge cases directly inside the docstrings – they are
  rendered automatically by Sphinx if a documentation site is generated.

## Releasing

1. Bump the version number in ``pyproject.toml`` following semantic versioning.
2. Update the changelog (create ``docs/changelog.md`` if it does not exist).
3. Tag the release and build wheels:

   ```bash
   python -m build
   ```

4. Publish to PyPI or your internal package index using ``twine``.
