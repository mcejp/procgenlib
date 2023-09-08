# procgenlib

---

## Test

    python -m pytest

## Format check

    black --diff **.py

## Type check

    mypy procgenlib

## Regenerate docs

    sphinx-apidoc -o doc procgenlib
    env PYTHONPATH=. sphinx-build doc doc/_build
