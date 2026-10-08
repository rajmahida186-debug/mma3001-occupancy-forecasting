# Source code

The reusable Python code lives in the `occupancy` package. Every function has a
docstring (what it does, its inputs and outputs) and inline comments. The
HTML documentation in `docs/` is generated from these docstrings with:

```bash
pdoc src/occupancy -o docs
```
