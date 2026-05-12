# Archived notebooks

These are the original Jupyter notebooks (and their `build_notebook.py`
generators) for the experiments. They are kept here for provenance only.

The repository's **main flow is `main.py`** (see the top-level `README.md`).
Each notebook has an executable Python equivalent under
`experiments/<EXP>/run_<exp>.py`, generated from the notebook by
`pipeline/convert_notebooks.py`. The generated runners are what `main.py`
executes, and they are kept in sync with these notebooks:

```bash
python pipeline/convert_notebooks.py          # regenerate the runners
python pipeline/convert_notebooks.py --check   # CI check that they match
```

If you edit an archived notebook, re-run the converter to refresh the
corresponding `run_<exp>.py`.
