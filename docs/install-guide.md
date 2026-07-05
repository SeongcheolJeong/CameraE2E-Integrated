# Install Guide

## Who This Is For

This guide is for first-time `pyisetcam` users who want to run the explicit-object imaging pipeline locally from this repository. It focuses on basic package use first and treats Octave parity tooling as optional developer setup.

## Requirements

- Python `3.12`
- Git
- enough disk space for the package environment plus the pinned upstream asset snapshot stored under `.cache/`

Optional:

- GNU Octave if you want to regenerate parity baselines or run Octave-backed parity workflows

## Recommended Install

The recommended path uses the repo's conda environment because it matches the project's tested setup, including optional Octave support.

```bash
conda env create -f environment.yml
conda activate isetcam-py
python -m pip install -e .
```

## Alternative Minimal Install

If you only want the Python package and do not need the conda environment, you can use a standard virtual environment instead.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Fetch Or Reuse Upstream Assets

`pyisetcam` resolves many scene, display, optics, and spectral assets from a pinned upstream ISETCam snapshot. In normal use, the first call that touches `AssetStore.default()` will fetch that snapshot automatically.

You can also fetch the snapshot explicitly:

```bash
python tools/fetch_upstream.py
```

Useful environment variables:

- `PYISETCAM_CACHE_ROOT`: move the default cache root away from `.cache/`
- `PYISETCAM_UPSTREAM_ROOT`: point the package at an already-available upstream snapshot

The default snapshot location is:

```text
.cache/upstream/isetcam/<pinned-sha>/
```

## Verify The Install

Run the two baseline example scripts:

```bash
python examples/end_to_end.py
python examples/scene_from_file.py
```

You should see output paths printed to the terminal, and these files should exist:

- `reports/tutorial/end_to_end/end_to_end.png`
- `reports/tutorial/scene_from_file/input.png`
- `reports/tutorial/scene_from_file/result.png`

If those files are created successfully, your basic install is working.

## Optional Developer And Parity Setup

Octave is only needed for parity and baseline workflows. It is not required for normal package use or the tutorial examples.

Useful developer commands:

```bash
pytest
python tools/parity_report.py
python tools/regenerate_parity_baselines.py
```

If your `octave` wrapper is not the right binary, point the tooling at a specific `octave-cli` executable:

```bash
PYISETCAM_OCTAVE_BIN=/abs/path/to/octave-cli python tools/regenerate_parity_baselines.py
```

## Troubleshooting

### Asset download problems

- Try `python tools/fetch_upstream.py` explicitly.
- If you already have the upstream snapshot elsewhere, set `PYISETCAM_UPSTREAM_ROOT`.
- If your cache location is restricted, set `PYISETCAM_CACHE_ROOT` to a writable path.

### Wrong Python version

- `pyproject.toml` requires Python `3.12`.
- Recreate the environment with Python `3.12` instead of attempting to force-install on an older interpreter.

### Octave parity tooling fails

- Confirm that Octave is installed only if you need parity tooling.
- If the discovered binary is wrong, set `PYISETCAM_OCTAVE_BIN=/abs/path/to/octave-cli`.
- The conda environment in `environment.yml` already includes Octave on the recommended path.

### Example scripts run but produce no files

- Check that the repo is writable.
- Confirm the scripts are writing into `reports/tutorial/...`.
- Re-run the examples and inspect the printed output paths.

## Next Step

Continue with the hands-on walkthrough in [Getting Started Tutorial](./tutorial.md).
