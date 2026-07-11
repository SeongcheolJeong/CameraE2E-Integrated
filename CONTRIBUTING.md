# Contributing to CameraE2E Integrated

## Development Setup

```bash
./tools/workbench.sh bootstrap
source .venv/bin/activate
```

Before opening a pull request:

```bash
./tools/workbench.sh check
python -m pytest
cd camerae2e-workbench && npm run build
```

Repository-wide Ruff and Mypy cleanup is an existing porting backlog and is not yet a
release gate. New or changed Python modules should still be checked with scoped Ruff and
Mypy commands where practical; do not apply repository-wide automatic fixes.

## Contribution Rules

- Keep research readiness and physical accuracy separate.
- Do not promote proxy, public-derived, or solver-only evidence to calibrated status.
- Record units, source, hashes, valid domain, dependencies, and validation gates for
  new DB/LUT artifacts.
- Do not commit proprietary datasets, detector weights, credentials, user projects, or
  generated run directories.
- Add focused tests for behavior changes and preserve existing public APIs unless a
  migration is documented.
- Keep UI controls connected to real computation. Disable unavailable functionality
  instead of fabricating output.

## Pull Requests

Describe the user problem, implementation, validation commands, fidelity implications,
and any required external assets. Screenshots are required for visible Workbench changes.

## Reporting Problems

Include the Git commit, operating system, Python and Node versions, `workbench.sh check`
output, relevant log excerpt, and whether KITTI/model/solver/calibration assets are local.
Do not attach restricted datasets or model weights to public issues.
