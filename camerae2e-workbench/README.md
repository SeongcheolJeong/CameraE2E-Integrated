# CameraE2E Workbench

React + FastAPI local Camera Design Decision Workbench.

## Run

```bash
cd CameraE2E-Integrated
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[workbench,yolo]"
cd camerae2e-workbench
npm install
npm run backend
npm run dev
```

Run `npm run backend` and `npm run dev` in separate terminals. Set
`PYTHON=/path/to/python` when the desired interpreter is not `python3`.

The default local URLs are:

- Frontend: http://127.0.0.1:5175 (or the next available Vite port)
- Backend: http://127.0.0.1:8010

The v2 UI uses persistent local projects and jobs for requirement-driven baseline
evaluation, sensitivity analysis, automated parameter optimization, shortlisted
solver validation, RAW dataset export, scoped calibration, and decision reports.
All results are stored as content-addressed evidence. Fidelity status explicitly
distinguishes analytic, LUT-backed, solver, and calibrated paths.

Local projects are written under `camerae2e-workbench/projects/` by default and
are intentionally ignored by Git. Set `CAMERAE2E_PROJECTS_ROOT` to use another
local root.

Set `CAMERAE2E_KITTI_ROOT` to a KITTI/YOLO layout containing
`images/{train,val}` and `labels/{train,val}`. Set `CAMERAE2E_YOLO_MODEL` to an
existing KITTI-trained checkpoint when it is stored outside the project. Local
datasets, trained weights, projects, and run artifacts are intentionally not
committed. Without a detector or labels, perception optimization is disabled;
the Workbench does not substitute a fake score or a generic COCO score.

The top-bar study selector keeps baseline and focused trade studies separate.
Successive-halving results mark final-budget candidates as `finalist`; screened
early-stage scores remain visible but cannot become the best candidate.

Finalists also carry scene-bootstrap confidence intervals and an explicit
`winner` or `indistinguishable` decision. The Dataset section reports full RAW,
RGB, label, checksum, and split-integrity validation, while Calibration shows
stage-scoped sensor, optics, ISP, and HW ISP evidence. L1 FDTD results remain
proxy evidence unless measured calibration is attached.
