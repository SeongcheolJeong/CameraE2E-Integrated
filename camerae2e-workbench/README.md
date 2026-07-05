# CameraE2E Workbench

React + FastAPI local workbench for running the bundled CameraE2E simulation APIs.

## Run

```bash
cd /Users/seongcheoljeong/Documents/CameraE2E-Integrated/camerae2e-workbench
npm install
npm run backend
npm run dev
```

The default local URLs are:

- Frontend: http://127.0.0.1:5175
- Backend: http://127.0.0.1:8010

The UI calls the real Python APIs for FACA simulation, automated parameter optimization, RAW dataset export, asset status, and report generation. Fidelity badges intentionally distinguish analytic, FDTD LUT-backed, RayOptics geometric, and TCAD calibration-required paths.

