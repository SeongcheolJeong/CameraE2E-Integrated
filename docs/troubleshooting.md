# Troubleshooting

## Run the Preflight First

```bash
./tools/workbench.sh status
./tools/workbench.sh check
```

Runtime logs are `.camerae2e/backend.log` and `.camerae2e/frontend.log`.

## Workbench Does Not Open

- Confirm Python 3.12 and Node.js 20 or newer are available.
- Run `./tools/workbench.sh bootstrap` after a clean clone.
- If a PID file is stale, `./tools/workbench.sh stop` cleans it before restart.
- Ports 8010 and 5175 must be available. The launcher reports the owning process when
  either port is already occupied.

## Internal Server Error

Inspect `.camerae2e/backend.log`, then open `http://127.0.0.1:8010/docs`. Common causes
are an invalid project database, missing external assets, and a model/dataset path that
the backend process cannot read. Restart after changing environment variables.

## Perception Target Is Blocked

This is expected when either the detector or labels are unavailable or preflight fails.

1. Set `CAMERAE2E_YOLO_MODEL` to a readable checkpoint.
2. Set `CAMERAE2E_KITTI_ROOT` to matching images and labels.
3. Restart the backend.
4. Inspect asset readiness and benchmark preflight in the Workbench.

CameraE2E does not replace missing detector metrics with a proxy score.

## Scene Preview or Output Does Not Change

- `Run Simulation` reruns the current configuration; unchanged inputs and seed should
  produce the same output.
- Change a relevant parameter and confirm a new job/config revision is shown.
- Scene input is the source and must not contain detector boxes. Labels belong in the
  overlay/result view.
- Browser caching is not simulation evidence; compare artifact hashes and job IDs.

## KITTI Images or Labels Are Missing

- Image and label filenames must share the same stem.
- Ensure the selected split exists under `images/` and `labels/`.
- A KITTI RGB image is a `display_rgb_proxy`; it is not original sensor RAW.
- Do not commit KITTI data to this repository. Check its license separately.

## FDTD, TCAD, or RayOptics Is Unavailable

Run `./tools/workbench.sh check`. Repository defaults must pass the integrated asset
check. A custom asset must have valid provenance, dependency lineage, units, domain,
and validation metadata. Missing L1 data may use recorded L0 fallback only when the
study permits proxy fallback.

## Results Cannot Be Marked Calibrated

This is correct when measured QE, PTC/noise, angular response, lens MTF/PSF, color chart,
or hardware timing evidence is absent. Attach measured evidence for the specific stage;
do not manually edit the readiness badge or report wording.

## Clean Reset of Local Runtime State

Stop the servers first. Projects contain experiment evidence, so back them up before
removing anything.

```bash
./tools/workbench.sh stop
rm -rf .camerae2e
```

Do not remove `camerae2e-workbench/projects/` unless project loss is intentional.
