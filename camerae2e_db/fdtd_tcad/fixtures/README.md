# FDTD/TCAD CameraE2E Fixtures

These small fixtures are copied from the local FDTD workspace so CameraE2E can
exercise LUT, TCAD generation-map, DEVSIM summary, and accuracy-gate loaders
without requiring the full `runs/` directory.

Source paths at import time:

- `fdtd_active_lut/camera_lut.json`: `/Users/seongcheoljeong/FDTD/runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json`
- `tcad_generation_map/tcad_generation_map_2d.npz`: `/Users/seongcheoljeong/FDTD/runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz`
- `devsim_split_pd_center/summary.json`: `/Users/seongcheoljeong/FDTD/runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json`
- `devsim_split_pd_edge/summary.json`: `/Users/seongcheoljeong/FDTD/runs/devsim_split_pd_2d_fdtd_map_proxy_edge20x_smoke/summary.json`
- `tcad_accuracy_gate/tcad_accuracy_gate.json`: `/Users/seongcheoljeong/FDTD/runs/tcad_accuracy_gate_reference_profile/tcad_accuracy_gate.json`

Important: these fixtures intentionally preserve the current active-lineage
mismatch. The FDTD LUT and TCAD generation map are not from the same run root.
CameraE2E must keep this as `stale_dependency` until a generation map is
regenerated from the active FDTD LUT or both are pointed to one validated run.
