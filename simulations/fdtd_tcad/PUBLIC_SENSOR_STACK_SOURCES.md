# Public Sensor-Stack Source Notes

This file records internet/public-source data reflected in the runnable proxy
stack. It does not make the stack product-calibrated.

## Reflected In Config

Config:

```text
/Users/seongcheoljeong/FDTD/configs/sensor_stack_proxy_1p4um.json
```

Reflected values:

- CFA film thickness: `0.80 um`
- OCL/microlens height: `0.657 um`
- CFA refractive-index anchors:
  - red: `n=1.59 @ 620 nm`
  - green: `n=1.60 @ 550 nm`
  - blue: `n=1.61 @ 450 nm`
- OCL polymer refractive-index anchor: `n=1.61`
- BSI-style effective passivation thickness: `0.08 um`
- Si optical thickness default: `2.80 um`

## Sources

- US9657182B2 reports an image-sensor-chip example with `0.8 um` color
  filters, `0.3 um` upper planarization, `0.657 um` microlens height, and
  measured R/G/B and microlens refractive indices.
  https://patents.google.com/patent/US9657182B2/en

- US20110101482A1 reports BSI backside Al2O3 passivation and gives a
  passivation thickness range of `1 nm` to `150 nm`.
  https://patents.google.com/patent/US20110101482A1/en

- IISW 2023 Trends and Developments in State-of-the-Art CMOS Image Sensors
  reports recent small-pixel trends, CFA grouping, aperture grids, and DTI depth
  examples. The proxy Si optical thickness uses `2.8 um` as a public-reference
  small-pixel anchor, not as a target-product value.
  https://www.imagesensors.org/Past%20Workshops/2023%20Workshop/2023%20Papers/R1.pdf

- Evolution of Optical Structure in Image Sensors reports BSI as the mainstream
  approach for sub-1.5 um pixels and gives example lightpipe material indices
  (`SiN n=2.05`, organic fill `n=1.6`).
  https://ptacts.uspto.gov/ptacts/public-informations/petitions/1550798/download-documents?artifactId=QFQI_f_G1F8TAPlesGyKpsIkr4F74yrtVmAL5q5KxSt5CObAqeEpCMs

- Meep documentation recommends keeping flux monitors out of PML, using enough
  post-source time for Fourier-transform convergence, supports
  `stop_when_fields_decayed`, and documents that flux signs follow monitor
  orientation conventions. The LUT runner now exposes `--decay-by` and
  `--decay-check-time`; convergence reports treat negative signed flux as a
  warning unless strict mode is enabled.
  https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/

## Not Publicly Solved

The following remain target-sensor inputs:

- actual OCL sag/profile and field-dependent OCL/CFA/PD shift law
- measured R/G/B CFA `n,k(lambda)` over the full camera spectral range
- actual passivation/AR multilayer stack, not a single effective layer
- DTI/BDTI/aperture-grid 3D geometry and material model
- calibrated metal model
- TCAD collection-efficiency map

The current public-source stack is better grounded than the initial pure proxy,
but it is still not an accuracy claim.
