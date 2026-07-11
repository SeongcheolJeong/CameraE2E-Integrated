# Research and Fidelity Boundaries

## Claim Policy

CameraE2E separates computational availability from physical validation. An asset can
execute without being calibrated, and a solver output can be numerically converged
without matching a manufactured camera.

| Evidence | Useful for | Not sufficient for |
|---|---|---|
| L0 analytic model | broad trends, sensitivity, search-space screening | pixel-stack or product sign-off |
| L1 LUT | interpolation and candidate ranking inside its valid domain | extrapolation or measured accuracy claims |
| L2 solver | shortlisted physics evidence and LUT generation | manufactured-device prediction without calibration |
| L3 calibrated | scoped quantitative comparison inside calibration domain | unmeasured stages or out-of-domain products |

## Scene and Re-capture

Synthetic spectral scenes can drive the full radiometric pipeline. A KITTI or other RGB
file has already passed through an unknown camera and ISP. CameraE2E can apply geometry
and a new camera response as a useful proxy, but cannot reconstruct lost spectra,
original photons, or source RAW uniquely. Crop/FOV transformation changes geometry;
optics, sensor, noise, CFA, and ISP stages change the simulated camera response.

## Optics and RayOptics

Analytic diffraction and aberration approximations are efficient for broad searches.
Bundled RayOptics PSFs are geometric ray histograms. They do not become diffraction or
wave-optics sign-off merely because they are used as a PSF. Compare geometric and
diffraction/WVF evidence explicitly and calibrate against measured field/wavelength PSF
or MTF for quantitative claims.

## Sensor, FDTD, and TCAD

Public sensor records and seed QE profiles support configuration and relative studies.
They are not vendor characterization. FDTD LUTs describe the modeled optical stack only
within their sampled wavelength/angle/geometry domain. TCAD collection artifacts depend
on process, doping, boundary, transport, and generation-map assumptions. FDTD and TCAD
must share run lineage; a mismatch is stale evidence, not a warning to ignore.

OCL/CRA, DTI, CFA topology, QE, binning, and pixel pitch interact. Analytic models are
appropriate for broad exploration; validated LUTs or solvers are appropriate for
shortlisted nonlinear regimes. The recommended workflow is therefore hybrid, not an
exclusive choice between analytic and FDTD.

## ISP, Color, and Hardware

A CCM may be fitted only with suitable color evidence and an independent holdout check.
Seed/public-derived CCM or ISP settings are starting points, not tuned product values.
HW ISP latency and 3A behavior require measured traces for timing sign-off.

## Perception

Detector optimization is valid only for the attached model, labels, class mapping,
scene selection, perturbations, and metric version. A generic COCO detector may provide
an engineering signal for cars and people, but it is not equivalent to a KITTI-trained
ADAS model. Image-quality metrics support diagnosis; they never replace missing mAP or
recall in a perception target.

## Approximation: Utility and Limit

Approximation is valuable because it makes thousands of consistent, differentiable or
searchable evaluations possible and exposes dominant trade-offs before expensive
measurement or solver work. Its error becomes dangerous when omitted coupling dominates,
parameters leave the calibrated domain, or a ranking difference is smaller than model
uncertainty. Promote only shortlisted decisions to higher fidelity and preserve the
lower-fidelity assumptions in every report.
