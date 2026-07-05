# n,k Tables

Tables use three columns:

```text
wavelength_um,n,k
```

`../Si-Green-2008.yml` is public silicon data from the refractiveindex.info
database. The CFA and OCL polymer files in this directory are anchored to
published image-sensor-chip example refractive-index values where available, but
their full dispersion and CFA extinction coefficients remain proxy values. They
are intentionally separated from the code so measured product data can replace
them without changing the solver.

For quantitative camera-system LUTs, replace the proxy files with measured or
process-deck n,k tables over the required wavelength range.
