# Sensor Stack Parameters

The runnable stack is defined in:

```text
/Users/seongcheoljeong/FDTD/configs/sensor_stack_proxy_1p4um.json
```

This file records the optical geometry currently used by the Meep supercell
runner: pixel pitch, PML thickness, air gap, OCL height and aperture, CFA
thickness, passivation thickness, Si thickness, and metal edge width.

The product-specific template is:

```text
/Users/seongcheoljeong/FDTD/configs/sensor_stack_actual_template.json
```

Use the template for actual sensor data. The key parameters that must come from
the target sensor/process are:

- pixel pitch
- OCL sag/height, aperture radius, edge gap, and shift-vs-field rule
- CFA thickness, aperture, and R/G/B n,k tables
- passivation/AR stack effective n,k and thickness
- metal aperture/shield geometry and validated optical metal model
- Si active thickness, wafer optical model, and any DTI/BDTI geometry
- split-PD gap, region boundaries, and active collection region definitions

The current defaults are not calibrated product data. Treat them as executable
proxy values until measured geometry and n,k files are supplied.
