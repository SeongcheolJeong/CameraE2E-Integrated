# HW ISP Parameter Profiles

This directory contains normalized seed profiles for `pyisetcam.hwisp`.

These files are not vendor sign-off databases. They are system-simulation inputs
that can be overridden with board-specific BSP, driver, tuning, or measured
latency values.

Use:

```python
from pyisetcam import hw_isp_config_from_profile

config = hw_isp_config_from_profile("generic_1080p_30fps")
```

Use `PYISETCAM_HWISP_DB=/path/to/profiles` to point the loader at collected or
product-specific profiles.
