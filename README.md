# StrikeCEM

**StrikeCEM** (Strike Computational Electromagnetics) is a research and simulation framework for **computational
electromagnetics (CEM)**, with applications in radar signature analysis and physics-based
modeling.

The project focuses on, high-performance workflows for electromagnetic scattering problems, radar cross
section (RCS) analysis.

---

## Features

* Schema system for configuration and experiment reproducibility
* Electromagnetic simulation solvers:
    * Physical Optics (PO)
    * Method of Moments (MoM)
    * Hybrid (PO + MoM)
* GPU acceleration with CUDA
* Advanced meshing engine:
    * Adaptive mesh refinement
    * Multi-resolution meshing
    * Facet sizing based on wavelength (λ/10, λ/20, etc.)

---

## Schemas

StrikeCEM uses JSON Schema for defining simulation configuration.
* [config.schema.json](schemas/config.schema.json) – core simulation setup.
---

## License
StrikeCEM is released under the **GNU LGPL-3.0** license.  
You are free to use, modify, and redistribute under the terms of the LGPL, while keeping modifications to StrikeCEM itself open-source.  
See the [LICENSE](../StrikeCEM/LICENSE) file for the full text.
