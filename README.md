# Going the short way – Path Optimization for the nSRT

This repository contains the Python implementation and simulation framework for my **Bachelor’s Thesis**:

**Title:** *Going the short way – Optimization of Slewing Paths for the new Small Radio Telescope (nSRT)*  
**Author:** Simon Wimmer  
**Supervision:** Armin Luntzer, BSc MSc  
**Institution:** University of Vienna, Department of Astrophysics  
**Date:** 2025
---

## Contact
For questions regarding this repository or the thesis, feel free to contact me via email:  
**simonwimmer427@gmail.com**

---

## Abstract
The new Small Radio Telescope (nSRT) at the University of Vienna operates with an azimuth-elevation (AZ/EL) mount.  
While the mechanics allow extended degrees of freedom — including over-/underrotation in azimuth and flip-over maneuvers in elevation — the standard control software does not yet make use of these features, often resulting in long slewing times.

In this thesis, several motion strategies were implemented and compared:
1. **Classical AZ/EL interpolation** (current software behavior)  
2. **Extended AZ/EL method** with over-/underrotation and flip-over maneuvers  
3. **SLERP (spherical linear interpolation)** as a theoretical benchmark  

Simulations based on random and critical waypoints show that the extended method reduces average slewing times by ~30% and halves the worst-case times near the zenith. The logic was successfully integrated into the MD01 control plugin, enabling the telescope to fully exploit its mechanical potential.

---

## Repository Contents
- `MAIN.ipynb` – Jupyter notebook with simulation code and plots  
- `figures/` – Example figures generated during the analysis  
- Python modules for coordinate transforms, path generation, and runtime calculations  

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/S-Wimmer1/Going-the-short-way.git
   ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Open the notebook:
   ```
   jupyter notebook MAIN.ipynb
   ```

---

## Acknowledgements
Special thanks to Armin Luntzer for supervision and integration of the developed logic into the real control software. (https://github.com/aluntzer/radtel)
