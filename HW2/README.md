# HW2: Kelvin–Helmholtz finite-volume simulation

## Overview
A 2D compressible Euler solver (MUSCL–Hancock with a Rusanov flux) that simulates the Kelvin–Helmholtz instability in a periodic unit box. The code is in `kh_sim.py`. The notebook `KH_finite_volume.ipynb` shows the evolution (movie and snapshots), the resolution and slope-limiter comparison, and a growth-rate test against linear theory (measured 4.56 at N=512 vs. theory 5.92).

## Files
| File | What it is |
|---|---|
| `KH_finite_volume.ipynb` | Main notebook: method, initial conditions, movie, snapshots, growth-rate test, conservation check |
| `kh_sim.py` | The solver (MUSCL–Hancock + Rusanov flux). The notebook imports it, so keep it in this folder |
| `results.pkl` | Saved results of the long runs (N = 64–512). The notebook loads this instead of rerunning them |
| `experiments.py` | Script that produced `results.pkl` (~25 min) |
| `figures.py` | Script that made `fig_density.png` and `fig_growth.png` |
| `kh_movie.mp4` / `kh_movie.gif` | The movie (regenerated when you run the notebook) |

## Knobs worth playing with
- `RECOMPUTE = True` (first code cell) redoes all the N=64–512 runs instead of loading `results.pkl`. Takes ~25 min.
- Section 3 movie cell: `kh_sim.run(N=256, ...)`. Try `N=128` for a fast (~10 s) run, or `N=512` for a pretty one (~8 min).
- `kh_sim.run(...)` arguments:
  - `N`: resolution
  - `tEnd`: end time
  - `w0`: seed amplitude
  - `limiter=True`: slope limiter on
  - `courant`: CFL factor (default 0.4; try 1.2 to watch it go unstable)
  - `snap_times`: times at which to save density snapshots

## How the code works (one paragraph)
Each cell stores the averages of mass, x-momentum, y-momentum and energy. The ideal-gas equation of state closes the system. Every step:
1. convert to ρ, v, P;
2. pick Δt from the CFL condition;
3. estimate linear slopes in each cell;
4. predict a half time step;
5. extrapolate to the cell faces;
6. compute the flux through each face with the Rusanov Riemann solver;
7. update each cell by flux in minus flux out.

Because of step 7, mass, momentum and energy are conserved to round-off. The diffusion term in the Rusanov flux is the main error: it is why the measured KH growth rate (4.56 at N=512) sits below linear theory (5.92).

Code adapted from P. Mocz, https://github.com/pmocz/finitevolume-python
