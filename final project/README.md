# Final project: spiral arms as shocks in a galactic gas disk

**Backbone:** the HW2 finite-volume solver (`kh_sim.py`, copied here unchanged). `spiral_sim.py` imports its numerical routines and adds the galaxy physics.

| File | What it is |
|---|---|
| `Spiral_Arms.ipynb` | Results notebook, with the movie, all figures and the analysis. Already executed, so you can just open it |
| `spiral_sim.py` | Galaxy model: isothermal gas, spiral potential, rotating frame, sponge zones |
| `kh_sim.py` | HW2 solver (the backbone: gradients, slope limiter, face reconstruction, flux update) |
| `shock_test.py` | Verification: isothermal colliding flows vs the exact solution D = M² |
| `shock_analysis.py` | Helper for cutting profiles across arms |
| `experiments.py` | All production runs (≈30 min) → `results_spiral.pkl.gz` |
| `figures.py` | Makes every `fig_*.png` from `results_spiral.pkl.gz` |
| `results_spiral.pkl.gz` | Saved results (compressed, ~20 MB). The notebook loads this instead of rerunning |
| `spiral_movie.mp4/.gif` | Movie of the fiducial run |
| `report/` | Project report: `main.tex` (LaTeX source, uses the `fig_*.png` here), the compiled PDF, and a ready-to-upload Overleaf zip |

## Run it (VS Code)
Use the same environment as HW2 (`numpy matplotlib pillow ipykernel`, plus ffmpeg for the mp4). Open this folder, select the kernel, and Run All. That takes about 30 s. Section 3 has a live 10-second run you can modify.

## Fiducial parameters
| Parameter | Value |
|---|---|
| Grid | N = 256² on [−1.1, 1.1]² |
| Gas | isothermal, c_s = 0.05 (≈11 km/s) |
| Rotation curve | flat (v0 = 1, core r_c = 0.05) |
| Spiral | 2-armed log spiral, pitch 20°, forcing F = 5% |
| Pattern speed | Ω_p = 0.9 (ILR at r = 0.33, corotation at r = 1.11) |
| Gas disk | 0.15 < r < 1.0 (sponge zones outside) |

Units: 10 kpc, 220 km/s, 44 Myr.
