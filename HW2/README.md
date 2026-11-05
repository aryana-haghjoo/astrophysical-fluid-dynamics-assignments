# HW2: Kelvin–Helmholtz finite-volume simulation

## Files
| File | What it is |
|---|---|
| `KH_finite_volume.ipynb` | Main notebook: method, initial conditions, movie, snapshots, growth-rate test, conservation check |
| `kh_sim.py` | The solver (MUSCL–Hancock + Rusanov flux). The notebook imports it, so keep it in this folder |
| `results.pkl` | Saved results of the long runs (N = 64–512). The notebook loads this instead of rerunning them |
| `experiments.py` | Script that produced `results.pkl` (~25 min) |
| `figures.py` | Script that made `fig_density.png` and `fig_growth.png` for the report |
| `AFD_HW2_KH_report.pdf` / `report.tex` | The 2-page report and its LaTeX source |
| `kh_movie.mp4` / `kh_movie.gif` | The movie (regenerated when you run the notebook) |

## Run it in VS Code
1. **Open the folder.** File → Open Folder… → this `HW2` folder. Open it as the workspace, not just the notebook file, so `import kh_sim` and `results.pkl` are found.
2. **Extensions.** Install **Python** and **Jupyter** (both by Microsoft) if you don't have them.
3. **Make an environment** (one time), in VS Code's terminal (Terminal → New Terminal):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Conda works too: `conda create -n kh python numpy matplotlib pillow ipykernel`.
4. **ffmpeg** (for the MP4 movie): `brew install ffmpeg`. If you skip this, see Troubleshooting.
5. **Pick the kernel.** Open `KH_finite_volume.ipynb`, click **Select Kernel** (top right), and choose the `.venv` (or `kh`) Python.
6. **Run All** (▶▶ at the top). The whole notebook takes **~1.5 min**. Almost all of that is the N=256 movie run in Section 3.

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

## Troubleshooting
- **`ModuleNotFoundError: kh_sim`**: VS Code is running from a different directory. Open the HW2 *folder* as the workspace, or add `import sys; sys.path.insert(0, '.')` at the top.
- **`FileNotFoundError: results.pkl`**: same cause as above, or set `RECOMPUTE = True`.
- **ffmpeg error in the movie cell**: you don't have ffmpeg. Either `brew install ffmpeg`, or delete the `anim.save('kh_movie.mp4', ...)` line and replace the last line `HTML(anim.to_html5_video())` with `HTML(anim.to_jshtml())`. That gives an interactive player with no ffmpeg needed.
- **Movie doesn't show in VS Code's viewer**: the GIF cell right below it always works.
- **iCloud**: if VS Code says a file is missing, it may not be downloaded yet. In Finder, right-click the HW2 folder → *Download Now*.

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
