"""
Kelvin-Helmholtz instability with a 2D finite-volume (MUSCL-Hancock) Euler solver.

Adapted from Philip Mocz (2020), "Create Your Own Finite Volume Fluid Simulation
(With Python)", https://github.com/pmocz/finitevolume-python
Changes from the original: packaged as a function, optional seed amplitude,
diagnostics (KH mode amplitude, conservation), and snapshot output.

Scheme, per time step:
  1. conserved -> primitive variables (rho, vx, vy, P)
  2. CFL time step  dt = C * min( dx / (c_s + |v|) )
  3. cell-centred gradients (central differences), optionally slope-limited
  4. predictor: advance primitives by dt/2 using the primitive Euler equations
  5. extrapolate to cell faces (piecewise-linear reconstruction)
  6. Rusanov (local Lax-Friedrichs) flux at each face
  7. conservative update: Q -= dt * dx * (flux_out - flux_in)
Periodic boundaries in x and y (np.roll).
"""
import numpy as np

R, L = -1, 1  # np.roll directions: right / left neighbour


def get_conserved(rho, vx, vy, P, gamma, vol):
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)) * vol
    return Mass, Momx, Momy, Energy


def get_primitive(Mass, Momx, Momy, Energy, gamma, vol):
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2)) * (gamma - 1)
    return rho, vx, vy, P


def gradient(f, dx):
    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2 * dx)
    return f_dx, f_dy


def slope_limit(f, dx, f_dx, f_dy):
    """Limit slopes so the reconstruction creates no new extrema (minmod-like)."""
    eps = lambda g: g + 1.0e-8 * (g == 0)
    f_dx = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=0)) / dx) / eps(f_dx))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=0)) / dx) / eps(f_dx))) * f_dx
    f_dy = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=1)) / dx) / eps(f_dy))) * f_dy
    f_dy = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=1)) / dx) / eps(f_dy))) * f_dy
    return f_dx, f_dy


def to_faces(f, f_dx, f_dy, dx):
    """Piecewise-linear extrapolation to the left/right states at each face."""
    f_XL = np.roll(f - f_dx * dx / 2, R, axis=0)
    f_XR = f + f_dx * dx / 2
    f_YL = np.roll(f - f_dy * dx / 2, R, axis=1)
    f_YR = f + f_dy * dx / 2
    return f_XL, f_XR, f_YL, f_YR


def apply_fluxes(F, fX, fY, dx, dt):
    return F - dt * dx * (fX - np.roll(fX, L, axis=0)) - dt * dx * (fY - np.roll(fY, L, axis=1))


def rusanov_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Local Lax-Friedrichs flux in the x-direction (swap vx/vy for y)."""
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2)
    rho_s = 0.5 * (rho_L + rho_R)
    mx_s = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    my_s = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_s = 0.5 * (en_L + en_R)
    P_s = (gamma - 1) * (en_s - 0.5 * (mx_s**2 + my_s**2) / rho_s)
    fM = mx_s
    fMx = mx_s**2 / rho_s + P_s
    fMy = mx_s * my_s / rho_s
    fE = (en_s + P_s) * mx_s / rho_s
    # maximum local signal speed -> numerical diffusion
    C = np.maximum(np.sqrt(gamma * P_L / rho_L) + np.abs(vx_L),
                   np.sqrt(gamma * P_R / rho_R) + np.abs(vx_R))
    fM = fM - C * 0.5 * (rho_L - rho_R)
    fMx = fMx - C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    fMy = fMy - C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    fE = fE - C * 0.5 * (en_L - en_R)
    return fM, fMx, fMy, fE


def mode_amplitude(vy, kx_index=2):
    """Amplitude of the sin(2*pi*kx_index*x) mode of vy, maximised over y."""
    N = vy.shape[0]
    ft = np.fft.rfft(vy, axis=0)  # along x
    return np.max(2 * np.abs(ft[kx_index, :]) / N)


def run(N=128, tEnd=2.0, w0=0.1, limiter=False, courant=0.4, gamma=5/3,
        snap_times=(), verbose=False):
    dx = 1.0 / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, 1 - 0.5 * dx, N)
    Y, X = np.meshgrid(xlin, xlin)  # arrays indexed [ix, iy]

    # Initial conditions: dense (rho=2) stream moving +x between y=0.25 and 0.75,
    # light (rho=1) stream moving -x outside -> two shear interfaces.
    sigma = 0.05 / np.sqrt(2.0)
    band = (np.abs(Y - 0.5) < 0.25)
    rho = 1.0 + band
    vx = -0.5 + band
    vy = w0 * np.sin(4 * np.pi * X) * (np.exp(-(Y - 0.25)**2 / (2 * sigma**2)) +
                                       np.exp(-(Y - 0.75)**2 / (2 * sigma**2)))
    P = 2.5 * np.ones_like(X)

    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, gamma, vol)
    tot0 = np.array([Mass.sum(), Momx.sum(), Momy.sum(), Energy.sum()])

    snaps = {}
    snap_times = sorted(snap_times)
    if snap_times and snap_times[0] == 0:
        snaps[0.0] = rho.copy(); snap_times = snap_times[1:]
    hist_t, hist_A = [0.0], [mode_amplitude(vy)]
    t, nstep = 0.0, 0
    while t < tEnd - 1e-12:
        rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, gamma, vol)
        dt = courant * np.min(dx / (np.sqrt(gamma * P / rho) + np.sqrt(vx**2 + vy**2)))
        # land exactly on snapshot / end times
        targets = [s for s in snap_times if s > t + 1e-12] + [tEnd]
        dt = min(dt, min(targets) - t)

        rho_dx, rho_dy = gradient(rho, dx)
        vx_dx, vx_dy = gradient(vx, dx)
        vy_dx, vy_dy = gradient(vy, dx)
        P_dx, P_dy = gradient(P, dx)
        if limiter:
            rho_dx, rho_dy = slope_limit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slope_limit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slope_limit(vy, dx, vy_dx, vy_dy)
            P_dx, P_dy = slope_limit(P, dx, P_dx, P_dy)

        # half-step predictor (primitive-variable Euler equations)
        rho_p = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_p = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + P_dx / rho)
        vy_p = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + P_dy / rho)
        P_p = P - 0.5 * dt * (gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

        rXL, rXR, rYL, rYR = to_faces(rho_p, rho_dx, rho_dy, dx)
        uXL, uXR, uYL, uYR = to_faces(vx_p, vx_dx, vx_dy, dx)
        wXL, wXR, wYL, wYR = to_faces(vy_p, vy_dx, vy_dy, dx)
        pXL, pXR, pYL, pYR = to_faces(P_p, P_dx, P_dy, dx)

        fMX, fMxX, fMyX, fEX = rusanov_flux(rXL, rXR, uXL, uXR, wXL, wXR, pXL, pXR, gamma)
        fMY, fMyY, fMxY, fEY = rusanov_flux(rYL, rYR, wYL, wYR, uYL, uYR, pYL, pYR, gamma)

        Mass = apply_fluxes(Mass, fMX, fMY, dx, dt)
        Momx = apply_fluxes(Momx, fMxX, fMxY, dx, dt)
        Momy = apply_fluxes(Momy, fMyX, fMyY, dx, dt)
        Energy = apply_fluxes(Energy, fEX, fEY, dx, dt)
        t += dt; nstep += 1

        rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, gamma, vol)
        hist_t.append(t); hist_A.append(mode_amplitude(vy))
        for s in list(snap_times):
            if abs(t - s) < 1e-9:
                snaps[s] = rho.copy(); snap_times.remove(s)
        if verbose and nstep % 500 == 0:
            print(f"N={N} step {nstep} t={t:.3f}", flush=True)

    tot1 = np.array([Mass.sum(), Momx.sum(), Momy.sum(), Energy.sum()])
    return dict(rho=rho, snaps=snaps, t=np.array(hist_t), A=np.array(hist_A),
                tot0=tot0, tot1=tot1, nstep=nstep)
