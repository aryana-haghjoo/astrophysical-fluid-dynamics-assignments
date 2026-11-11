"""Verification: isothermal colliding flows, run with the SAME isothermal MUSCL-Hancock
update as the galaxy code (kh_sim helpers + rusanov_isothermal).
Two streams rho=1, v=+-U collide at x=0, launching two shocks. Exact solution:
gas between the shocks is at rest with density rho2, and the shocks move at Vs, where
  mass flux:            rho2 * Vs = (U + Vs)        (shock frame, rho1 = 1)
  isothermal jump:      rho2 = M1^2,  M1 = (U + Vs)/cs     (A.13 with gamma -> 1)
=> Vs = (-U + sqrt(U^2 + 4 cs^2))/2 ,  rho2 = ((U+Vs)/cs)^2."""
import numpy as np
from kh_sim import gradient, slope_limit, to_faces, apply_fluxes
from spiral_sim import rusanov_isothermal

def exact(U, cs):
    Vs = 0.5 * (-U + np.sqrt(U**2 + 4 * cs**2))
    return Vs, ((U + Vs) / cs)**2

def run(U=0.5, cs=0.05, N=512, tEnd=0.5, limiter=True, courant=0.4):
    Ny = 4
    dx = 1.0 / N
    x = np.linspace(-0.5 + 0.5*dx, 0.5 - 0.5*dx, N)
    X = np.repeat(x[:, None], Ny, axis=1)
    rho = np.ones_like(X); vx = np.where(X < 0, U, -U); vy = np.zeros_like(X)
    vol = dx**2
    Mass, Momx, Momy = rho*vol, rho*vx*vol, rho*vy*vol
    t = 0
    while t < tEnd - 1e-12:
        rho = Mass/vol; vx = Momx/Mass; vy = Momy/Mass
        dt = min(courant*np.min(dx/(cs + np.abs(vx) + np.abs(vy))), tEnd - t)
        r_dx, r_dy = gradient(rho, dx); u_dx, u_dy = gradient(vx, dx); w_dx, w_dy = gradient(vy, dx)
        if limiter:
            r_dx, r_dy = slope_limit(rho, dx, r_dx, r_dy); u_dx, u_dy = slope_limit(vx, dx, u_dx, u_dy)
            w_dx, w_dy = slope_limit(vy, dx, w_dx, w_dy)
        rp = rho - 0.5*dt*(vx*r_dx + rho*u_dx + vy*r_dy + rho*w_dy)
        up = vx - 0.5*dt*(vx*u_dx + vy*u_dy + cs**2*r_dx/rho)
        wp = vy - 0.5*dt*(vx*w_dx + vy*w_dy + cs**2*r_dy/rho)
        rXL, rXR, rYL, rYR = to_faces(rp, r_dx, r_dy, dx)
        uXL, uXR, uYL, uYR = to_faces(up, u_dx, u_dy, dx)
        wXL, wXR, wYL, wYR = to_faces(wp, w_dx, w_dy, dx)
        fM, fMx, fMy = rusanov_isothermal(rXL, rXR, uXL, uXR, wXL, wXR, cs)
        gM, gMy, gMx = rusanov_isothermal(rYL, rYR, wYL, wYR, uYL, uYR, cs)
        Mass = apply_fluxes(Mass, fM, gM, dx, dt); Momx = apply_fluxes(Momx, fMx, gMx, dx, dt)
        Momy = apply_fluxes(Momy, fMy, gMy, dx, dt)
        t += dt
    return x, (Mass/vol)[:, 0], (Momx/Mass)[:, 0]
