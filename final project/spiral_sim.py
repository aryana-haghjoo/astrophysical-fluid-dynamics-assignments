"""
Gas response to a spiral-arm potential in a 2D galactic disk.

BACKBONE: the MUSCL-Hancock finite-volume scheme from HW2 (kh_sim.py).
The gradient, slope-limiter, face-extrapolation and flux-update routines are
imported unchanged from kh_sim. What is new here:

  * isothermal equation of state  P = cs^2 rho  (no energy equation)
  * Rusanov flux rewritten for the isothermal system
  * source terms: gravity of a fixed galaxy potential + a rotating 2-armed
    logarithmic spiral perturbation, Coriolis and centrifugal forces
    (we work in the frame co-rotating with the spiral pattern, Omega_p)
  * damping ("sponge") zones near the centre and the box edge that hold the
    gas at its initial circular-rotation state, since the Cartesian box is
    periodic but the disk is not

Units: v0 = 1 (flat rotation speed), r in units of R0. If R0 = 10 kpc and
v0 = 220 km/s, one time unit = 44 Myr and cs = 0.05 is about 11 km/s.
"""
import numpy as np
from kh_sim import gradient, slope_limit, to_faces, apply_fluxes   # <- HW2 backbone

R, L = -1, 1


def rusanov_isothermal(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, cs):
    """Rusanov flux for the isothermal Euler equations (x-direction)."""
    def flux(rho, vx, vy):
        return rho * vx, rho * vx**2 + cs**2 * rho, rho * vx * vy
    fL = flux(rho_L, vx_L, vy_L)
    fR = flux(rho_R, vx_R, vy_R)
    C = cs + np.maximum(np.abs(vx_L), np.abs(vx_R))
    qL = (rho_L, rho_L * vx_L, rho_L * vy_L)
    qR = (rho_R, rho_R * vx_R, rho_R * vy_R)
    # same naming convention as kh_sim.getFlux: "L" is the state on the +x side
    return tuple(0.5 * (a + b) - 0.5 * C * (ql - qr) for a, b, ql, qr in zip(fL, fR, qL, qR))


class Galaxy:
    def __init__(self, N=256, Lbox=1.1, cs=0.05, F=0.03, pitch_deg=20.0, m=2,
                 Omega_p=0.9, rc=0.05, r_in=0.15, r_out=1.0, t_ramp=3.0,
                 courant=0.4, limiter=True):
        self.N, self.cs, self.F, self.m, self.Op = N, cs, F, m, Omega_p
        self.rc, self.r_in, self.r_out, self.t_ramp = rc, r_in, r_out, t_ramp
        self.courant, self.limiter = courant, limiter
        self.cot_i = 1.0 / np.tan(np.radians(pitch_deg))
        # potential amplitude: radial force of the perturbation = F x axisymmetric force
        self.eps = F / (m * self.cot_i)
        self.dx = 2 * Lbox / N
        x = np.linspace(-Lbox + 0.5 * self.dx, Lbox - 0.5 * self.dx, N)
        self.X, self.Y = np.meshgrid(x, x, indexing='ij')      # arrays indexed [ix, iy]
        self.r = np.sqrt(self.X**2 + self.Y**2)
        self.phi = np.arctan2(self.Y, self.X)
        # circular velocity of the axisymmetric potential  Phi0 = 0.5 ln(r^2 + rc^2)
        self.vc = self.r / np.sqrt(self.r**2 + rc**2)
        # initial state in the rotating frame: uniform density, circular rotation minus pattern
        vphi = self.vc - Omega_p * self.r
        self.rho0 = np.ones_like(self.r)
        self.vx0 = -vphi * np.sin(self.phi)
        self.vy0 = vphi * np.cos(self.phi)
        # sponge weight: 1 inside r < r_in and r > r_out, smooth ramp of width 0.05
        w_in = np.clip((r_in + 0.05 - self.r) / 0.05, 0, 1)
        w_out = np.clip((self.r - r_out) / 0.05, 0, 1)
        self.wsponge = np.maximum(w_in, w_out)
        # inner taper so the perturbation does not blow up at r -> 0
        self.taper = 1.0 / (1.0 + (r_in / np.maximum(self.r, 1e-6))**4)
        self.reset()

    def reset(self):
        vol = self.dx**2
        self.vol = vol
        self.Mass = self.rho0 * vol
        self.Momx = self.rho0 * self.vx0 * vol
        self.Momy = self.rho0 * self.vy0 * vol
        self.t = 0.0

    # ----- physics ---------------------------------------------------------------
    def spiral_phase(self):
        """psi = m*phi + m*cot(i)*ln r : trailing arms for counter-clockwise rotation."""
        return self.m * self.phi + self.m * self.cot_i * np.log(np.maximum(self.r, 1e-6))

    def Phi1(self, t):
        amp = self.eps * min(t / self.t_ramp, 1.0)
        return amp * np.cos(self.spiral_phase())

    def accel(self, vx, vy, t):
        X, Y, r = self.X, self.Y, np.maximum(self.r, 1e-6)
        # axisymmetric gravity: -dPhi0/dr = -r/(r^2+rc^2)
        g0 = -r / (r**2 + self.rc**2)
        ax = g0 * X / r
        ay = g0 * Y / r
        # spiral perturbation: Phi1 = A cos(psi); -grad Phi1 = A sin(psi) grad psi
        amp = self.eps * min(t / self.t_ramp, 1.0)
        psi = self.spiral_phase()
        dpsi_dr = self.m * self.cot_i / r
        dpsi_rdphi = self.m / r                     # (1/r) dpsi/dphi
        s = amp * np.sin(psi) * self.taper
        gr, gphi = s * dpsi_dr, s * dpsi_rdphi
        c, sn = X / r, Y / r
        ax += gr * c - gphi * sn
        ay += gr * sn + gphi * c
        # rotating frame (Omega_p z-hat): Coriolis -2 Omega x v, centrifugal Omega^2 r
        Op = self.Op
        ax += 2 * Op * vy + Op**2 * X
        ay += -2 * Op * vx + Op**2 * Y
        return ax, ay

    # ----- one MUSCL-Hancock step (HW2 structure) --------------------------------
    def primitives(self):
        rho = self.Mass / self.vol
        return rho, self.Momx / (rho * self.vol), self.Momy / (rho * self.vol)

    def step(self, dt_max=np.inf):
        cs, dx = self.cs, self.dx
        rho, vx, vy = self.primitives()
        dt = min(self.courant * np.min(dx / (cs + np.sqrt(vx**2 + vy**2))), dt_max)

        rho_dx, rho_dy = gradient(rho, dx)
        vx_dx, vx_dy = gradient(vx, dx)
        vy_dx, vy_dy = gradient(vy, dx)
        if self.limiter:
            rho_dx, rho_dy = slope_limit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slope_limit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slope_limit(vy, dx, vy_dx, vy_dy)

        # half-step predictor (isothermal primitive equations + half-step source terms)
        ax, ay = self.accel(vx, vy, self.t)
        rho_p = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_p = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + cs**2 * rho_dx / rho - ax)
        vy_p = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + cs**2 * rho_dy / rho - ay)

        rXL, rXR, rYL, rYR = to_faces(rho_p, rho_dx, rho_dy, dx)
        uXL, uXR, uYL, uYR = to_faces(vx_p, vx_dx, vx_dy, dx)
        wXL, wXR, wYL, wYR = to_faces(vy_p, vy_dx, vy_dy, dx)

        fMX, fMxX, fMyX = rusanov_isothermal(rXL, rXR, uXL, uXR, wXL, wXR, cs)
        fMY, fMyY, fMxY = rusanov_isothermal(rYL, rYR, wYL, wYR, uYL, uYR, cs)

        self.Mass = apply_fluxes(self.Mass, fMX, fMY, dx, dt)
        self.Momx = apply_fluxes(self.Momx, fMxX, fMxY, dx, dt)
        self.Momy = apply_fluxes(self.Momy, fMyX, fMyY, dx, dt)

        # time-centred source terms (evaluated with the predicted state)
        axp, ayp = self.accel(vx_p, vy_p, self.t + 0.5 * dt)
        self.Momx += dt * rho_p * axp * self.vol
        self.Momy += dt * rho_p * ayp * self.vol

        # sponge zones: relax toward the initial axisymmetric state (tau = 0.05)
        f = np.clip(self.wsponge * dt / 0.05, 0, 1)
        rho, vx, vy = self.primitives()
        rho = rho + f * (self.rho0 - rho)
        vx = vx + f * (self.vx0 - vx)
        vy = vy + f * (self.vy0 - vy)
        self.Mass, self.Momx, self.Momy = rho * self.vol, rho * vx * self.vol, rho * vy * self.vol
        self.t += dt
        return dt

    def run(self, tEnd, snap_times=(), verbose=False):
        snaps = {}
        targets = sorted(snap_times)
        if targets and targets[0] <= 0:
            snaps[0.0] = self.primitives()[0].copy(); targets = targets[1:]
        nstep = 0
        while self.t < tEnd - 1e-12:
            nxt = min([s for s in targets if s > self.t + 1e-12] + [tEnd])
            self.step(dt_max=nxt - self.t)
            nstep += 1
            for s in list(targets):
                if abs(self.t - s) < 1e-9:
                    snaps[s] = self.primitives()[0].copy(); targets.remove(s)
            if verbose and nstep % 1000 == 0:
                print(f'N={self.N} t={self.t:.2f}', flush=True)
        self.nstep = nstep
        return snaps

    # ----- diagnostics -------------------------------------------------------------
    def ring_profile(self, r0, nphi=720):
        """Bilinear-sample rho, normal velocity etc. on the ring r = r0."""
        rho, vx, vy = self.primitives()
        ph = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
        xs, ys = r0 * np.cos(ph), r0 * np.sin(ph)
        Lb = -self.X[0, 0] + 0.5 * self.dx
        fi = (xs + Lb) / self.dx - 0.5
        fj = (ys + Lb) / self.dx - 0.5
        i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
        ti, tj = fi - i0, fj - j0
        def samp(a):
            return ((1 - ti) * (1 - tj) * a[i0, j0] + ti * (1 - tj) * a[i0 + 1, j0] +
                    (1 - ti) * tj * a[i0, j0 + 1] + ti * tj * a[i0 + 1, j0 + 1])
        R_, VX, VY = samp(rho), samp(vx), samp(vy)
        # unit normal to the arm (along grad psi), in Cartesian components
        cot = self.cot_i
        nr, nph = cot / np.hypot(cot, 1), 1 / np.hypot(cot, 1)
        nx = nr * np.cos(ph) - nph * np.sin(ph)
        ny = nr * np.sin(ph) + nph * np.cos(ph)
        un = VX * nx + VY * ny
        psi = self.m * ph + self.m * cot * np.log(r0)
        Phi1 = self.eps * np.cos(psi)
        return dict(phi=ph, rho=R_, un=un, Phi1=Phi1, psi=np.mod(psi, 2 * np.pi))
