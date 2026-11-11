import numpy as np

def _interp(g, a, xs, ys):
    Lb = -g.X[0, 0] + 0.5 * g.dx
    fi = (xs + Lb) / g.dx - 0.5; fj = (ys + Lb) / g.dx - 0.5
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    ti, tj = fi - i0, fj - j0
    return ((1-ti)*(1-tj)*a[i0, j0] + ti*(1-tj)*a[i0+1, j0] + (1-ti)*tj*a[i0, j0+1] + ti*tj*a[i0+1, j0+1])

def shock_cut(g, r0, half=0.12, ns=400):
    """Find the arm shock on the ring r=r0 and sample a 1D cut along the local shock normal.
    Returns s (distance along normal, flow toward +s), rho(s), u_n(s), u_t(s), the jump D and
    upstream normal Mach number M1."""
    rho, vx, vy = g.primitives()
    p = g.ring_profile(r0, 4096)
    d = np.roll(p['rho'], -1) - p['rho']
    i = d.argmax(); ph = p['phi'][i]
    x0, y0 = r0*np.cos(ph), r0*np.sin(ph)
    gx = (np.roll(rho, -1, 0) - np.roll(rho, 1, 0)) / (2*g.dx)
    gy = (np.roll(rho, -1, 1) - np.roll(rho, 1, 1)) / (2*g.dx)
    nx, ny = _interp(g, gx, np.array([x0]), np.array([y0]))[0], _interp(g, gy, np.array([x0]), np.array([y0]))[0]
    nrm = np.hypot(nx, ny); nx, ny = nx/nrm, ny/nrm
    # orient normal along the flow (pre-shock gas moves into the shock)
    vx0, vy0 = _interp(g, vx, np.array([x0]), np.array([y0]))[0], _interp(g, vy, np.array([x0]), np.array([y0]))[0]
    if vx0*nx + vy0*ny < 0:
        nx, ny = -nx, -ny
    # density gradient points from low (upstream) to high (downstream) ... so flow is along -grad? check:
    s = np.linspace(-half, half, ns)
    xs, ys = x0 + s*nx, y0 + s*ny
    R_ = _interp(g, rho, xs, ys); VX = _interp(g, vx, xs, ys); VY = _interp(g, vy, xs, ys)
    un = VX*nx + VY*ny; ut = -VX*ny + VY*nx
    up = s < 0; dn = s > 0
    iu = np.where(up)[0][np.argmin(R_[up])]
    idn = np.where(dn)[0][np.argmax(R_[dn])]
    return dict(s=s, rho=R_, un=un, ut=ut, x0=x0, y0=y0, n=(nx, ny),
                rho1=R_[iu], rho2=R_[idn], u1=un[iu], u2=un[idn],
                M1=un[iu]/g.cs, M2=un[idn]/g.cs, D=R_[idn]/R_[iu])


def shock_jump(g, r0, half=0.15, ns=600, frac=0.1):
    """Measure the isothermal shock jump at radius r0.
    1. locate the shock on the ring and take a cut along grad(rho);
    2. pick the upstream point (just before the rise, where rho = rho_min + frac*jump)
       and the downstream point (density peak);
    3. refine the shock normal so the tangential velocity is equal at both points
       (tangential momentum conservation, as in A.14);
    4. return the upstream normal Mach number M1 and the density jump D.
    For an isothermal shock the jump conditions give D = M1^2 exactly."""
    rho, vx, vy = g.primitives()
    c = shock_cut(g, r0, half=half, ns=ns)
    s, R_ = c['s'], c['rho']
    ipk = np.where(s > 0)[0][np.argmax(R_[s > 0])]
    up = np.where((s < 0))[0]
    imin = up[np.argmin(R_[up])]
    thr = R_[imin] + frac * (R_[ipk] - R_[imin])
    # last upstream point (before the peak) still below threshold
    cand = np.where((np.arange(len(s)) < ipk) & (R_ < thr))[0]
    iu = cand[-1]
    nx, ny = c['n']
    xu, yu = c['x0'] + s[iu] * nx, c['y0'] + s[iu] * ny
    xd, yd = c['x0'] + s[ipk] * nx, c['y0'] + s[ipk] * ny
    v1 = np.array([_interp(g, vx, np.array([xu]), np.array([yu]))[0], _interp(g, vy, np.array([xu]), np.array([yu]))[0]])
    v2 = np.array([_interp(g, vx, np.array([xd]), np.array([yd]))[0], _interp(g, vy, np.array([xd]), np.array([yd]))[0]])
    # tangential continuity: (v1 - v2) parallel to n  ->  n = (v1 - v2)/|v1 - v2|
    dv = v1 - v2
    n = dv / np.linalg.norm(dv)
    u1, u2 = v1 @ n, v2 @ n
    rho1, rho2 = R_[iu], R_[ipk]
    ang = np.degrees(np.arccos(abs(n @ np.array([nx, ny]))))
    return dict(M1=u1 / g.cs, M2=u2 / g.cs, D=rho2 / rho1, rho1=rho1, rho2=rho2, u1=u1, u2=u2,
                flux_ratio=(rho2 * u2) / (rho1 * u1), dtheta=ang, cut=c, iu=iu, ipk=ipk)
