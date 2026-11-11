"""All runs for the spiral-arm project. Results -> results_spiral.pkl.gz (~5 min at N<=256, +12 min for N=512)."""
import numpy as np, pickle, time
import spiral_sim as S
import shock_test as ST

T0 = time.time()
out = {}
RADII = np.round(np.arange(0.25, 0.96, 0.05), 3)

def ring_stats(g):
    st = {}
    for r0 in RADII:
        p = g.ring_profile(r0, 2048)
        st[r0] = dict(max=p['rho'].max(), min=p['rho'].min(), phi=p['phi'], rho=p['rho'],
                      un=p['un'], Phi1=p['Phi1'])
    return st

# 1. fiducial run: movie frames + contrast time series
g = S.Galaxy(N=256, F=0.05)
frames, tser, contrast = {}, [], []
for t in np.round(np.arange(0, 20.0001, 0.2), 6):
    if t > 0:
        g.run(t)
    frames[t] = g.primitives()[0].astype(np.float32)
    p = g.ring_profile(0.45, 1024)
    tser.append(t); contrast.append(p['rho'].max() / p['rho'].min())
out['fid'] = dict(frames=frames, t=np.array(tser), contrast=np.array(contrast),
                  rings=ring_stats(g), vel=[a.astype(np.float32) for a in g.primitives()[1:]],
                  params=dict(N=256, F=0.05, cs=0.05, pitch=20, Omega_p=0.9, m=2))
print('fid', time.time() - T0, flush=True)

# 2. forcing-amplitude scan
for F in (0.005, 0.01, 0.02, 0.05, 0.1):
    gg = S.Galaxy(N=256, F=F); gg.run(15.0)
    out[('F', F)] = dict(rho=gg.primitives()[0].astype(np.float32), rings=ring_stats(gg))
    print('F', F, time.time() - T0, flush=True)

# 3. sound-speed scan (gas temperature)
for cs in (0.025, 0.05, 0.1):
    gg = S.Galaxy(N=256, F=0.05, cs=cs); gg.run(15.0)
    out[('cs', cs)] = dict(rho=gg.primitives()[0].astype(np.float32), rings=ring_stats(gg))
    print('cs', cs, time.time() - T0, flush=True)

# 4. resolution
for N in (128, 256, 512):
    gg = S.Galaxy(N=N, F=0.05); gg.run(15.0)
    out[('N', N)] = dict(rho=gg.primitives()[0].astype(np.float32), rings=ring_stats(gg))
    print('N', N, time.time() - T0, flush=True)

# 5. 1D colliding-flow shock verification
ver = []
for U in (0.05, 0.1, 0.2, 0.3):
    Vs, r2 = ST.exact(U, 0.05); T = 0.3 / (U + 0.05)
    x, rho, v = ST.run(U=U, N=512, tEnd=T)
    m = np.abs(x) < 0.6 * Vs * T
    ver.append(dict(U=U, M1=(U + Vs) / 0.05, rho2_exact=r2, rho2_sim=rho[m].mean(), x=x, rho=rho, T=T, Vs=Vs))
out['shocktest'] = ver
out['radii'] = RADII
import gzip
out['fid']['frames'] = {k: v.astype(np.float16) for k, v in out['fid']['frames'].items()}
pickle.dump(out, gzip.open('results_spiral.pkl.gz', 'wb'))
print('done', time.time() - T0, flush=True)
