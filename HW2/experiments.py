import numpy as np, pickle, time
import kh_sim

out = {}
t0 = time.time()
# A. fiducial run with snapshots
out['fid'] = kh_sim.run(N=256, tEnd=2.0, w0=0.1, snap_times=(0, 0.5, 1.0, 1.5, 2.0))
print('fid', time.time() - t0, flush=True)
# B. resolution + limiter at t=2
for N in (64, 128, 256, 512):
    if N == 256:
        out[('res', N, False)] = out['fid']
        continue
    out[('res', N, False)] = kh_sim.run(N=N, tEnd=2.0, w0=0.1)
    print('res', N, time.time() - t0, flush=True)
out[('res', 256, True)] = kh_sim.run(N=256, tEnd=2.0, w0=0.1, limiter=True)
print('lim', time.time() - t0, flush=True)
# C. linear growth with a small seed
for N in (128, 256, 512):
    out[('lin', N)] = kh_sim.run(N=N, tEnd=1.5, w0=1e-3)
    print('lin', N, time.time() - t0, flush=True)

slim = {}
for k, v in out.items():
    slim[k] = {kk: vv for kk, vv in v.items() if kk != 'rho' or not str(k).startswith("('lin")}
pickle.dump(slim, open('results.pkl', 'wb'))
print('done', time.time() - t0)
