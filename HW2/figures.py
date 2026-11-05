import numpy as np, pickle, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
R = pickle.load(open('results.pkl', 'rb'))
summary = {}

# --- Fig 1: snapshots (fiducial) + resolution/limiter row -------------------------
fid = R['fid']
times = [0.0, 0.5, 1.0, 2.0]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.75), dpi=220,
                         gridspec_kw=dict(wspace=0.05, hspace=0.22))
for ax, tt in zip(axes[0], times):
    im = ax.imshow(fid['snaps'][tt].T, origin='lower', extent=[0, 1, 0, 1],
                   cmap='magma', vmin=0.8, vmax=2.2)
    ax.set_title(f'N=256, t = {tt:g}', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
res_panels = [(('res', 64, False), 'N=64'), (('res', 128, False), 'N=128'),
              (('res', 512, False), 'N=512'), (('res', 256, True), 'N=256 + limiter')]
for ax, (key, lab) in zip(axes[1], res_panels):
    ax.imshow(R[key]['rho'].T, origin='lower', extent=[0, 1, 0, 1], cmap='magma', vmin=0.8, vmax=2.2)
    ax.set_title(f'{lab}, t = 2', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label(r'density $\rho$')
fig.savefig('fig_density.png', bbox_inches='tight')
plt.close(fig)

# --- Fig 2: linear growth vs theory ---------------------------------------------
k, dU, r1, r2 = 4 * np.pi, 1.0, 1.0, 2.0
sigma_th = k * dU * np.sqrt(r1 * r2) / (r1 + r2)
summary['sigma_theory'] = sigma_th
fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=220)
cols = {128: '#9ecae1', 256: '#4292c6', 512: '#08306b'}
fit_win = (0.5, 1.2)
for N in (128, 256, 512):
    r = R[('lin', N)]
    t, A = r['t'], r['A']
    m = (t >= fit_win[0]) & (t <= fit_win[1])
    s, c = np.polyfit(t[m], np.log(A[m]), 1)
    summary[f'sigma_N{N}'] = s
    ax.semilogy(t, A, color=cols[N], lw=1.4, label=fr'N={N}: $\sigma$={s:.2f}')
# theory line anchored at N=512 run at t=0.5
r = R[('lin', 512)]; i0 = np.argmin(abs(r['t'] - 0.5))
tt = np.linspace(0.3, 1.5, 50)
ax.semilogy(tt, r['A'][i0] * np.exp(sigma_th * (tt - 0.5)), 'k--', lw=1,
            label=fr'linear theory: $\sigma$={sigma_th:.2f}')
ax.axvspan(*fit_win, color='0.92', zorder=0)
ax.set_xlabel('t'); ax.set_ylabel(r'$|\hat v_y|$ of $k=4\pi$ mode')
ax.set_ylim(2e-4, 0.3); ax.legend(frameon=False, fontsize=6.5, loc='upper left')
fig.tight_layout(); fig.savefig('fig_growth.png'); plt.close(fig)

# --- conservation -----------------------------------------------------------------
cons = {}
for key in [('res', 64, False), ('res', 128, False), ('res', 256, False), ('res', 512, False), ('res', 256, True)]:
    r = R[key]
    d = r['tot1'] - r['tot0']
    cons[str(key)] = dict(dM_rel=float(d[0] / r['tot0'][0]), dE_rel=float(d[3] / r['tot0'][3]),
                          dPx=float(d[1]), dPy=float(d[2]), nstep=int(r['nstep']))
summary['conservation'] = cons
json.dump(summary, open('summary.json', 'w'), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
