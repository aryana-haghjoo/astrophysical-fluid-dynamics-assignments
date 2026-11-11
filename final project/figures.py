import numpy as np, pickle, gzip, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spiral_sim as S

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'savefig.dpi': 200})
R = pickle.load(gzip.open('results_spiral.pkl.gz', 'rb'))
CM, VMIN, VMAX = 'inferno', -0.6, 0.9
EXT = [-1.1, 1.1, -1.1, 1.1]
g0 = S.Galaxy(N=256, F=0.05)
cs, Op, sini = 0.05, 0.9, np.sin(np.radians(20))
summ = {}

def show(ax, rho, title=None, lim=1.0):
    im = ax.imshow(np.log10(rho).T, origin='lower', extent=EXT, cmap=CM, vmin=VMIN, vmax=VMAX)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    return im

# --- setup: potential + frequencies --------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
Phi1 = g0.eps * np.cos(g0.spiral_phase()) * (g0.r < 1.0) * (g0.r > 0.15)
ax[0].imshow(Phi1.T, origin='lower', extent=EXT, cmap='RdBu_r')
ax[0].set_xlim(-1, 1); ax[0].set_ylim(-1, 1); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].set_title('imposed spiral potential $\\Phi_1$ (blue = arm)')
for rr, lab in [(0.325, 'ILR'), (1.0, 'disk edge')]:
    th = np.linspace(0, 2*np.pi, 200); ax[0].plot(rr*np.cos(th), rr*np.sin(th), 'k--', lw=0.8)
r = np.linspace(0.05, 1.6, 300)
Om = 1/r; kap = np.sqrt(2)/r
ax[1].plot(r, Om, 'k', lw=2, label=r'$\Omega = v_c/r$')
ax[1].plot(r, Om - kap/2, 'C0', lw=2, label=r'$\Omega-\kappa/2$')
ax[1].axhline(Op, c='C3', lw=2, label=r'pattern speed $\Omega_p=0.9$')
ax[1].axvline(0.325, c='0.5', ls=':'); ax[1].text(0.2, 3.3, 'ILR', color='0.4')
ax[1].axvline(1/Op, c='0.5', ls=':'); ax[1].text(1/Op+0.02, 3.3, 'corotation', color='0.4')
ax[1].axvspan(0.15, 1.0, color='0.94', zorder=0); ax[1].text(0.55, 0.25, 'gas disk', ha='center', color='0.4')
ax[1].set_ylim(0, 3.6); ax[1].set_xlim(0, 1.6); ax[1].set_xlabel('r'); ax[1].set_ylabel('angular frequency')
ax[1].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig('fig_setup.png'); plt.close(fig)

# --- fiducial snapshots ------------------------------------------------------------
fr = R['fid']['frames']
fig, axs = plt.subplots(1, 4, figsize=(12, 3.3))
for a, t in zip(axs, [1.0, 3.0, 6.0, 20.0]):
    im = show(a, fr[t], f't = {t:g}  ({t*44:.0f} Myr)')
fig.colorbar(im, ax=axs, fraction=0.015, label=r'$\log_{10}\rho$')
fig.savefig('fig_snapshots.png'); plt.close(fig)

# --- streamlines on final state -----------------------------------------------------
rho = fr[20.0]; vx, vy = R['fid']['vel']
fig, ax = plt.subplots(figsize=(5.2, 5.2))
show(ax, rho, 'gas flow in the pattern frame (t = 20)', lim=0.75)
x = np.linspace(-1.1 + 1.1/256, 1.1 - 1.1/256, 256)
mask = (g0.r > 0.17) & (g0.r < 0.98)
ax.streamplot(x, x, np.where(mask, vx, np.nan).T, np.where(mask, vy, np.nan).T,
              color='w', density=1.2, linewidth=0.6, arrowsize=0.6)
fig.savefig('fig_streamlines.png'); plt.close(fig)

# --- forcing scan -----------------------------------------------------------------
Fs = [0.005, 0.01, 0.02, 0.05, 0.1]
fig = plt.figure(figsize=(13, 3.4))
for k, F in enumerate(Fs):
    a = fig.add_subplot(1, 6, k+1); show(a, R[('F', F)]['rho'], f'F = {100*F:g}%', lim=0.9)
a = fig.add_subplot(1, 6, 6)
for rr, c in [(0.35, 'C3'), (0.5, 'C1'), (0.8, 'C0')]:
    con = [R[('F', F)]['rings'][rr]['max']/R[('F', F)]['rings'][rr]['min'] - 1 for F in Fs]
    a.loglog(Fs, con, 'o-', c=c, label=f'r = {rr}')
a.loglog([0.005, 0.1], [0.005*20, 0.1*20], 'k:', lw=1); a.text(0.012, 0.15, '∝ F (linear)', fontsize=8)
a.set_xlabel('forcing F'); a.set_ylabel(r'arm contrast $\rho_{max}/\rho_{min}-1$'); a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('fig_forcing.png'); plt.close(fig)
summ['contrast_vs_F_r0.35'] = {F: float(R[('F', F)]['rings'][0.35]['max']/R[('F', F)]['rings'][0.35]['min']) for F in Fs}

# --- radial: incoming normal Mach number vs arm contrast -----------------------------
radii = R['radii']
rings = R['fid']['rings']
Mn = (radii/np.sqrt(radii**2 + 0.05**2) - Op*radii) * sini / cs
con = np.array([rings[r]['max']/rings[r]['min'] for r in radii])
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.plot(radii, Mn, 'C0o-', label=r'predicted inflow Mach $\mathcal{M}_\perp=(v_c-\Omega_p r)\sin i/c_s$')
ax.plot(radii, con, 'C3s-', label=r'measured arm contrast $\rho_{max}/\rho_{min}$')
ax.axhline(1, c='k', lw=0.8, ls=':'); ax.text(0.9, 1.15, 'sonic', fontsize=8, ha='right')
ax.set_xlabel('radius r'); ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, max(Mn.max(), con.max())*1.1)
fig.tight_layout(); fig.savefig('fig_radial.png'); plt.close(fig)
summ['Mn'] = dict(zip(map(float, radii), map(float, Mn))); summ['contrast'] = dict(zip(map(float, radii), map(float, con)))

# --- azimuthal profile: shock sits upstream of the potential minimum ----------------
fig, ax = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
rr = 0.45; p = rings[rr]
ph = p['phi']
ax[0].plot(ph, p['rho'], 'k', lw=1.8); ax[0].set_ylabel(r'gas density $\rho$')
ax[1].plot(ph, p['Phi1']/g0.eps, 'C0', lw=1.8); ax[1].set_ylabel(r'$\Phi_1$ (normalised)')
ax2 = ax[1].twinx(); ax2.plot(ph, p['un']/cs, 'C3', lw=1.2); ax2.set_ylabel(r'$u_\perp/c_s$', color='C3')
for a in ax:
    for pm in ph[np.where((p['Phi1'] < np.roll(p['Phi1'], 1)) & (p['Phi1'] < np.roll(p['Phi1'], -1)))]:
        a.axvline(pm, c='C0', ls=':', lw=1)
ax[1].set_xlabel(r'azimuth $\phi$ (gas flows toward $+\phi$)')
ax[0].set_title(f'ring r = {rr}: density peak (shock) vs potential minimum (dotted)', fontsize=10)
fig.tight_layout(); fig.savefig('fig_profile.png'); plt.close(fig)
# phase offset vs radius (shock upstream of potential min = negative offset)
offs = []
for r0 in radii:
    q = rings[r0]
    i_pk = np.argmax(q['rho']); phpk = q['phi'][i_pk]
    mins = q['phi'][np.where((q['Phi1'] < np.roll(q['Phi1'], 1)) & (q['Phi1'] < np.roll(q['Phi1'], -1)))]
    d = np.angle(np.exp(1j*2*(phpk - mins)))/2        # mod pi (2 arms)
    offs.append(np.degrees(d[np.argmin(np.abs(d))]))
summ['offset_deg'] = dict(zip(map(float, radii), offs))

# --- sound speed scan ------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.3))
for a, c in zip(axs, [0.025, 0.05, 0.1]):
    show(a, R[('cs', c)]['rho'], f'$c_s$ = {c} (~{c*220:.0f} km/s)', lim=0.9)
fig.tight_layout(); fig.savefig('fig_cs.png'); plt.close(fig)

# --- resolution ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for N, c in [(128, '#9ecae1'), (256, '#4292c6'), (512, '#08306b')]:
    q = R[('N', N)]['rings'][0.35]
    ax.plot(q['phi'], q['rho'], c=c, lw=1.6, label=f'N = {N}²')
ax.set_xlim(-2.2, -0.6); ax.set_xlabel(r'azimuth $\phi$ at r = 0.35'); ax.set_ylabel(r'$\rho$'); ax.legend(frameon=False)
ax.set_title('resolution: arm position converges, peak density keeps rising', fontsize=10)
fig.tight_layout(); fig.savefig('fig_resolution.png'); plt.close(fig)
summ['peak_vs_N_r0.35'] = {N: float(R[('N', N)]['rings'][0.35]['max']) for N in (128, 256, 512)}

# --- shock-tube verification -----------------------------------------------------------
ver = R['shocktest']
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for v, c in zip(ver, ['C0', 'C1', 'C2', 'C3']):
    ax[0].semilogy(v['x'], v['rho'], c=c, lw=1.5, label=fr"$\mathcal{{M}}_1$={v['M1']:.1f}")
    xs = v['Vs']*v['T']; ax[0].semilogy([-xs, -xs, xs, xs], [1, v['rho2_exact'], v['rho2_exact'], 1], c=c, ls=':', lw=1)
ax[0].set_xlim(-0.12, 0.12); ax[0].set_ylim(0.7, 70); ax[0].set_xlabel('x'); ax[0].set_ylabel(r'$\rho$'); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title('colliding flows: simulation (solid) vs exact (dotted)', fontsize=9)
M = np.linspace(1, 7, 100)
ax[1].plot(M, M**2, 'k', lw=1.5, label=r'isothermal jump $D=\mathcal{M}_1^2$')
ax[1].plot([v['M1'] for v in ver], [v['rho2_sim'] for v in ver], 'C3o', ms=7, label='simulation')
ax[1].set_xlabel(r'upstream Mach $\mathcal{M}_1$'); ax[1].set_ylabel(r'density jump $D$'); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig('fig_shocktest.png'); plt.close(fig)
summ['shocktest'] = [dict(M1=float(v['M1']), exact=float(v['rho2_exact']), sim=float(v['rho2_sim'])) for v in ver]

# --- contrast vs time ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3))
ax.plot(R['fid']['t'], R['fid']['contrast'], 'k', lw=1.6)
ax.axvspan(0, 3, color='0.92'); ax.text(1.5, 1.2, 'ramp-up', ha='center', fontsize=8)
ax.set_xlabel('t'); ax.set_ylabel(r'arm contrast at r = 0.45')
fig.tight_layout(); fig.savefig('fig_time.png'); plt.close(fig)

json.dump(summ, open('summary_spiral.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
