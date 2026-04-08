import sys

expected = "/opt/anaconda3/envs/triadic_plot/bin/python"
if sys.executable != expected:
    raise RuntimeError(
        f"Wrong Python interpreter.\n"
        f"Current:  {sys.executable}\n"
        f"Expected: {expected}\n"
        f"Open Spyder with the triadic_plot environment or set Spyder's interpreter to that path."
    )

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# USER SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
STATS_DIR = os.path.join(REPO_DIR, "STATS")
OUTDIR = SCRIPT_DIR
NSNAPS = 30
BLOCKID = 2
NMODES_COEFF = 10
DPI = 300

os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# LOAD DATA
# ============================================================
svd_s_path = os.path.join(STATS_DIR, f"svd_s_{NSNAPS}.parquet")
svd_v_path = os.path.join(STATS_DIR, f"svd_v_{NSNAPS}.parquet")
triad_path = os.path.join(STATS_DIR, f"py_tr_tot_0_{BLOCKID}.npy")

svd_s = pd.read_parquet(svd_s_path).iloc[:, 0].to_numpy()
svd_v = pd.read_parquet(svd_v_path).to_numpy()
triad = np.load(triad_path, allow_pickle=False)

mode_idx = np.arange(1, len(svd_s) + 1)
snap_idx = np.arange(1, svd_v.shape[1] + 1)
ak = svd_s[:, None] * svd_v

# ============================================================
# 1) CUMULATIVE ENERGY
# ============================================================
energy = svd_s**2
energy /= energy.sum()

fig, ax = plt.subplots(figsize=(4.0, 2.8))
ax.plot(mode_idx, energy, "-o", ms=3, label="Mode energy")
ax.plot(mode_idx, np.cumsum(energy), "-s", ms=2.5, label="Cumulative")
ax.set_xlabel("Mode index")
ax.set_ylabel("Energy fraction")
ax.set_ylim(0.0, 1.02)
ax.set_title("POD energy content")
ax.legend(frameon=False)
savefig(fig, "pod_energy.png")

# ============================================================
# 2) TEMPORAL COEFFICIENTS
# ============================================================
# nshow = min(NMODES_COEFF, ak.shape[0])
nshow = 9

ncols = 3
nrows = int(np.ceil(nshow / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(5.2, 1 * nrows),
    sharex=True
)

axes = np.atleast_1d(axes).ravel()

for k in range(nshow):
    ax = axes[k]
    ax.plot(snap_idx, ak[k, :], lw=1.0)
    ax.set_title(f"Mode {k+1}", fontsize=9)
    ax.set_ylabel(r"$\chi_k$")
    ax.tick_params(labelsize=8)

for ax in axes[:nshow]:
    ax.grid(False)

for ax in axes[nshow:]:
    ax.axis("off")

for ax in axes[-ncols:]:
    ax.set_xlabel("Spanshot index")

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "temporal_coefficients.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 3) TRIADIC CUBE
# ============================================================
nm = min(30, triad.shape[0])
cube = triad[:nm, :nm, :nm]

threshold = 0.01 * np.max(np.abs(cube))
mask = np.abs(cube) >= threshold

ii, jj, kk = np.where(mask)
c = cube[mask]

x = jj + 1
y = ii + 1
z = kk + 1

if c.size == 0:
    raise ValueError("Threshold removed all points. Lower the threshold.")

vmax = np.max(np.abs(c))/1000
norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

fig = plt.figure(figsize=(4.4, 3.9))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    x, y, z,
    c=c,
    cmap="coolwarm",
    norm=norm,
    s=6,
    alpha=0.6,
    linewidths=0.0,
)

ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$m$")
ax.set_zlabel(r"$n$")
ax.set_title("Triadic interaction tensor")
ax.set_xlim(1, nm)
ax.set_ylim(1, nm)
ax.set_zlim(1, nm)
ax.invert_yaxis()
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=-45)
ax.grid(False)

cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.08)
cbar.set_label("Transfer")

savefig(fig, "triadic_cube.png")

print(f"Saved figures in: {OUTDIR}")
