"""Plot G(T) from what 04_gibbs_curve.py wrote, and report the crossover.

    python 05_plot_gibbs_curve.py gibbs_SrTiO3.json

The quantity that matters is the gap, dG = G_amorphous - G_crystal: it is the
driving force for crystallisation, and being a difference it cancels most of
the systematic error in either branch. So it gets its own panel, on its own
axis -- never a second y-axis on the G(T) plot.
"""
from __future__ import annotations

import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "gibbs_SrTiO3.json"
with open(path) as f:
    curve = json.load(f)

T = np.array(curve["temperatures"], dtype=float)
g_crystal = np.array(curve["g_crystal"], dtype=float)
g_amorphous = np.array(curve["g_amorphous"], dtype=float)
gap = 1000.0 * (g_amorphous - g_crystal)        # meV/atom

print(f"{'T (K)':>8} {'G_xtal':>12} {'G_amorph':>12} {'dG (meV/at)':>13}")
for t, gc, ga, d in zip(T, g_crystal, g_amorphous, gap):
    print(f"{t:8.0f} {gc:12.6f} {ga:12.6f} {d:13.2f}")

# A sign change means the two branches cross inside the reported window --
# for a stable crystal they should not, and a crossing usually means one
# anchor is wrong rather than that the amorphous phase became stable.
if np.any(np.diff(np.sign(gap))):
    t_cross = float(np.interp(0.0, gap, T))
    print(f"\nWARNING: branches cross at ~{t_cross:.0f} K. Check the anchors.")
else:
    print(f"\nno crossing in {T.min():.0f}-{T.max():.0f} K; "
          f"dG runs {gap.min():.1f} to {gap.max():.1f} meV/atom")

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed; numbers above are the whole result")

fig, (ax_g, ax_gap) = plt.subplots(2, 1, figsize=(5.2, 6.0), sharex=True,
                                   height_ratios=[2, 1], constrained_layout=True)

ax_g.plot(T, g_crystal, lw=2, color="#2f6fb5", label="crystal")
ax_g.plot(T, g_amorphous, lw=2, color="#c2622d", label="amorphous")
ax_g.set_ylabel("G (eV/atom)")
ax_g.legend(frameon=False, loc="best")
# label the series directly too, so identity is never colour alone
ax_g.annotate("crystal", (T[-1], g_crystal[-1]), xytext=(-4, 6),
              textcoords="offset points", ha="right", color="#2f6fb5", fontsize=9)
ax_g.annotate("amorphous", (T[-1], g_amorphous[-1]), xytext=(-4, 6),
              textcoords="offset points", ha="right", color="#c2622d", fontsize=9)

ax_gap.plot(T, gap, lw=2, color="#4a4a4a")
ax_gap.axhline(0.0, lw=1, color="#bbbbbb", zorder=0)
ax_gap.set_ylabel(r"$G_{\rm amorph} - G_{\rm xtal}$" "\n(meV/atom)")
ax_gap.set_xlabel("temperature (K)")

for ax in (ax_g, ax_gap):
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

out = path.replace(".json", ".png")
fig.savefig(out, dpi=200)
print(f"figure written to {out}")
