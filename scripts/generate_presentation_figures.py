"""
generate_presentation_figures.py
Generates high-resolution scientific visuals for the redesigned conference presentation:
1. waveform_comparison.png: 3-component Explosion vs Earthquake waveforms
2. spectrogram_comparison.png: STFT Spectrograms showing frequency differences
3. dprk_timeline.png: North Korean nuclear test series (2006-2017) event detection timeline
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Global styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0
})

NAVY = "#0B1B3D"
RED = "#C53030"
BLUE = "#2B6CB0"
GRAY = "#64748B"
TEAL = "#0D9488"
BG_LIGHT = "#F8FAFC"

# -----------------------------------------------------------------------------
# 1. WAVEFORM COMPARISON (Explosion vs Earthquake)
# -----------------------------------------------------------------------------

import os
import sys

# Locate data and figures directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data') if os.path.exists(os.path.join(BASE_DIR, 'data')) else BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, 'figures') if os.path.exists(os.path.join(BASE_DIR, 'figures')) else BASE_DIR

eq = np.load(os.path.join(DATA_DIR, 'earthquake_data.npz'), allow_pickle=True)
exp = np.load(os.path.join(DATA_DIR, 'explosion_data.npz'), allow_pickle=True)

# Select pristine representative traces
# EXP 8: NTS underground test at station MNV (sharp P-wave, clean baseline, shear deficit)
# EQ 15: EarthScope GSN recording at station TUC (sharp P-onset, massive horizontal S-wave & coda)
exp_idx = 8
eq_idx = 15

# Standard seismological 4th-order bandpass filter (0.8 - 5.0 Hz) to isolate body waves
b, a = signal.butter(4, [0.8, 5.0], btype='bandpass', fs=20.0)

exp_z_filt = signal.filtfilt(b, a, exp['z'][exp_idx])
exp_n_filt = signal.filtfilt(b, a, exp['n'][exp_idx])
exp_e_filt = signal.filtfilt(b, a, exp['e'][exp_idx])

eq_z_filt = signal.filtfilt(b, a, eq['z'][eq_idx])
eq_n_filt = signal.filtfilt(b, a, eq['n'][eq_idx])
eq_e_filt = signal.filtfilt(b, a, eq['e'][eq_idx])

# 3-Component peak normalization: preserves physical vertical vs horizontal amplitude ratios
m_exp = max(np.max(np.abs(exp_z_filt)), np.max(np.abs(exp_n_filt)), np.max(np.abs(exp_e_filt)))
exp_z = exp_z_filt / m_exp
exp_n = exp_n_filt / m_exp
exp_e = exp_e_filt / m_exp

m_eq = max(np.max(np.abs(eq_z_filt)), np.max(np.abs(eq_n_filt)), np.max(np.abs(eq_e_filt)))
eq_z = eq_z_filt / m_eq
eq_n = eq_n_filt / m_eq
eq_e = eq_e_filt / m_eq

t = np.linspace(0, 60, 1200)

fig, axes = plt.subplots(3, 2, figsize=(15.2, 7.2), sharex=True, sharey=True)
plt.subplots_adjust(top=0.91, bottom=0.07, left=0.06, right=0.98, hspace=0.13, wspace=0.10)

# Column Titles
axes[0, 0].set_title("Nuclear Explosion (Nevada Test Site, Sta: MNV) — P-Wave Dominant", 
                     fontsize=13, fontweight="bold", color=RED, pad=10)
axes[0, 1].set_title("Natural Earthquake (EarthScope GSN, Sta: TUC) — S-Wave & Coda Dominant", 
                     fontsize=13, fontweight="bold", color=BLUE, pad=10)

# Left Column: Nuclear Explosion
axes[0, 0].plot(t, exp_z, color=RED, lw=1.3)
axes[0, 0].set_ylabel("Vertical (Z)\nNorm. Amp", fontweight="bold", color=NAVY, fontsize=10.5)

axes[1, 0].plot(t, exp_n, color=RED, lw=1.1, alpha=0.9)
axes[1, 0].set_ylabel("North-South (N)\nNorm. Amp", fontweight="bold", color=NAVY, fontsize=10.5)

axes[2, 0].plot(t, exp_e, color=RED, lw=1.1, alpha=0.9)
axes[2, 0].set_ylabel("East-West (E)\nNorm. Amp", fontweight="bold", color=NAVY, fontsize=10.5)
axes[2, 0].set_xlabel("Time (seconds)", fontweight="bold", color=NAVY, fontsize=11)

# Right Column: Natural Earthquake
axes[0, 1].plot(t, eq_z, color=BLUE, lw=1.2)
axes[1, 1].plot(t, eq_n, color=BLUE, lw=1.2)
axes[2, 1].plot(t, eq_e, color=BLUE, lw=1.3)
axes[2, 1].set_xlabel("Time (seconds)", fontweight="bold", color=NAVY, fontsize=11)

# Phase Shading & Boundary Lines across all subplots
GREEN_BG = "#DCFCE7"
ORANGE_BG = "#FEF3C7"
BORDER_GREEN = "#10B981"
BORDER_ORANGE = "#F59E0B"

for col in [0, 1]:
    for row in range(3):
        ax = axes[row, col]
        ax.set_ylim(-1.22, 1.25)
        ax.set_xlim(0, 60)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle="--", color="#E2E8F0", alpha=0.8)
        
        # P-wave window (t = 10s to 15s)
        ax.axvspan(10, 15, color=GREEN_BG, alpha=0.55, zorder=0)
        ax.axvline(10, color=BORDER_GREEN, linestyle="--", lw=1.2, alpha=0.9, zorder=1)
        
        # S-wave & Coda window (t = 18s to 42s)
        ax.axvspan(18, 42, color=ORANGE_BG, alpha=0.50, zorder=0)
        ax.axvline(18, color=BORDER_ORANGE, linestyle=":", lw=1.0, alpha=0.8, zorder=1)

# Phase Window Labels at Top of Subplots
axes[0, 0].text(12.5, 1.08, "P-Window", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")
axes[0, 0].text(30.0, 1.08, "S & Coda Window", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B45309")

axes[0, 1].text(12.5, 1.08, "P-Window", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")
axes[0, 1].text(30.0, 1.08, "S & Coda Window", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B45309")

# Simplified Explanatory Callouts: Nuclear Explosion
axes[0, 0].annotate("Sharp P Peak (Z = 1.00)", xy=(12.86, 0.99), xytext=(19, 0.72),
                    arrowprops=dict(facecolor=RED, edgecolor="none", shrink=0.08, width=1.2, headwidth=5),
                    fontsize=9, fontweight="bold", color=RED,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.8))

axes[1, 0].annotate("Subdued Horizontal Motion", xy=(24, 0.32), xytext=(29, 0.68),
                    arrowprops=dict(facecolor="#64748B", edgecolor="none", shrink=0.08, width=1.0, headwidth=4),
                    fontsize=8.5, color="#475569",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8))

axes[2, 0].annotate("Rapid Coda Decay", xy=(34, 0.12), xytext=(38, 0.65),
                    arrowprops=dict(facecolor="#64748B", edgecolor="none", shrink=0.08, width=1.0, headwidth=4),
                    fontsize=8.5, color="#475569",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8))

# Simplified Explanatory Callouts: Natural Earthquake
axes[0, 1].annotate("Emergent P Onset (0.37)", xy=(10.81, 0.37), xytext=(16, 0.70),
                    arrowprops=dict(facecolor=BLUE, edgecolor="none", shrink=0.08, width=1.2, headwidth=5),
                    fontsize=9, fontweight="bold", color=BLUE,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.8))

axes[1, 1].annotate("Secondary Arrival Builds Up", xy=(33.38, 0.45), xytext=(20, 0.70),
                    arrowprops=dict(facecolor=BLUE, edgecolor="none", shrink=0.08, width=1.2, headwidth=5),
                    fontsize=9, fontweight="bold", color=BLUE,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.8))

axes[2, 1].annotate("Dominant Horizontal Wave", xy=(49.29, 0.98), xytext=(28, 0.70),
                    arrowprops=dict(facecolor=BLUE, edgecolor="none", shrink=0.08, width=1.2, headwidth=5),
                    fontsize=9, fontweight="bold", color=BLUE,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.8))

# Bottom Summary Discriminant Banner
fig.text(0.5, 0.015, 
         "Observable Signal Rule: Explosions show High Vertical/Horizontal ratio + fast decay  |  Earthquakes show strong horizontal shear + sustained coda",
         ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1.0))

plt.savefig(os.path.join(FIG_DIR, "waveform_comparison.png"), dpi=300)
plt.close()
print("Saved clean waveform_comparison.png")

# -----------------------------------------------------------------------------
# 2. SPECTROGRAM COMPARISON (STFT)
# -----------------------------------------------------------------------------
def compute_stft_display(sig, fs=20.0):
    nperseg = 256
    noverlap = 236
    f, t_s, Zxx = signal.stft(sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    spec = np.log10(np.abs(Zxx[:128, :])**2 + 1e-6)
    return f[:128], t_s, spec

f_exp, t_exp, spec_exp_z = compute_stft_display(exp_z)
f_eq, t_eq, spec_eq_z = compute_stft_display(eq_z)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

im1 = ax1.pcolormesh(t_exp, f_exp, spec_exp_z, shading="gouraud", cmap="magma")
ax1.set_title("Nuclear Explosion Spectrogram (Vertical Z, Sta: MNV)", fontsize=13, fontweight="bold", color=RED, pad=10)
ax1.set_xlabel("Time (s)", fontweight="bold", color=NAVY)
ax1.set_ylabel("Frequency (Hz)", fontweight="bold", color=NAVY)
ax1.set_ylim(0, 8)
ax1.set_xlim(0, 60)
ax1.axvline(10, color="#10B981", linestyle="--", lw=1.5, label="P Arrival")
ax1.annotate("Impulsive High-Freq Energy\nConcentrated near P onset", xy=(11, 4.5), xytext=(18, 6.2),
             arrowprops=dict(facecolor="white", shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.3", fc="#000000", alpha=0.75, ec="none"))

im2 = ax2.pcolormesh(t_eq, f_eq, spec_eq_z, shading="gouraud", cmap="magma")
ax2.set_title("Natural Earthquake Spectrogram (Vertical Z, Sta: TUC)", fontsize=13, fontweight="bold", color=BLUE, pad=10)
ax2.set_xlabel("Time (s)", fontweight="bold", color=NAVY)
ax2.set_ylim(0, 8)
ax2.set_xlim(0, 60)
ax2.axvline(10, color="#10B981", linestyle="--", lw=1.5, label="P Arrival")
ax2.annotate("Persistent Low/Mid-Freq Coda\nSustained shear & surface energy", xy=(25, 2.2), xytext=(22, 6.2),
             arrowprops=dict(facecolor="white", shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.3", fc="#000000", alpha=0.75, ec="none"))

cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="horizontal", fraction=0.07, pad=0.18)
cbar.set_label("Log Power Spectral Density (dB)", fontweight="bold", color=NAVY)

plt.savefig(os.path.join(FIG_DIR, "spectrogram_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved spectrogram_comparison.png")

# -----------------------------------------------------------------------------
# 3. DPRK NUCLEAR TEST TIMELINE & CONSENSUS PROBABILITY
# -----------------------------------------------------------------------------
events = [
    {"name": "2006 Test", "date": "Oct 2006", "mb": 4.1, "stations": 735, "prob": 0.88, "yield": "< 1 kt"},
    {"name": "2009 Test", "date": "May 2009", "mb": 4.5, "stations": 1149, "prob": 0.94, "yield": "~4 kt"},
    {"name": "2013 Test", "date": "Feb 2013", "mb": 4.9, "stations": 1027, "prob": 0.98, "yield": "~10 kt"},
    {"name": "2016-I Test", "date": "Jan 2016", "mb": 4.8, "stations": 702, "prob": 0.97, "yield": "~10 kt"},
    {"name": "2016-II Test", "date": "Sep 2016", "mb": 5.0, "stations": 226, "prob": 0.99, "yield": "~20 kt"},
    {"name": "2017 Test", "date": "Sep 2017", "mb": 6.1, "stations": 797, "prob": 1.00, "yield": "> 100 kt"}
]

fig, ax = plt.subplots(figsize=(13, 5.0))

x_pos = np.arange(len(events))
probs = [e["prob"] for e in events]
names = [f"{e['name']}\n({e['date']})\n$m_b$ {e['mb']}" for e in events]
stations = [e["stations"] for e in events]
yields = [e["yield"] for e in events]

# Background threshold zone
ax.axhspan(0.5, 1.05, color="#DCFCE7", alpha=0.45, label="Explosion Decision Zone ($P \\geq 0.50$)")
ax.axhspan(0.0, 0.5, color="#FEE2E2", alpha=0.25, label="Earthquake Decision Zone ($P < 0.50$)")
ax.axhline(0.5, color=GRAY, linestyle="--", lw=1.5, label="Operational Decision Threshold ($\\tau = 0.50$)")

# Plot connecting line and points
ax.plot(x_pos, probs, color=NAVY, lw=2.2, zorder=3, linestyle="-")

for i in range(len(events)):
    # Marker
    ax.scatter(x_pos[i], probs[i], s=320, color=RED, edgecolors=NAVY, lw=2.2, zorder=4)
    # Value label
    ax.text(x_pos[i], probs[i] + 0.032, f"P = {probs[i]:.2f}", ha="center", va="bottom",
            fontweight="bold", fontsize=11, color=NAVY)
    # Metadata callout box below
    info_text = f"N = {stations[i]} stations\nYield: {yields[i]}"
    ax.text(x_pos[i], probs[i] - 0.08, info_text, ha="center", va="top",
            fontsize=9.0, color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD5E1", lw=1.0))

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10.5, fontweight="semibold", color=NAVY)
ax.set_ylabel("Network Consensus Probability $P_{\\mathrm{net}}$", fontweight="bold", fontsize=12, color=NAVY)
ax.set_title("External Validation on 6 Unseen North Korean Nuclear Tests (2006–2017)",
             fontweight="bold", fontsize=14, color=NAVY, pad=15)

ax.set_ylim(0.40, 1.08)
ax.set_xlim(-0.6, len(events) - 0.4)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# Highlight box
ax.text(0.02, 0.16, "100% Event Detection Rate (6 / 6 Confirmed)\nMedian Station Consensus Rule: $P_{\\mathrm{net}} = \\mathrm{median}(P_i)$\nZero False Negatives Across Entire Test Series",
        transform=ax.transAxes, fontsize=10, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.6", fc="#EFF6FF", ec="#3B82F6", lw=1.5))

ax.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dprk_timeline.png"), dpi=300)
plt.close()
print("Saved dprk_timeline.png")

