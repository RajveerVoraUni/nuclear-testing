"""
generate_transformation_fig.py
Generates the core visual for redesigned Slide 2:
Shows the transformation from raw 1D seismic trace to 2D STFT spectrograms,
contrasting the time-frequency signature of an explosion vs an earthquake.
Uses real curated bandpassed waveforms from Station MNV and TUC.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data') if os.path.exists(os.path.join(BASE_DIR, 'data')) else BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, 'figures') if os.path.exists(os.path.join(BASE_DIR, 'figures')) else BASE_DIR

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 1.0
})

NAVY = '#0B1B3D'
RED = '#C53030'
BLUE = '#2B6CB0'
GRAY = '#64748B'
TEAL = '#0D9488'

eq = np.load(os.path.join(DATA_DIR, 'earthquake_data.npz'), allow_pickle=True)
exp = np.load(os.path.join(DATA_DIR, 'explosion_data.npz'), allow_pickle=True)

exp_idx = 8
eq_idx = 15

b, a = signal.butter(4, [0.8, 5.0], btype='bandpass', fs=20.0)

exp_z_filt = signal.filtfilt(b, a, exp['z'][exp_idx])
eq_z_filt = signal.filtfilt(b, a, eq['z'][eq_idx])

m_exp = max(np.max(np.abs(exp_z_filt)), np.max(np.abs(signal.filtfilt(b, a, exp['n'][exp_idx]))), np.max(np.abs(signal.filtfilt(b, a, exp['e'][exp_idx]))))
m_eq = max(np.max(np.abs(eq_z_filt)), np.max(np.abs(signal.filtfilt(b, a, eq['n'][eq_idx]))), np.max(np.abs(signal.filtfilt(b, a, eq['e'][eq_idx]))))

exp_z = exp_z_filt / m_exp
eq_z = eq_z_filt / m_eq

t = np.linspace(0, 60, 1200)

def compute_spec(sig, fs=20.0):
    f, t_s, Zxx = signal.stft(sig, fs=fs, window='hann', nperseg=256, noverlap=236)
    spec = np.log10(np.abs(Zxx[:128, :])**2 + 1e-6)
    return f[:128], t_s, spec

f_exp, t_exp, spec_exp = compute_spec(exp_z)
f_eq, t_eq, spec_eq = compute_spec(eq_z)

fig = plt.figure(figsize=(13.8, 3.4), dpi=300)
gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.25, 1.4, 1.4], wspace=0.30)

# Panel 1: Time Domain (1D Waveform)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, exp_z, color=RED, lw=1.1, label='Explosion (Z)', alpha=0.9)
ax0.plot(t, eq_z, color=BLUE, lw=1.0, label='Earthquake (Z)', alpha=0.75)
ax0.set_title('1. Time Domain: Overlapping Signatures', fontweight='bold', fontsize=10.5, color=NAVY, pad=8)
ax0.set_xlabel('Time (s)', fontweight='bold', color=NAVY)
ax0.set_ylabel('Normalized Amplitude', fontweight='bold', color=NAVY)
ax0.set_xlim(0, 60)
ax0.set_ylim(-1.20, 1.20)
ax0.grid(True, linestyle='--', color='#E2E8F0', alpha=0.8)
ax0.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Panel 2: STFT Bridge
ax_arrow = fig.add_subplot(gs[0, 1])
ax_arrow.axis('off')
ax_arrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                  arrowprops=dict(facecolor=TEAL, edgecolor=TEAL, shrink=0.05, width=2.5, headwidth=8))
ax_arrow.text(0.5, 0.72, 'STFT', ha='center', va='center', fontweight='bold', fontsize=12, color=TEAL)
ax_arrow.text(0.5, 0.28, 'Windowed\nFFT', ha='center', va='center', fontsize=8, color=GRAY)

vmin = -6.0
vmax = -2.0

# Panel 3: Explosion Spectrogram
ax1 = fig.add_subplot(gs[0, 2])
im1 = ax1.pcolormesh(t_exp, f_exp, spec_exp, shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax)
ax1.set_title('2a. Explosion: High Freq, Fast Decay', fontweight='bold', fontsize=10.5, color=RED, pad=8)
ax1.set_xlabel('Time (s)', fontweight='bold', color=NAVY)
ax1.set_ylabel('Frequency (Hz)', fontweight='bold', color=NAVY)
ax1.set_ylim(0, 8)
ax1.set_xlim(0, 60)
ax1.annotate('Impulsive P burst\n(3.8 Hz, fast decay)', xy=(12.5, 3.8), xytext=(19, 5.8),
             arrowprops=dict(facecolor='white', shrink=0.08, width=1.2, headwidth=5),
             fontsize=8, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.25', fc='#000000', alpha=0.75, ec='none'))

# Panel 4: Earthquake Spectrogram
ax2 = fig.add_subplot(gs[0, 3])
im2 = ax2.pcolormesh(t_eq, f_eq, spec_eq, shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax)
ax2.set_title('2b. Earthquake: Persistent Shear Coda', fontweight='bold', fontsize=10.5, color=BLUE, pad=8)
ax2.set_xlabel('Time (s)', fontweight='bold', color=NAVY)
ax2.set_ylabel('Frequency (Hz)', fontweight='bold', color=NAVY)
ax2.set_ylim(0, 8)
ax2.set_xlim(0, 60)
ax2.annotate('Prolonged Coda\n(Sustained ~4 Hz)', xy=(35.0, 4.3), xytext=(20, 6.0),
             arrowprops=dict(facecolor='white', shrink=0.08, width=1.2, headwidth=5),
             fontsize=8, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.25', fc='#000000', alpha=0.75, ec='none'))

out_path = os.path.join(FIG_DIR, 'transformation_motivation.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved {out_path} successfully!')
