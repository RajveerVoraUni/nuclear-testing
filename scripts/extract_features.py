import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data') if os.path.exists(os.path.join(BASE_DIR, 'data')) else BASE_DIR

def get_data_path(fname):
    p = os.path.join(DATA_DIR, fname)
    return p if os.path.exists(p) else fname
"""
extract_features.py
Extracts pure 3-component (Z, N, E) physical discrimination features:
- Cross-component P/S ratio (P on Z vs Shear on H)
- Horizontal-to-Vertical (H/V) ratio and horizontal shear energy fraction
- Cross-phase spectral ratio (High-freq P on Z vs Low-freq S on H)
- 3D particle motion rectilinearity and planarity (covariance tensor)
- Spectral centroids, spreads, entropy, and coda decays for both Z and H
Reads get_data_path("earthquake_data.npz") and get_data_path("explosion_data.npz").
Saves tabular dataset to get_data_path("features.csv").
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def extract_3c_trace_features(z_raw, n_raw, e_raw, fs=20.0):
    min_len = min(len(z_raw), len(n_raw), len(e_raw))
    if min_len < 600:
        return None

    z = np.array(z_raw[:min_len], dtype=np.float64)
    n = np.array(n_raw[:min_len], dtype=np.float64)
    e = np.array(e_raw[:min_len], dtype=np.float64)

    # Demean
    z = z - np.mean(z)
    n = n - np.mean(n)
    e = e - np.mean(e)

    std_z = np.std(z)
    std_n = np.std(n)
    std_e = np.std(e)
    if std_z < 1e-10 or std_n < 1e-10 or std_e < 1e-10:
        return None

    h = np.sqrt(n**2 + e**2)

    total_3c_energy = np.sum(z**2 + n**2 + e**2) + 1e-12
    h_energy_frac = float(np.sum(h**2) / total_3c_energy)
    hv_rms_ratio = float(np.log10((np.sqrt(np.mean(h**2)) + 1e-10) / (np.sqrt(np.mean(z**2)) + 1e-10)))

    # P-wave arrival index (10s at 20Hz = 200)
    p_idx = int(10.0 * fs)
    p_burst_z = z[p_idx : p_idx + int(3.0 * fs)]          # early P-wave burst
    p_burst_h = h[p_idx : p_idx + int(3.0 * fs)]
    s_coda_h = h[p_idx + int(5.0 * fs) : min(min_len, p_idx + int(25.0 * fs))] # shear wave window
    s_coda_z = z[p_idx + int(5.0 * fs) : min(min_len, p_idx + int(25.0 * fs))]

    if len(p_burst_z) == 0 or len(s_coda_h) == 0:
        return None

    # Cross-Component P/S Ratio (P on Z vs S on H)
    rms_p_z = np.sqrt(np.mean(p_burst_z**2)) + 1e-10
    rms_s_h = np.sqrt(np.mean(s_coda_h**2)) + 1e-10
    ps_ratio_cross = float(np.log10(rms_p_z / rms_s_h))

    # 3D Polarization & Rectilinearity
    analysis_win = np.vstack([
        z[p_idx : p_idx + int(10.0 * fs)],
        n[p_idx : p_idx + int(10.0 * fs)],
        e[p_idx : p_idx + int(10.0 * fs)]
    ])
    cov_matrix = np.cov(analysis_win)
    eigvals = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
    l1, l2, l3 = max(eigvals[0], 1e-12), max(eigvals[1], 1e-12), max(eigvals[2], 1e-12)

    rectilinearity = float(1.0 - ((l2 + l3) / (2.0 * l1)))
    planarity = float(1.0 - (2.0 * l3 / (l1 + l2)))

    # Spectral analysis on Z and H
    def get_spectral_props(sig):
        sig_norm = sig / (np.std(sig) + 1e-10)
        win_sig = sig_norm[p_idx : min(len(sig_norm), p_idx + int(25.0 * fs))]
        w = np.hanning(len(win_sig))
        psd = np.abs(np.fft.rfft(win_sig * w))**2
        freqs = np.fft.rfftfreq(len(win_sig), 1.0 / fs)
        tot_p = np.sum(psd) + 1e-12

        low_p = np.sum(psd[(freqs >= 0.5) & (freqs <= 1.5)]) + 1e-12
        high_p = np.sum(psd[(freqs >= 3.5) & (freqs <= 8.0)]) + 1e-12
        spec_ratio = float(np.log10(high_p / low_p))

        centroid = float(np.sum(freqs * psd) / tot_p)
        spread = float(np.sqrt(np.sum(((freqs - centroid)**2) * psd) / tot_p))

        p_norm = psd / tot_p
        p_norm = p_norm[p_norm > 0]
        entropy = float(-np.sum(p_norm * np.log2(p_norm)))

        return spec_ratio, centroid, spread, entropy, high_p, low_p

    s_rat_z, cent_z, spread_z, ent_z, high_pz, low_pz = get_spectral_props(z)
    s_rat_h, cent_h, spread_h, ent_h, high_ph, low_ph = get_spectral_props(h)

    # Cross-phase spectral ratio: High-frequency P on Z vs Low-frequency S on H
    cross_phase_spectral_ratio = float(np.log10(high_pz / low_ph))

    # Coda decays
    coda_decay_z = float(rms_p_z / (np.sqrt(np.mean(s_coda_z**2)) + 1e-10))
    coda_decay_h = float((np.sqrt(np.mean(p_burst_h**2)) + 1e-10) / rms_s_h)

    # Kurtosis
    kurt_z = float(kurtosis(z[p_idx : min(min_len, p_idx + int(25.0 * fs))]))
    kurt_h = float(kurtosis(h[p_idx : min(min_len, p_idx + int(25.0 * fs))]))

    return {
        # Cross-Component 3D Features (Shear Wave Physics & 3D Polarization)
        "ps_ratio_cross": ps_ratio_cross,
        "hv_rms_ratio": hv_rms_ratio,
        "h_energy_frac": h_energy_frac,
        "cross_phase_spectral_ratio": cross_phase_spectral_ratio,
        "rectilinearity": rectilinearity,
        "planarity": planarity,
        # Vertical Z Features (Compressional Wave Physics)
        "spec_ratio_z": s_rat_z,
        "spec_centroid_z": cent_z,
        "spec_spread_z": spread_z,
        "spec_entropy_z": ent_z,
        "coda_decay_z": coda_decay_z,
        "kurt_z": kurt_z,
        # Horizontal H Features (Shear Wave Spectral Physics)
        "spec_ratio_h": s_rat_h,
        "spec_centroid_h": cent_h,
        "spec_spread_h": spread_h,
        "spec_entropy_h": ent_h,
        "coda_decay_h": coda_decay_h,
        "kurt_h": kurt_h
    }

def main():
    print("Loading 3-component earthquake and nuclear explosion datasets...")
    eq_data = np.load(get_data_path("earthquake_data.npz"))
    exp_data = np.load(get_data_path("explosion_data.npz"))

    rows = []

    # Process Earthquakes (label = 0)
    for i in range(len(eq_data["z"])):
        f = extract_3c_trace_features(
            eq_data["z"][i], eq_data["n"][i], eq_data["e"][i], fs=20.0
        )
        if f is not None:
            f.update({
                "evid": str(eq_data["evid"][i]),
                "station": str(eq_data["station"][i]),
                "label": 0,
                "event_type": "earthquake"
            })
            rows.append(f)

    # Process Nuclear Explosions (label = 1)
    for i in range(len(exp_data["z"])):
        f = extract_3c_trace_features(
            exp_data["z"][i], exp_data["n"][i], exp_data["e"][i], fs=20.0
        )
        if f is not None:
            f.update({
                "evid": str(exp_data["evid"][i]),
                "station": str(exp_data["station"][i]),
                "label": 1,
                "event_type": "explosion"
            })
            rows.append(f)

    df_3c = pd.DataFrame(rows)
    print(f"\nExtracted 3C features for {len(df_3c)} total traces.")
    print("Class distribution:")
    print(df_3c["event_type"].value_counts())

    out_file = get_data_path("features.csv")
    df_3c.to_csv(out_file, index=False)
    print(f"Saved pure 3-component feature dataset to {out_file} successfully!")

if __name__ == "__main__":
    main()

