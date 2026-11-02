"""
prepare_explosions.py
Extracts 3-component (Z, R, T / Z, N, E) recordings of underground nuclear tests
from GTUNE (final_llnl_uncut_data.h5).
Slices from 10s pre-P to 50s post-P and resamples each channel to 20 Hz (1200 samples).
Saves to 'explosion_data.npz'.
"""

import pandas as pd
import numpy as np
import scipy.signal as signal

def main():
    print("Loading LLNL uncut dataset from GTUNE...")
    df = pd.read_hdf("GTUNE data/final_llnl_uncut_data.h5", key="df")
    
    target_fs = 20.0
    target_samples = 1200 # 60 seconds

    z_list, n_list, e_list = [], [], []
    evids, stas, dists = [], [], []

    groups = df.groupby(["ev_id", "station"])
    print(f"Total station-event groups: {len(groups)}")

    for (ev, sta), grp in groups:
        z_row = grp[grp["chn"].str.endswith("Z") | (grp["chn"] == "V")]
        n_row = grp[grp["chn"].str.endswith(("N", "R")) | (grp["chn"].isin(["N", "R"]))]
        e_row = grp[grp["chn"].str.endswith(("E", "T")) | (grp["chn"].isin(["E", "T"]))]

        if len(z_row) > 0 and len(n_row) > 0 and len(e_row) > 0:
            z_wf = z_row["waveform"].iloc[0]
            n_wf = n_row["waveform"].iloc[0]
            e_wf = e_row["waveform"].iloc[0]
            fs = float(z_row["smprt"].iloc[0])
            p_epoch = float(z_row["phase_epoch"].iloc[0])
            t_start = float(z_row["trace_start"].iloc[0])
            p_offset = int((p_epoch - t_start) * fs)

            start_idx = p_offset - int(10.0 * fs)
            end_idx = p_offset + int(50.0 * fs)
            min_len = min(len(z_wf), len(n_wf), len(e_wf))

            if start_idx >= 0 and end_idx <= min_len:
                z_cut = z_wf[start_idx:end_idx]
                n_cut = n_wf[start_idx:end_idx]
                e_cut = e_wf[start_idx:end_idx]

                num_resamp = int(len(z_cut) * (target_fs / fs))
                z_res = signal.resample(z_cut, num_resamp)[:target_samples]
                n_res = signal.resample(n_cut, num_resamp)[:target_samples]
                e_res = signal.resample(e_cut, num_resamp)[:target_samples]

                if len(z_res) == target_samples and len(n_res) == target_samples and len(e_res) == target_samples:
                    if np.std(z_res) > 1e-8 and np.std(n_res) > 1e-8 and np.std(e_res) > 1e-8:
                        z_list.append(z_res.astype(np.float32))
                        n_list.append(n_res.astype(np.float32))
                        e_list.append(e_res.astype(np.float32))
                        evids.append(str(ev))
                        stas.append(sta)
                        dists.append(float(z_row["delta"].iloc[0]) if "delta" in z_row else 50.0)

    # Subsample to a balanced set of ~180-200 explosion triplets across unique events
    df_meta = pd.DataFrame({"evid": evids, "idx": range(len(evids))})
    keep_indices = df_meta.groupby("evid").head(3)["idx"].tolist()
    if len(keep_indices) > 200:
        np.random.seed(42)
        keep_indices = np.random.choice(keep_indices, size=200, replace=False)

    z_arr = np.array(z_list)[keep_indices]
    n_arr = np.array(n_list)[keep_indices]
    e_arr = np.array(e_list)[keep_indices]
    evid_arr = np.array(evids)[keep_indices]
    sta_arr = np.array(stas)[keep_indices]
    dist_arr = np.array(dists)[keep_indices]

    np.savez_compressed(
        "explosion_data.npz",
        z=z_arr,
        n=n_arr,
        e=e_arr,
        evid=evid_arr,
        station=sta_arr,
        dist=dist_arr
    )
    print(f"\nSaved {len(z_arr)} balanced 3-component explosion triplets across {len(set(evid_arr))} unique nuclear tests to explosion_data.npz!")

if __name__ == "__main__":
    main()

