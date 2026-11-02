"""
download_earthquakes.py
Downloads synchronized 3-component (Z, N, E / Z, 1, 2) broadband seismic waveforms
from EarthScope/IRIS for moderate natural earthquakes (M 5.0 - 6.2).
Resamples each channel to 20 Hz and standardizes to 1200 samples (60s window).
Saves dataset to 'earthquake_data.npz'.
"""

import time
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import scipy.signal as signal

def main():
    print("Initializing FDSN clients for 3-Component EarthScope retrieval...")
    client_usgs = Client("USGS")
    client_iris = Client("EARTHSCOPE")
    taup_model = TauPyModel(model="iasp91")

    stations = [
        {"net": "IU", "sta": "ANMO", "lat": 34.9459, "lon": -106.4572},
        {"net": "IU", "sta": "COR",  "lat": 44.5855, "lon": -123.3044},
        {"net": "IU", "sta": "TUC",  "lat": 32.3098, "lon": -110.7846},
        {"net": "IU", "sta": "HRV",  "lat": 42.5064, "lon": -71.5583},
        {"net": "IU", "sta": "KIP",  "lat": 21.4233, "lon": -157.9947},
    ]

    print("Querying USGS catalog for moderate global earthquakes (2020-2023)...")
    events = client_usgs.get_events(
        minmagnitude=5.2,
        maxmagnitude=6.2,
        starttime=UTCDateTime("2020-01-01"),
        endtime=UTCDateTime("2023-01-01"),
        limit=80
    )
    print(f"Retrieved {len(events)} candidate earthquakes.")

    target_fs = 20.0
    target_samples = 1200 # 60 seconds

    z_list, n_list, e_list = [], [], []
    evids, stas, dists = [], [], []

    for ev in events:
        origin = ev.origins[0]
        ev_time = origin.time
        ev_lat = origin.latitude
        ev_lon = origin.longitude
        ev_depth_km = origin.depth / 1000.0 if origin.depth else 10.0
        evid = str(ev.resource_id).split("/")[-1]

        for s in stations:
            dist_deg = locations2degrees(ev_lat, ev_lon, s["lat"], s["lon"])
            if dist_deg < 20.0 or dist_deg > 90.0:
                continue

            arrivals = taup_model.get_travel_times(
                source_depth_in_km=ev_depth_km,
                distance_in_degree=dist_deg,
                phase_list=["P", "Pn"]
            )
            if not arrivals:
                continue

            p_travel_sec = arrivals[0].time
            p_arrival_utc = ev_time + p_travel_sec
            t_start = p_arrival_utc - 10.0
            t_end = p_arrival_utc + 50.0

            try:
                st = client_iris.get_waveforms(
                    network=s["net"],
                    station=s["sta"],
                    location="*",
                    channel="BH?",
                    starttime=t_start,
                    endtime=t_end
                )
                if len(st) < 3:
                    continue

                tr_z = [t for t in st if t.stats.channel.endswith("Z")]
                tr_n = [t for t in st if t.stats.channel.endswith(("N", "1", "R"))]
                tr_e = [t for t in st if t.stats.channel.endswith(("E", "2", "T"))]

                if not tr_z or not tr_n or not tr_e:
                    continue

                z_tr, n_tr, e_tr = tr_z[0], tr_n[0], tr_e[0]
                
                for tr in [z_tr, n_tr, e_tr]:
                    tr.detrend("demean")
                    tr.resample(target_fs)

                z_data = z_tr.data[:target_samples]
                n_data = n_tr.data[:target_samples]
                e_data = e_tr.data[:target_samples]

                min_len = min(len(z_data), len(n_data), len(e_data))
                if min_len < target_samples:
                    pad = target_samples - min_len
                    z_data = np.pad(z_data[:min_len], (0, pad))
                    n_data = np.pad(n_data[:min_len], (0, pad))
                    e_data = np.pad(e_data[:min_len], (0, pad))

                if np.std(z_data) < 1e-8 or np.std(n_data) < 1e-8 or np.std(e_data) < 1e-8:
                    continue

                z_list.append(z_data.astype(np.float32))
                n_list.append(n_data.astype(np.float32))
                e_list.append(e_data.astype(np.float32))
                evids.append(evid)
                stas.append(s["sta"])
                dists.append(dist_deg)

                print(f"[{len(z_list)}] Downloaded 3C for {s['sta']} (Event {evid})")

                if len(z_list) >= 150:
                    break
            except Exception:
                continue

        if len(z_list) >= 150:
            break

    np.savez_compressed(
        "earthquake_data.npz",
        z=np.array(z_list),
        n=np.array(n_list),
        e=np.array(e_list),
        evid=np.array(evids),
        station=np.array(stas),
        dist=np.array(dists)
    )
    print(f"Saved {len(z_list)} 3-component earthquake traces to earthquake_data.npz successfully!")

if __name__ == "__main__":
    main()

