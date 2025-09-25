# src/heterogeneities/processing.py

import obspy

import numpy as np
import pandas as pd
from obspy.core.trace import Trace
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from future.builtins import *  # NOQA
from datetime import timedelta
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.inventory import read_inventory
import numpy as np
from obspy.clients.fdsn.client import Client




def linear_interpolation(trace: Trace, interpolation_limit: int | None = 1):
    """
    Interpolate -1 gaps linearly inside a Trace.
    Returns the original mask (before interpolation).
    """
    trace.data = np.ma.masked_where(trace.data == -1, trace.data)
    original_mask = np.ma.getmask(trace.data)

    s = pd.Series(trace.data)
    s.interpolate(
        method="linear",
        axis=0,
        limit=interpolation_limit,
        inplace=True,
        limit_direction=None,
        limit_area="inside",
        downcast=None,
    )
    s.fillna(-1.0, inplace=True)

    # keep dtype float to avoid truncation after processing
    trace.data = s.to_numpy(dtype=float)
    trace.data = np.ma.masked_where(trace.data == -1, trace.data)
    return original_mask


# Helper function: compute_spectrogram
def compute_spectrogram(trace, window_sec=0.1, db_limits=None):

    fs = trace.stats.sampling_rate
    print(f"Sampling rate: {fs} Hz")
    nfft = int(fs * window_sec)
    noverlap = int(nfft * 0.4)
    x = np.asarray(trace.data, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    Pxx, freqs, bins = plt.mlab.specgram(x, NFFT=nfft, Fs=fs, noverlap=noverlap)
    dB = 10 * np.log10(np.maximum(Pxx, 1e-30))
    if db_limits is not None:
        dB = np.clip(dB, db_limits[0], db_limits[1])
    extent = [bins[0], bins[-1], freqs[0], freqs[-1]]
    return dB, extent, freqs, bins



# Helper function: preprocess_trace (unchanged)
def preprocess_trace(trace):
    """
    Pre-processing workflow:
      1. Fill masked data with interpolation if possible
      2. Detrend (linear) and demean
      3. Apply de-glitching
    """
    proc = trace.copy()
    
    if np.ma.is_masked(proc.data):
        mask = np.ma.getmaskarray(proc.data)
        if np.all(mask):
            proc.data = np.zeros_like(proc.data, dtype=float)
        else:
            x = np.arange(len(proc.data))
            valid_indices = ~mask
            if np.any(valid_indices):
                valid_x = x[valid_indices]
                valid_y = proc.data[valid_indices]
                proc.data = np.interp(x, valid_x, valid_y)
            else:
                proc.data = proc.data.filled(0)
    
    proc.detrend(type='linear')
    proc.detrend(type='demean')
    
    #proc.data = deglitch_data(proc.data, threshold=3)
    
    return proc

R_MOON_KM = 1737.4
def _latlon_depth_to_cartesian_vec(lat_deg, lon_deg, depth_km, R_moon=R_MOON_KM):
    """
    Vectorized: lat_deg, lon_deg, depth_km can be numpy arrays or pandas Series.
    Returns (x, y, z) arrays (km) in a Moon-centered Cartesian frame.
    """

    R_moon = 1737.4

    r = R_moon - depth_km
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def compute_and_save_hypo_distances(
    df_time: pd.DataFrame,
    file_path: str,
    R_moon_km: float = R_MOON_KM,
    stations: dict | None = None,
    out_suffix: str = "_with_dist.csv"
):
    """
    Adds hypocentral distance columns to df_time (km) for each station in `stations`
    and saves to a new CSV next to `file_path`.

    Parameters
    ----------
    df_time : DataFrame with columns ["Lat", "Long", "Depth"] (deg, deg, km).
    file_path : Path to the original CSV (used to derive output filename).
    R_moon_km : Lunar radius in km.
    stations : dict mapping short name -> (lat_deg, lon_deg, depth_km).
               If None, defaults to Apollo 14/15/16.
    out_suffix : Suffix appended before .csv in the output filename.

    Returns
    -------
    updated_df : DataFrame (same object as df_time, modified in place).
    out_path : str, path to the saved CSV.
    """
    from heterogeneities.processing import _latlon_depth_to_cartesian_vec
    # Default stations (lat, lon in deg; depth in km)
    if stations is None:
        stations = {
            "A14": (-3.645, 342.552, 0.0),
            "A15": (26.132,   3.633, 0.0),
            "A16": (-8.976,  15.499, 0.0),
        }

    required_cols = {"Lat", "Long", "Depth"}
    missing = required_cols - set(df_time.columns)
    if missing:
        raise ValueError(f"df_time is missing required columns: {sorted(missing)}")

    # Vectorized event coordinates
    ex, ey, ez = _latlon_depth_to_cartesian_vec(
        df_time["Lat"].to_numpy(),
        df_time["Long"].to_numpy(),
        df_time["Depth"].to_numpy(),
        R_moon=R_moon_km
    )
    event_xyz = np.stack([ex, ey, ez], axis=1)  # shape (N, 3)

    # Compute distance to each station in a vectorized way
    for key, (slat, slon, sdepth) in stations.items():
        sx, sy, sz = _latlon_depth_to_cartesian_vec(slat, slon, sdepth, R_moon=R_moon_km)
        stn = np.array([sx, sy, sz])  # shape (3,)
        d = np.linalg.norm(event_xyz - stn, axis=1)  # km
        df_time[f"hypo_dist_{key}_km"] = d

    # Save updated file
    out_path = file_path if file_path.endswith(".csv") else f"{file_path}.csv"
    out_path = out_path.replace(".csv", f"{out_suffix}")
    df_time.to_csv(out_path, index=False)

    return df_time, out_path