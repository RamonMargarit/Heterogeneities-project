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