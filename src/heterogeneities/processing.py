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
