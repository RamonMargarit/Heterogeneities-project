# src/heterogeneities/main.py
from obspy import UTCDateTime
from .io import view_apollo
from heterogeneities.io import plot_spectrogram
from heterogeneities.processing import compute_spectrogram
import matplotlib.pyplot as plt



def main():
    starttime = UTCDateTime("1975-01-29T09:55:00")
    endtime   = UTCDateTime("1975-01-29T10:10:00")
    network   = "XA"
    station   = "S15"
    channel   = "SHZ"
    location  = "*"

    stream = view_apollo(stream=None,
        starttime=starttime,
        endtime=endtime,
        network=network,
        station=station,
        channel=channel,
        location=location,
        plot_seismogram=False,
        plot_response=False,
    )
    st = stream[0].copy()
    dB, extent, freqs, times = compute_spectrogram(st, window_sec=0.5, db_limits=[-250, -170])
    plot_spectrogram(dB, extent, st, vmin=None, vmax=None, show = False)

    try:
        stream.plot(equal_scale=False, method="full", size=(1000, 600), show=True)
    except TypeError:
        pass

    plt.show()
if __name__ == "__main__":
    main()
