# src/heterogeneities/main.py
from obspy import UTCDateTime
from .io import view_apollo

def main():
    starttime = UTCDateTime("1976-03-06T10:05:00")
    endtime   = UTCDateTime("1976-03-06T10:35:00")
    network   = "XA"
    station   = "S15"
    channel   = "SHZ"
    location  = "*"

    view_apollo(stream=None,
        starttime=starttime,
        endtime=endtime,
        network=network,
        station=station,
        channel=channel,
        location=location,
        plot_seismogram=True,
        plot_response=False,
    )

if __name__ == "__main__":
    main()
