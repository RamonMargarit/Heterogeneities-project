# src/heterogeneities/io.py
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
from .processing import linear_interpolation


#plt.style.use('ggplot')
#plt.rcParams['figure.figsize'] = 10, 4
#plt.rcParams['lines.linewidth'] = 0.5
#plt.rcParams['font.size'] = 16
SECONDS_PER_DAY=3600.*24

def view_apollo(stream=None,
                starttime= UTCDateTime('1976-03-06T10:05:00.0'),
                endtime = UTCDateTime('1976-03-06T11:30:00.0'),
                network='XA',station='S15',channel='SHZ',location='*',
                plot_seismogram=True,plot_response=False):

    client = Client("IRIS")

    # get the response file (wildcards allowed)
    inv = client.get_stations(starttime=starttime, endtime=endtime,
        network=network, sta=station, loc=location, channel=channel,
        level="response")

    if stream is None:
        stream = client.get_waveforms(network=network, station=station, channel=channel, location=location, starttime=starttime, endtime=endtime)

    else:
        stream.trim(starttime=starttime,endtime=endtime)
        
    
    for tr in stream:
        # interpolate across the gaps of one sample 
        linear_interpolation(tr,interpolation_limit=1)
    stream.merge()
    
    for tr in stream:
        # optionally interpolate across any gap 
        # for removing the instrument response from a seimogram, 
        # it is useful to get a mask, then interpolate across the gaps, 
        # then mask the trace again. 
        if tr.stats.channel in ['MH1', 'MH2', 'MHZ']:

            # add linear interpolation but keep the original mask
            original_mask = linear_interpolation(tr,interpolation_limit=None)
            # remove the instrument response
            pre_filt = [0.1,0.3,0.9,1.1]
            tr.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP",
                       water_level=None, plot=plot_response)
            if plot_response:
                tr.plot(equal_scale=False, size=(1000, 600), method="full", show=False)
            tr.data = np.ma.masked_array(tr.data, mask=original_mask)

        elif tr.stats.channel in ['SHZ']:

            # add linear interpolation but keep the original mask
            original_mask = linear_interpolation(tr,interpolation_limit=None)
            # remove the instrument response
            #pre_filt = [0.01,0.1, 12, 15] 
            #tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL",
            #           water_level=None, plot=plot_response)
            tr.remove_response(inventory=inv, output="VEL",
                       water_level=None)
            #if plot_response:
            #    plt.show()
            
            # apply the mask back to the trace 
            tr.data = np.ma.masked_array(tr.data, mask=original_mask)

    if plot_seismogram:
        tr.plot()
        #tr = stream[0]  # un Trace
        #times = np.arange(len(tr.data)) * tr.stats.delta  # delta = sample spacing

        #plt.figure(figsize=(12,5))
        #plt.plot(times, tr.data, linewidth=0.5)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Amplitude")
        #plt.title(f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel} | relative time")
        #plt.show()
    
    return stream  # return it so tests/notebooks can use it

def plot_spectrogram_title(dB, extent, trace, vmin=None, vmax=None, cmap='inferno', show=False, hypo_dist=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(dB, aspect="auto", extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    title = (
        f"{trace.stats.network}."
        f"{trace.stats.station}."
        f"{trace.stats.channel} | "
        f"{trace.stats.starttime.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if hypo_dist is not None:
        title += f" | {hypo_dist:.1f} km"

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Power (dB)", rotation=270, labelpad=20)
    plt.tight_layout()
    if show:
        plt.show()
