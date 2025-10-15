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
            pre_filt = [0.01,0.1, 12, 15] 
            tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL",
                       water_level=None, plot=plot_response)
            
            if plot_response:
                plt.show()
            
            # apply the mask back to the trace 
            tr.data = np.ma.masked_array(tr.data, mask=original_mask)


def get_Apollo(starttime=None, endtime=None, station="A11", channel="SHZ", 
                units="DU", year=None, month=None):
    """
    Download seismogram stream for given parameters.
    
    If units=='DU' (default), returns raw digital units.
    If units=='ACC', instrument response is removed so that output is acceleration.
    
    """
    import numpy as np
    import pandas as pd
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    
    # --- 1. Figure out time window ---
    if (starttime is not None or endtime is not None) and (year is not None or month is not None):
        raise ValueError("Please specify either (starttime, endtime) OR (year, month), not both.")
    
    if year is not None and month is not None:
        # Create a starttime as the first day of the given month
        starttime = UTCDateTime(year, month, 1, 0, 0, 0)
        # Create an endtime as the last second of that same month
        if month == 12:
            endtime = UTCDateTime(year + 1, 1, 1, 0, 0, 0) - 1
        else:
            endtime = UTCDateTime(year, month + 1, 1, 0, 0, 0) - 1
    else:
        # If no year/month was given, we assume starttime/endtime are valid
        if starttime is None or endtime is None:
            raise ValueError("You must provide either (starttime, endtime) OR (year, month).")


    # --- 2. Use FDSN client to get data and process ---
    client = Client("IRIS")

    # Retrieve station metadata (inventory) for response
    inv = client.get_stations(starttime=starttime, endtime=endtime,
                              network='XA', sta=station, loc='*', channel=channel,
                              level="response")
    
    # Retrieve waveforms
    st = client.get_waveforms(network='XA', station=station, channel=channel, location='*',
                              starttime=starttime, endtime=endtime)
    st.merge()
    if not st:
             print(f"Warning: No data returned via FDSN for {station} {channel} between {starttime} and {endtime}")
             return None, [starttime, endtime, station, channel]
    for tr in st:
        if tr.data is not None and len(tr.data) > 0 and np.issubdtype(tr.data.dtype, np.number):
            try:
                # Calculate the median value of the trace data
                # Use np.ma.median if data might already contain masks/NaNs after merging
                if isinstance(tr.data, np.ma.MaskedArray):
                    # Calculate median ignoring masked values
                    median_val = np.ma.median(tr.data)
                else:
                    # Calculate median on the numpy array
                    median_val = np.median(tr.data)

                # Check if median calculation was successful (e.g., not NaN)
                if median_val is not None and not np.isnan(median_val):
                    invalid_value = median_val - 511

                    # Create a boolean mask where data equals the invalid value
                    invalid_mask = (tr.data == invalid_value)

                    # Apply the mask if any invalid values were found
                    if np.any(invalid_mask):
                        # Ensure the data is a masked array before applying the mask
                        if not isinstance(tr.data, np.ma.MaskedArray):
                             tr.data = np.ma.masked_array(tr.data, mask=invalid_mask)
                        else:
                             # Combine with existing mask using logical OR
                             tr.data.mask = np.logical_or(tr.data.mask, invalid_mask)
                        # Optional: print message
                        # num_masked = np.sum(invalid_mask)
                        # print(f"Masked {num_masked} points with invalid value {invalid_value} in trace {tr.id}.")

                # else: # Optional: handle cases where median couldn't be calculated
                #     print(f"Warning: Could not calculate a valid median for trace {tr.id}. Skipping invalid value masking.")

            except Exception as e:
                # Keep the original trace data if masking fails
                print(f"Warning: Error during invalid value masking for trace {tr.id}: {e}. Proceeding with original data.")
    

    # Process gaps with linear interpolation
    for tr in st:
        linear_interpolation(tr, interpolation_limit=1)

    # Remove instrument response if needed
    for tr in st:
        # On Apollo missions, channels can be MH1, MH2, MHZ, or SHZ, etc.
        if tr.stats.channel in ['MH1', 'MH2', 'MHZ']:
            original_mask = linear_interpolation(tr, interpolation_limit=None)
            if units.upper() == 'VEL':
                # Pre-filter band might be different for MH* channels
                pre_filt = [1/350, 1/300, 1.5, 2]
                tr.remove_response(inventory=inv, pre_filt=pre_filt,
                                   output="VEL", water_level=None, plot=False)
            tr.data = np.ma.masked_array(tr.data, mask=original_mask)
        elif tr.stats.channel in ['SHZ']:
            original_mask = linear_interpolation(tr, interpolation_limit=None)
            if units.upper() == 'VEL':
                # Different pre-filter band for SHZ
                #pre_filt = [1/100, 1/50, 20, 22]
                pre_filt = [0.05, 0.1, 14.0, 15.0]
                tr.remove_response(inventory=inv,
                                   output="VEL", water_level=None, pre_filt=pre_filt, plot=True)
            tr.data = np.ma.masked_array(tr.data, mask=original_mask)

    # For simplicity, return just the first Trace in the Stream
    if len(st) == 0:
        print(f"No data for {station} {channel} {starttime}–{endtime}")

    tr = st[0]
    return tr


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
    #ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0.1,25)
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
