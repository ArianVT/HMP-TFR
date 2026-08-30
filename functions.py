import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import mne

def get_context(event_probs, left, right):
    event = np.argmax(event_probs)
    context = np.zeros(3)
    
    context[0] = max(0, event - left)
    context[1] = event
    context[2] = event + right

    return context #context in samples   

def get_tfr(data, freqs, baseline, avg=False, jobs=2, cycles=2):
    tfr = data.compute_tfr(method="morlet", freqs=freqs, n_cycles=cycles, use_fft=True, average=avg, n_jobs=jobs).apply_baseline(baseline, mode='logratio')
    tfr._data = tfr._data*10                            
    return tfr

def get_itc(data, freqs, jobs=2, cycles=2):
    _, itc = data.compute_tfr(method="morlet", freqs=freqs, n_cycles=cycles, use_fft=True, average=True, n_jobs=jobs, return_itc=True)                            
    return itc
    
def get_tfr_events(tfr, probs, left, right, sfreq, log=True):
    if log: print("retreiving TFR events")
    tfr_shape = tfr.shape
    interval = left+right 
    tfr_events = []                                                                         
    tfr_event = copy.copy(tfr).crop(tmin = 0, tmax = (interval/sfreq), include_tmax=False)                                                                                                                                          
    
    #create centred tfr_event per event onset
    for event in range(probs.shape[1]):
        if log: print(f"event: {event}, trial: ----", end='\r')                                      

        #create NaNs matrix for the event data
        tfr_event = copy.deepcopy(tfr_event)
        tfr_event_data = np.full([tfr_shape[0], tfr_shape[1], tfr_shape[2], interval], np.nan)

        #add the centred data for every trail using context
        for trial in range(tfr._data.shape[0]):
            if log: print(f"event: {event}, trial: {trial + 1}", end='\r', flush=True)                 
            begin, _, end = get_context(probs[trial, event, :], left, right)                            #use context to find center

            if end - begin == interval:
                tfr_event_data[trial, :, :, :] = tfr._data[trial, :, :, int(begin):int(end)]            #change event data to centred TFR

        #create an averaged TFR using the centred data
        tfr_event._data = tfr_event_data                                                        
        averaged_tfr_event = tfr_event.average()

        #add the averaged TFR to the events array 
        tfr_events.append(averaged_tfr_event) 

    if log: print("retreived TFR events succesfully")
    return tfr_events

def get_itc_events(epochs, probs, sfreq, freqs, info, cycles=2, jobs=1, log=True):
    itc_events = []

    for event in range(probs.shape[1]):
        if log: print(f"event: {event}, trial: ----", end='\r')                                      
        itc_event = []

        #add the centred data for every trail using context
        for trial in range(epochs._data.shape[0]):
            if log: print(f"event: {event}, trial: {trial + 1}", end='\r', flush=True)                 
            _, center, _ = get_context(probs[trial, event, :], 20, 20)
            center = int(center) - 1
            if center > 0:
                itc_event.append((epochs._data[trial, :, :center], epochs._data[trial, :, center], epochs._data[trial, :, center+1:]))

        if len(itc_event) > 0:
            center, data = center_matrix(itc_event)
            centred_epochs = mne.EpochsArray(np.nan_to_num(data, nan=0.0), info, tmin=-(center/sfreq))
            itc_event = get_itc(centred_epochs, freqs, jobs=jobs, cycles=cycles)

        print(f"event {event}:{center}")
        itc_events.append(itc_event)

    return itc_events


def plot_tfrs(tfr, events, channels, labels, x_axis):
    _, ax = plt.subplots(2,3, figsize=(24,16))
    for n, event in enumerate(events):
        x = (3+n)%3
        y = math.floor(n/3)
        ax[y,x].set_title(labels[n])
        if np.any(~np.isnan(event._data)):
            event.plot(axes=ax[y,x], picks=channels, vlim=(-3,3), show=False, cmap='turbo', combine='mean')
            t_event = ax[y,x].get_xticks()[-1]/2
            ax[y,x].axvline(x=t_event, color='grey', linestyle='--', label='Event')
            ax[y,x].set_xticklabels([round(x - t_event, 2) for x in ax[y,x].get_xticks()])

    ax[1,2].set_title("Total")
    tfr.plot(picks=channels, axes=ax[1,2], tmin=x_axis[0], tmax=x_axis[1], vlim=(-3,3), show=False, cmap='turbo', combine='mean')        

    plt.show()

def plot_itcs(itc, events, channels, labels):
    _, ax = plt.subplots(2,3, figsize=(24,16))
    for n, event in enumerate(events):
        x = (3+n)%3
        y = math.floor(n/3)
        ax[y,x].set_title(labels[n])
        if type(event) != list:
            for channel in channels:
                tmin, tmax = adaptive_xaxis(event)
                event.plot(picks=channel, axes=ax[y,x], tmin=tmin, tmax=tmax, vlim=(0,1), show=False, cmap='turbo', combine='mean')

    ax[1,2].set_title("Total")
    itc.plot(picks=channels, axes=ax[1,2], vlim=(0,1), tmin=-0.2, tmax=1, show=False, cmap='turbo', combine='mean')

    plt.show()

def plot_topomaps(tfr):
    _, ax = plt.subplots(2,3, figsize=(24,16))
    ax[0,0].set_title("Delta")
    tfr.plot_topomap(axes=ax[0,0], tmin=0, tmax=0.8, fmin=1, fmax=4, show_names=True, show=False, cmap='turbo')
    ax[0,1].set_title("Theta")
    tfr.plot_topomap(axes=ax[0,1], tmin=0, tmax=0.8, fmin=4, fmax=8, show_names=True, show=False, cmap='turbo')
    ax[0,2].set_title("Alpha")
    tfr.plot_topomap(axes=ax[0,2], tmin=0, tmax=0.8, fmin=8, fmax=12, show_names=True, show=False, cmap='turbo')
    ax[1,0].set_title("Beta")
    tfr.plot_topomap(axes=ax[1,0], tmin=0, tmax=0.8, fmin=12, fmax=25, show_names=True, show=False, cmap='turbo')
    ax[1,1].set_title("Gamma")
    tfr.plot_topomap(axes=ax[1,1], tmin=0, tmax=0.8, fmin=25, show_names=True, show=False, cmap='turbo')
    ax[1,2].set_title("Total")
    tfr.plot_topomap(axes=ax[1,2], tmin=0, tmax=0.8, show_names=True, show=False, cmap='turbo')
    plt.show()

def center_matrix(epochs):
    left = np.amax([epochs[x][0].shape[1] for x in range(len(epochs))])
    right = np.amax([epochs[x][2].shape[1] for x in range(len(epochs))])
    event_data = np.full([len(epochs), 64, left + right + 1], np.nan)
    for n, epoch in enumerate(epochs):
        event_data[n, :, (left-epoch[0].shape[1]):left] = epoch[0]
        event_data[n, :, left] = epoch[1]
        event_data[n, :, (left+1):((left+1) + epoch[2].shape[1])] = epoch[2]

    return left, event_data

def adaptive_xaxis(event):
    i = 0
    left = -np.inf
    right = np.inf 

    while True:
        if np.isnan(event._data[:,:, int(event.tmin*-200 + i)]).any():
            if right == np.inf:
                right = i - 1

        if np.isnan(event._data[:,:, int(event.tmin*-200 - i)]).any():
            if left == -np.inf:
                left = i - 1

        if left > -np.inf and right < np.inf:
            return -left/event.sfreq, right/event.sfreq

        i += 1