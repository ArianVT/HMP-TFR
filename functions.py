import numpy as np
import math
import matplotlib.pyplot as plt
import mne

def get_center(event_probs):
    return np.argmax(event_probs)

def get_tfr(data, freqs, baseline, jobs=2, cycles=2):
    tfr = data.compute_tfr(method="morlet", freqs=freqs, n_cycles=cycles, use_fft=True, average=False, n_jobs=jobs).apply_baseline(baseline, mode='logratio') #get logratio
    tfr._data = tfr._data*10 #multiply to get the log_decibel                           
    return tfr

def get_averaged_tfr(data, info, times, freqs):
    tfr = mne.time_frequency.EpochsTFRArray(info=info,data=data,times=times,freqs=freqs)                           
    return tfr.average()

def get_itc(data, freqs, jobs=2, cycles=2):
    _, itc = data.compute_tfr(method="morlet", freqs=freqs, n_cycles=cycles, use_fft=True, average=True, n_jobs=jobs, return_itc=True)                            
    return itc

def get_events(matrix, probs, freqs, info, labels, baseline=None, instance='tfr', cycles=2, jobs=1, log=True):
    events = []

    for n_event in range(probs.shape[1]):
        if log: print(f"event: {n_event}, trial: ----", end='\r')                                      
        event = []

        #add the centred data for every trail using context
        for trial in range(matrix._data.shape[0]):
            if log: print(f"event: {n_event}, trial: {trial + 1}", end='\r', flush=True)                 
            center = int(get_center(probs[trial, n_event, :])) - 1

            if center > 0:
                match instance:
                    case 'tfr':
                        event.append((matrix._data[trial, :, :, :center], matrix._data[trial, :, :, center], matrix._data[trial, :, :, center+1:]))
                    case 'itc':
                        event.append((matrix._data[trial, :, :center], matrix._data[trial, :, center], matrix._data[trial, :, center+1:]))

        if len(event) > 0:
            center, centred_matrix = center_matrix(event, instance=instance)
            
            match instance:
                case 'tfr':
                    event = get_averaged_tfr(centred_matrix, matrix.info, [(t - center)/matrix.sfreq for t in range(centred_matrix.shape[3])], matrix.freqs)
                    del centred_matrix #delete centred matrix to save data
                    event._data[event._data == -np.inf] = np.nan
                case 'itc':
                    centred_epochs = mne.EpochsArray(np.nan_to_num(centred_matrix, nan=0.0), info, tmin=-(center/info['sfreq']))
                    event = get_itc(centred_epochs, freqs, jobs=jobs, cycles=cycles)

            events.append((event, labels[n_event]))

    return events

def plot_events(total, events, channels, x_axis='adaptive', instance='tfr', topo=True):

    #set z-axis
    match instance:
        case 'itc':
            vlim=(0,1)
        case 'tfr':
            vlim=(-3,3)

    #plot per event 
    _, ax1 = plt.subplots(2,3, figsize=(24,16))
        
    for n, (event, label) in enumerate(events):
        x = (3+n)%3
        y = math.floor(n/3)

        #set x-axis
        if x_axis == 'adaptive':
            tmin, tmax = adaptive_xaxis(event)
        elif type(x_axis) == tuple:
            tmin, tmax = x_axis

        #plot event
        ax1[y,x].set_title(label)
        event.plot(axes=ax1[y,x], picks=channels, vlim=vlim, tmin=tmin, tmax=tmax, show=False, cmap='turbo', combine='mean')
        ax1[y,x].axvline(x=0, color='grey', linestyle='--', label='Event')

    #plot total
    ax1[1,2].set_title("Total")
    tmin, tmax = adaptive_xaxis(total)
    total.plot(picks=channels, axes=ax1[1,2], tmin=tmin, tmax=tmax, vlim=vlim, show=False, cmap='turbo', combine='mean') 

    #plot topo
    if topo:
        _, ax2 = plt.subplots(2,3, figsize=(24,16))
        bands = [
            ((1,4),"Delta"),
            ((4,8),"Theta"),
            ((8,12),"Alpha"),
            ((12,25),"Beta"),
            ((25,40),"Gamma"),
        ]
        for n, ((fmin, fmax), label) in enumerate(bands):
            x = (3+n)%3
            y = math.floor(n/3)
    
            ax2[y,x].set_title(label)
            total.plot_topomap(axes=ax2[y,x], tmin=tmin, tmax=tmax - 0.2, fmin=fmin, fmax=fmax, show_names=True, show=False, cmap='turbo')

        ax2[1,2].set_title("Total")
        total.plot_topomap(axes=ax2[1,2], tmin=tmin, tmax=tmax - 0.2, show_names=True, show=False, cmap='turbo')      

    plt.show()

def center_matrix(matrix, instance='tfr'):
    match instance:
        case 'tfr': #center TFRs (trials, channels, time, frequencies)
                    #create empty matrix
                    left = np.amax([matrix[x][0].shape[2] for x in range(len(matrix))])
                    right = np.amax([matrix[x][2].shape[2] for x in range(len(matrix))])
                    event_data = np.full([len(matrix), 64, 30, left + right + 1], np.nan)
        
                    #fill matrix per trial
                    for n, epoch in enumerate(matrix):
                        event_data[n, :, :, (left-epoch[0].shape[2]):left] = epoch[0]
                        event_data[n, :, :, left] = epoch[1]
                        event_data[n, :, :, (left+1):((left+1) + epoch[2].shape[2])] = epoch[2]
                        
        case 'itc': #center epochs (trials, channels, time)
            #create empty matrix
            left = np.amax([matrix[x][0].shape[1] for x in range(len(matrix))])
            right = np.amax([matrix[x][2].shape[1] for x in range(len(matrix))])
            event_data = np.full([len(matrix), 64, left + right + 1], np.nan)

            #fill matrix per trial
            for n, epoch in enumerate(matrix):
                event_data[n, :, (left-epoch[0].shape[1]):left] = epoch[0]
                event_data[n, :, left] = epoch[1]
                event_data[n, :, (left+1):((left+1) + epoch[2].shape[1])] = epoch[2]

    return left, event_data

def adaptive_xaxis(event):
    i = 0
    left = -np.inf
    right = np.inf 

    #look at the ith matrix from the center until there is an NaN found
    while True:
        if np.isnan(event._data[:,:, int(event.tmin*-event.sfreq + i)]).any(): #there is an NaN found on the right
            if right == np.inf:
                right = i - 1

        if np.isnan(event._data[:,:, int(event.tmin*-event.sfreq - i)]).any(): #there is an NaN found on the left
            if left == -np.inf:
                left = i - 1

        #return if both sides have found an NaN, extend if not
        if left > -np.inf and right < np.inf: 
            return -left/event.sfreq, right/event.sfreq
        else:
            i += 1