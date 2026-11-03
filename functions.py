import numpy as np
import math
import matplotlib.pyplot as plt
import mne
import os
import copy

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

def get_itc_baseline(itc, baseline):
    left, right = baseline

    if left != None: left = int(itc.info['sfreq'] * -(itc.tmin - left))
    else: left = 0
    
    if right != None: right = int(itc.info['sfreq']* - right)
    else: right = -1
        
    return np.average(itc._data[:,:,left:right], axis=(0,2))

def get_events(matrix, probs, freqs, info, labels, baseline=None, instance='tfr', cycles=2, jobs=1):
    events = []

    for n_event in range(probs.shape[1]):                                   
        event = []

        #add the centred data for every trail using context
        for trial in range(matrix.shape[0]):                
            center = int(get_center(probs[trial, n_event, :])) - 1

            if center > 0: #only add the data if an event onset was found
                match instance:
                    case 'tfr': event.append((matrix[trial, :, :, :center], matrix[trial, :, :, center], matrix[trial, :, :, center+1:]))
                    case 'itc' | 'eeg': event.append((matrix[trial, :, :center], matrix[trial, :, center], matrix[trial, :, center+1:]))

        if len(event) > 0:
            #if an event onset was found for at least one trial, create an fitting event.
            match instance:
                case 'tfr':
                    center, centred_matrix = center_matrix(event, instance=instance)
                    event = get_averaged_tfr(centred_matrix, info, [(t - center)/info['sfreq'] for t in range(centred_matrix.shape[3])], freqs)
                    del centred_matrix #delete centred matrix to save data
                    event._data[event._data == -np.inf] = np.nan
                case 'itc':
                    center, centred_matrix = center_matrix(event, instance=instance)
                    centred_epochs = mne.EpochsArray(np.nan_to_num(centred_matrix, nan=0.0), info, tmin=-(center/info['sfreq']))
                    event = get_itc(centred_epochs, freqs, jobs=jobs, cycles=cycles)
                    for x, y in enumerate((event.shape[0], event.shape[1])): event._data[x, :, y] -= baseline
                    del centred_matrix #delete centred matrix to save data
                case 'eeg':
                    center, centred_matrix = center_matrix(event, instance=instance)
                    event = mne.EpochsArray(np.nan_to_num(centred_matrix, nan=0.0), info, tmin=-(center/info['sfreq']))
                    del centred_matrix #delete centred matrix to save data


            events.append((event, f"{n_event}_{labels[n_event]}")) #add event to the events list

    return events

def center_matrix(matrix, instance='tfr'):
    match instance:
        case 'tfr': #center TFRs/phases (trials, channels, frequencies, time)
            #create empty matrix
            left = np.amax([matrix[x][0].shape[2] for x in range(len(matrix))])
            right = np.amax([matrix[x][2].shape[2] for x in range(len(matrix))])
            event_data = np.empty([len(matrix), 64, 30, left + right + 1])
            event_data.fill(np.nan)
        
            #fill matrix per trial
            for n, epoch in enumerate(matrix):
                event_data[n, :, :, (left-epoch[0].shape[2]):left] = epoch[0]
                event_data[n, :, :, left] = epoch[1]
                event_data[n, :, :, (left+1):((left+1) + epoch[2].shape[2])] = epoch[2]

        case 'itc'| 'eeg': #center epochs (trials, channels, time)
            #create empty matrix
            left = np.amax([matrix[x][0].shape[1] for x in range(len(matrix))])
            right = np.amax([matrix[x][2].shape[1] for x in range(len(matrix))])
            event_data = np.full([len(matrix), matrix[0][1].shape[0], left + right + 1], np.nan)

            #fill matrix per trial
            for n, epoch in enumerate(matrix):
                event_data[n, :, (left-epoch[0].shape[1]):left] = epoch[0]
                event_data[n, :, left] = epoch[1]
                event_data[n, :, (left+1):((left+1) + epoch[2].shape[1])] = epoch[2]

    return left, event_data

def combine_participants(directory_name, centers=None, instance='tfr', combine=True):
    
    #create dict
    dict = {}
    for participant in os.scandir(f"output/{directory_name}/{instance}"):
        if not participant.name.endswith(".pkl"):
            for event in os.scandir(f"output/{directory_name}/{instance}/{participant.name}"):
                #set key
                if event.name.startswith("total"): key = 'total'
                else: key = f"{event.name.split("_")[0]}: {event.name.split("_")[1]}"

                #set value
                match instance:
                    case 'tfr' | 'itc': value =  mne.time_frequency.read_tfrs(event.path)
                    case 'eeg': value = mne.read_epochs(event.path).average()
                    case 'phases': value = (np.load(event.path), int(event.name.split("_")[2])) 

                #update dict
                if key not in dict.keys(): dict.update({key: [value]})
                else: dict[key].append(value)
    
    #allign dicts according to fitting format
    match instance:
        case 'tfr' | 'itc' | 'eeg':
            for key, matrix in dict.items():
                tmin = np.amax([matrix[x].tmin for x in range(len(matrix))])
                tmax = np.amin([matrix[x].tmax for x in range(len(matrix))])
                alligned_matrix = [tfr.crop(tmin=tmin, tmax=tmax) for tfr in matrix]

                if combine:
                    summed_matrix = np.zeros(alligned_matrix[0]._data.shape)
                    for i in range(len(alligned_matrix)): summed_matrix += alligned_matrix[i]._data
                    summed_matrix /= len(alligned_matrix)

                    mean_matrix = alligned_matrix[0]
                    mean_matrix._data = summed_matrix

                    dict[key] = mean_matrix
                else:
                    dict[key] = alligned_matrix

        case 'phases':
            for key, matrix in dict.items():
                t_left = np.amin([matrix[x][1] for x in range(len(matrix))])
                t_right = np.amin([(matrix[x][0].shape[1] - matrix[x][1]) for x in range(len(matrix))])
                alligned_matrix = []

                for x in range(len(matrix)):
                    phases = matrix[x][0]
                    center = matrix[x][1]
                    alligned_matrix.append(phases[:, center-t_left:center+t_right])

                if combine: 
                    mean_matrix = alligned_matrix[0]
                    for i in range(len(alligned_matrix)-1):
                        mean_matrix += alligned_matrix[i+1]

                    mean_matrix /= len(alligned_matrix)

                    dict[key] = (mean_matrix, t_left)
                else:
                    dict[key] = alligned_matrix
    
    if centers != None: return (dict, centers)
    else: return dict          

def adaptive_xaxis(event, measure='time'):

    #initialize variables
    i = 0                       #i (int): an index for how far the pointer is removed from the center
    left = -np.inf              #left (int): starts at -np.inf, converges to i - 1 once the pointer runs into and NaN on the left side
    right = np.inf              #right (int): starts at np.inf, converges to i - 1 once the pointer runs into and NaN on the right side


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
            match measure: 
                case 'time': return -left/event.sfreq, right/event.sfreq
                case 'samples': return math.ceil((-event.tmin*event.sfreq) - left), math.floor((-event.tmin*event.sfreq) + right + 1)
        else:
            i += 1

def plot_topos(dict, vlim=None):

    #extract events from dictionary
    events = copy.copy(dict)
    events.pop('total')

    #set variables
    bandwidths = {
        'delta': (1,4),
        'theta': (4,8),
        'alpha': (8,12),
        'beta': (12,25),
        'gamma': (25,40)
    }
    n_events = len(events.values())
    n_bandwidths = len(bandwidths.values())

    #plot the topo for each (bandwidth: y, event: x) combination
    _, ax = plt.subplots(n_bandwidths, n_events, figsize=(6 + 6*n_events, 32))
    for y, (fname, (flow, fhigh)) in enumerate(bandwidths.items()):
        for x, (label, event) in enumerate(events.items()): 
            event.plot_topomap(tmin=0, tmax=0, fmin=flow, fmax=fhigh, axes=ax[y, x], show=False, cmap='turbo', vlim=vlim, show_names=True)
            ax[y, x].set_title(label) 

        #y-axis label
        ax[y, 0].set_ylabel(fname, fontsize=16, rotation=90, labelpad=20)

def plot_tfr(dict, eeg, channels, vlim=(-3,3), tmin=-0.2, tmax=0.2, statistic_dict=None):

    #create subplots
    _, ax = plt.subplots(len(channels) + 1, len(dict.values()), figsize=(32,6 + 6*len(dict.values())))

    #for every (event: x) plot raw event on the first row
    for x, (label, event) in enumerate(eeg.items()):
        mne.viz.plot_topomap(event._data[:, int(-event.tmin*event.info['sfreq'])], event.info, show=False, names=event.info['ch_names'], cmap='turbo', axes=ax[0, x])
        ax[0, x].set_title(label)
        ax[0, x].legend()
        ax[0, x].set_ylabel("amplitude (\u03BCV)")

    #remove the empty subplot on the first row
    ax[0, -1].remove() 


    #plot the tfr for each (channel: y, event: x) combination
    for y, channel in enumerate(channels):
        for x, (label, event) in enumerate(dict.items()):

            #if permutation, create mask
            if statistic_dict != None: mask = permutation_test(statistic_dict, label, channel)

            match label: 
                #plot total tfr
                case 'total':
                    min, max = adaptive_xaxis(event)
                    event.plot(cmap='turbo', picks=[channel], tmin=min, tmax=max, vlim=vlim, show=False, axes=ax[y + 1, x])

                #plot event tfr
                case _:
                    if 'mask' in locals():
                        event.plot(cmap='gray', picks=[channel], tmin=tmin, tmax=tmax, vlim=vlim, show=False, axes=ax[y + 1, x], colorbar=False)
                        
                        if type(mask) == np.ndarray:
                            bar = True 
                            for m in range(mask.shape[0]): 
                                event.plot(cmap='turbo', picks=[channel], tmin=tmin, tmax=tmax, vlim=vlim, show=False, axes=ax[y + 1, x], mask=mask[m, :, int(event.tmin*-event.sfreq-40):int(event.tmin*-event.sfreq+41)], colorbar=bar)
                                bar = False
                    else:
                        event.plot(cmap='turbo', picks=[channel], tmin=tmin, tmax=tmax, vlim=vlim, show=False, axes=ax[y + 1, x])

            ax[y+1, x].set_title(label)

        ax[y+1, 0].annotate(channel, xy=(-0.28, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=18)

def permutation_test(dict, label, channel):
    data = np.stack([x._data.astype('float16') for x in dict[label]])
    event = dict[label][0]
    data = np.average(data, axis=1) - data[:,event.info['ch_names'].index(channel),:,:]


    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(data, n_permutations=1000, tail=0, n_jobs=-1, out_type='mask')
    indices = np.where(cluster_pv <= 0.05)
    if indices[0].shape[0] > 0: 
        return np.stack([clusters[x] for x in indices[0]])
    else: 
        print("No mask found for TFR")
        return None
  