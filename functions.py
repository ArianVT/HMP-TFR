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

def get_events(matrix, probs, freqs, info, labels, instance='tfr', cycles=2, jobs=1, log=True):
    events = []

    for n_event in range(probs.shape[1]):
        if log: print(f"event: {n_event}, trial: ----", end='\r')                                      
        event = []

        #add the centred data for every trail using context
        for trial in range(matrix.shape[0]):
            if log: print(f"event: {n_event}, trial: {trial + 1}", end='\r', flush=True)                 
            center = int(get_center(probs[trial, n_event, :])) - 1

            if center > 0: #only add the data if an event onset was found
                match instance:
                    case 'tfr' | 'phases': event.append((matrix[trial, :, :, :center], matrix[trial, :, :, center], matrix[trial, :, :, center+1:]))
                    case 'itc': event.append((matrix[trial, :, :center], matrix[trial, :, center], matrix[trial, :, center+1:]))

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
                    del centred_matrix #delete centred matrix to save data


            events.append((event, f"{n_event}_{labels[n_event]}")) #add event to the events list

    return events

def plot_events(dict, channels, centers, x_axis='adaptive', instance='tfr', topo=True, phases=None, offset=0):
    total = dict['total']
    events = copy.deepcopy(dict)
    events.pop('total')

    #set z-axis
    match instance:
        case 'itc':
            vlim = (0,1)
        case 'tfr':
            vlim = (-3,3)

    #create subplots
    match phases: 
        case None: _, ax1 = plt.subplots(2,3, figsize=(24,16))  #2x3 subplots
        case _: _, ax1 = plt.subplots(4,3, figsize=(24,32))     #4x2 subplots
             
    #plot per event  
    for n, (label, event) in enumerate(events.items()):

        #determine sub plot
        match phases: 
            case None: (y, x) = (math.floor(n/3), (3+n)%3)
            case _: (y, x) = (2*math.floor(n/3), (3+n)%3)

        #set x-axis
        if x_axis == 'adaptive': tmin, tmax = adaptive_xaxis(event)
        elif type(x_axis) == tuple: tmin, tmax = x_axis

        #plot event
        ax1[y,x].set_title(label)
        event.plot(axes=ax1[y,x], picks=channels, vlim=vlim, tmin=tmin, tmax=tmax, show=False, cmap='turbo', combine='mean')
        ax1[y,x].axvline(x=0, color='grey', linestyle='--', label='Event')

        #if given, plot phase variance
        if phases != None: 
            center = phases[label][1]
            match channels:
                case 'all': 
                    data = np.average(phases[label][0], axis=0)
                case _:
                    data = np.average(phases[label][0][[dict['total'].info['ch_names'].index(channel) for channel in channels], :], axis=0)


            lower =  max(0,center-50)
            upper = center+50 
            ax1[y+1,x].plot(data[lower:upper])
            ax1[y+1,x].set_xticklabels([((x+lower)-center)/total.sfreq for x in ax1[y+1,x].get_xticks()])
            pos = ax1[y+1, x].get_position()
            ax1[y+1,x].axvline(x=center-lower, color='grey', linestyle='--', label='Event')
            ax1[y+1, x].set_position([pos.x0, pos.y0+0.09, pos.width*0.8, pos.height*0.5])
            ax1[y+1, x].set_xlabel('Time(s)')
            ax1[y+1, x].set_ylabel('Phase variance')

    #plot total
    match phases:
        case None: (y, x) = (1, 2)
        case _: (y, x) = (2, 2)
        
    ax1[y,x].set_title("Total")
    tmin, tmax = adaptive_xaxis(total)
    total.plot(picks=channels, axes=ax1[y,x], tmin=tmin, tmax=tmax, vlim=vlim, show=False, cmap='turbo', combine='mean')
    old_ticks = ax1[y,x].get_xticks()[1:-1] 
    ax1[y,x].set_xticks(total.times[int(tmin*total.sfreq):int(tmax*total.sfreq)])
    for center in centers: ax1[y,x].axvline(x=round((center)/total.sfreq+min(total.tmin,0),3), color='grey', linestyle='--', label='Event')
    ax1[y,x].set_xticks(old_ticks)

    #if given, plot topo
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
            total.plot_topomap(axes=ax2[y,x], tmin=tmin, tmax=tmax-offset, fmin=fmin, fmax=fmax, vlim=vlim, show_names=True, show=False, cmap='turbo')

        ax2[1,2].set_title("Total")
        total.plot_topomap(axes=ax2[1,2], tmin=tmin, tmax=tmax-offset, vlim=vlim, show_names=True, show=False, cmap='turbo')     

    plt.show()

def center_matrix(matrix, instance='tfr'):
    match instance:
        case 'tfr' | 'phases': #center TFRs/phases (trials, channels, frequencies, time)
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

        case 'itc': #center epochs (trials, channels, time)
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

def combine_participants(directory_name, centers=None, instance='tfr'):
    dict = {}
    match instance:
        case 'tfr' | 'itc':
            for participant in os.scandir(f"output/{directory_name}/{instance}"):
                for event in os.scandir(f"output/{directory_name}/{instance}/{participant.name}"):
                    if event.name.startswith("total"): event_name = 'total'
                    else: event_name = f"{event.name.split("_")[0]}: {event.name.split("_")[1]}"
                    
                    if event_name not in dict.keys(): 
                        dict.update({event_name: [mne.time_frequency.read_tfrs(event.path)]})
                    else: 
                        dict[event_name].append(mne.time_frequency.read_tfrs(event.path))

            for key, matrix in dict.items():
                tmin = np.amax([matrix[x].tmin for x in range(len(matrix))])
                tmax = np.amin([matrix[x].tmax for x in range(len(matrix))])
                cropped_matrix = [tfr.crop(tmin=tmin, tmax=tmax) for tfr in matrix]

                total = cropped_matrix[0]
                for i in range(len(cropped_matrix)-1):
                    total = total.__add__(cropped_matrix[i+1])

                total._data /= len(cropped_matrix)
                dict[key] = total

        case 'phases':
            for participant in os.scandir(f"output/{directory_name}/{instance}"):
                for event in os.scandir(f"output/{directory_name}/{instance}/{participant.name}"):
                    if event.name.startswith("total"): event_name = 'total'
                    else: event_name = f"{event.name.split("_")[0]}: {event.name.split("_")[1]}"
                    
                    if event_name not in dict.keys(): 
                        dict.update({event_name: [(np.load(event.path), int(event.name.split("_")[2]))]})
                    else: 
                        dict[event_name].append((np.load(event.path), int(event.name.split("_")[2])))

            for key, matrix in dict.items():
                t_left = np.amin([matrix[x][1] for x in range(len(matrix))])
                t_right = np.amin([(matrix[x][0].shape[1] - matrix[x][1]) for x in range(len(matrix))])
                cropped_matrix = []
                for x in range(len(matrix)):
                    phases = matrix[x][0]
                    center = matrix[x][1]
                    cropped_matrix.append(phases[:, center-t_left:center+t_right])

                total = cropped_matrix[0]
                for i in range(len(cropped_matrix)-1):
                    total += cropped_matrix[i+1]

                total /= len(cropped_matrix)
                dict[key] = (total, t_left) 
    
    if centers != None: return (dict, centers)
    else: return dict          
    