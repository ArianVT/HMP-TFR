import numpy as np
import copy
import math
import matplotlib.pyplot as plt

def get_context(event_probs, left, right):
    event = np.argmax(event_probs)
    context = np.zeros(3)
    
    context[0] = max(0, event - left)
    context[1] = event
    context[2] = event + right

    return context #context in samples   

def get_tfr(data, freqs, baseline, avg=False, jobs=2, decim=1, cycles=2):
    return data.compute_tfr(method="morlet", freqs=freqs, n_cycles=cycles, use_fft=True, average=avg, n_jobs=jobs, decim=decim).apply_baseline(baseline, mode='logratio') # Db?
    

def get_tfr_events(tfr, probs, left, right, sfreq, decim=1):
    print("retreiving TFR events")
    tfr_events = []
    tfr_shape = tfr.shape
    interval = left+right
    decim_interval = int(math.floor(interval/decim))
    tfr_event = copy.copy(tfr).crop(tmin = 0, tmax = (interval/sfreq), include_tmax=False) #create shallow copy of tfr_event
    
    #create centred tfr_event per event onset
    for event in range(probs.shape[1]):
        print(f"event: {event}, trial: ----", end='\r')
        tfr_event = copy.deepcopy(tfr_event)
        tfr_event_data = np.zeros((tfr_shape[0], tfr_shape[1], tfr_shape[2], decim_interval))
        tfr_event_data[:] = np.nan
        for trial in range(tfr._data.shape[0]):
            print(f"event: {event}, trial: {trial + 1}", end='\r', flush=True)
            context = get_context(probs[trial, event, :], left, right) #use event propabilities to find center
            begin = context[0]
            end = context[2]
            if end - begin == interval:
                tfr_event_data[trial, :, :, :] = tfr._data[trial, :, :, int(begin/decim):int(end/decim)] #change data to centred TFR

        tfr_event._data = tfr_event_data #change event data to cropped TFR
        averaged_tfr_event = tfr_event.average() 
        tfr_events.append(averaged_tfr_event) #append tfr_event
    print("retreived TFR events succesfully")
    return tfr_events

def plot_tfr_event(tfr, channels, title, lim):
    _, ax = plt.subplots()
    tfr.plot(axes=ax, picks=channels, vlim=lim, title=title, show=False)
    t_event = ax.get_xticks()[-1]/2
    ax.axvline(x=t_event, color='grey', linestyle='--', label='Event')
    ax.set_xticklabels([round(x - t_event, 2) for x in ax.get_xticks()])
    plt.show()