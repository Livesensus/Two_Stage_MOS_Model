import librosa
import scipy
import math
import scipy
import numpy as np
import pickle



'''
description:
    return the start and end times of zero-frame intervals in the given audio in a list

Input:
    audio: audio array. (scipy uses integers, librosa uses floatpoint)
    sf: sampling frequncy of the aduio
    delay: the minimum length of the zero frames that will be detected (defalt=0, delay = sf/10 when default)
    threshold: <= threshold will be considered as zero frames

output:
    zero_frames: list of start and end times of the zero frame intervals in seconds.
'''
def zero_frame_detection(audio, sf, delay=0, threshold=0):

    zero_frames = []

    l = len(audio)
    if delay == 0:
        delay = int(sf/1000)          # 1/10 of the sf is 100ms

    j = 0           # keeps track of number of zero frames in current interval
    start = 0       # keeps track of start frame index of current zero frames interval

    # gap is 60ms
    gap = int(sf*0.06)
    ct_bin = []
    curr_bin = 0
    # max vel is when we have ie [0,gap, 0,gap] for ex. gap cus each gap is num. samples and all 0s. if odd l/gap we rd down. draw it out and see ie [0,gap,0,gap,0] there are 2 gaps. then preceed with calculation
    max_vel = (int(int(l/gap)/2)*gap)/int(l/gap)


    

    for i in range(l):
        if j == 0:
            start = i
            
        if abs(audio[i]) <= threshold:
            j += 1
        elif abs(audio[i]) > threshold and j <= delay:
            j = 0
        else:
            # start = int(start/sf)               # remove int() if want more precise start/end time
            # end = int(i/sf)
            start = start
            end = i
            zero_frames.append([start, end])
            j = 0


        if i%gap == 0 and i!=0:
            

            ct_bin.append(curr_bin)
            curr_bin = 0
        elif abs(audio[i] <= threshold):
            curr_bin += 1

    

    if j > delay:
        # start = int(start/sf)                   # remove int() if want more precise start/end time
        # end = int(i/sf)
        start = start
        end = i
        zero_frames.append([start, end])

    return zero_frames, ct_bin, max_vel



'''
description:
    Intensity module for packetloss

Input:
    zero_frames: a list of [[st_0,ed_0],[st_1,ed_1],...] that descbie the positon of such "zero" frames, same as how it was defined
                in the packetloss module
    a_len: length of the current audio, for example if we are passing 3 seconds segments, then this would be the length
                of that audio segments,
    sr: sampling rate of the audio segments
    num_seg: the granuarity, for example num_seg = 3 will look at the audio as 3 smaller segments, and measure
                the intensity and "velocity" of those 3 segments

output:
    int_percent: a value from [0,1] that is a weighted sum of the percentage of the audio that is zero frames
    int_vel: a value from [0,1] that is a weighted sum of the percentage of how many of these zero frames occur
'''
def packetloss_intensity(zero_frames , a_len, ct_bin,max_vel,sr, num_seg=3):

    inten = []
    vel = []
    for i in range(num_seg):
        st = int(a_len/num_seg)*i
        ed = int(a_len/num_seg)*(i+1)-1

        cumsum = 0
        # now we do velocity
        ct_i = 0

        for pair in zero_frames:

            if pair[0] > ed:
                break
            # if current frame lost overlap from last frame to current
            if pair[0] < st and pair[1] > st and pair[1]<= ed:
                cumsum += pair[1] - st
                ct_i += 1
            # if the current lost packet started before current segment, and end in the next one
            elif pair[0] < st and pair[1] > ed:
                cumsum += ed-st
                ct_i += 1
            # if current lost packet extend to the next one
            elif pair[0] <= ed and pair[1] > ed:
                cumsum += ed - pair[0]
                ct_i += 1
            elif pair[1] <= ed and pair[0]>= st:
                cumsum += pair[1]-pair[0]
                ct_i += 1
        inten.append(cumsum/(ed-st))
        vel.append(ct_i)

    vel = np.array(vel)
    """
    Apply Dynamic range compression equation to make values more spread out

    (log(1+x)*x_max)/log(1+x)

    x_max is 100 and we change to to 100 since it gives better representation

    """
    # max number of 0 segments expected
    vel_threhold = 10
   
    beta = 0.000001

    int_percent_max = 1000
    w_i = [0.2,0.35,0.45]
    int_percent= sum(np.multiply(inten, w_i))/num_seg
    int_percent *= 1000
    int_percent = (np.log(1+int_percent+beta)*int_percent_max)/np.log(1+int_percent_max+beta)
    int_percent /= 1000

    """
    For vel, lets use a more scintific approach. Source: https://depts.washington.edu/sphsc461/temp_res/temp_res.pdf
    Around 50-75ms for 8000Hz Gap detection. Hence use 60ms as a unit.

    Velocity is number of zeros per 

    we also take the absolute of velocity because for now we assume going faster or slower has the same effect
    """
    # get the weighted average, then normalise
 
    #int_vel = (sum(np.multiply(vel/sum(vel), w_vel))/num_seg)/vel_threhold
    #int_vel = vel[2]-vel[1] + vel[1]-vel[0]
    
    vel = sum(ct_bin[1:]-ct_bin[0:len(ct_bin)-1])/(len(ct_bin)-1)
    
    int_vel = vel/max_vel
    int_vel = abs(int_vel)
    
    int_vel_max = 1000
    int_vel *= 1000
    int_vel = (np.log(1+int_vel+beta)*int_vel_max)/np.log(1+int_vel_max+beta)
    int_vel /= 1000


    # now 0 is the best at least for int_perent, but change to 1 is the best
    return 1-int_percent, 1-int_vel


'''
Description: 
    performs spectrogram autocorrelation, searches for peaks
    that would correlate to echo. This can detect echo that has at least
    a 0.1 volume multiplier (or higher) and has at least a delay of 30 ms

Inputs:
        f = file stream in bytes
        sr = sampling rate

Outputs:
    delay = delay in seconds of biggest peak
    intensity = volume multiplier of added signal

'''
def detectEcho(wav):
    f,t,a = scipy.signal.stft(wav)
    ff = np.fft.fft( ( a), axis=1)
    af = abs( np.fft.ifft( ff * np.conj( ff))).sum(0)

    peak, props = scipy.signal.find_peaks(af[5:int(len(af)/2)], height=af[0]*0.2) #clip is decimal between 0.1 & 0.3

    maxH = 0
    maxPeak = 0

    if (len(peak) > 0):
        prominences = scipy.signal.peak_prominences(af, peak, wlen=None)
        maxProm = np.argmax(prominences[0])
        maxPeak = peak[maxProm] + 0
        maxH = props['peak_heights'][maxProm]

    if (maxPeak < 4):
        maxPeak = 0
        maxH = 0

    delay = maxPeak * 1.33
    # print("maxH", maxH)
    intensity = maxH / af[0]
    return delay, intensity


'''
Description: 
    This module predicts the frequency bandwidth that is being transmitted. 
    While Opus and other codecs sometimes restruct to 4000, 6000 and others, this predictor will not return one of these numbers

Inputs:
    x: make sure this is audio in float pointing 32, this is taken care of in the 
        model pipeline though

Outputs:

'''
def bandwidth_est(x,sr):

    D = np.abs(librosa.stft(x))
    
    # this is a new energy. Basically sum across all the rows (frequency bins) of the spectrogram, as find the first index where the signla
    # is below threshold which is then converted to corrspounding freq. This tells us the percentage of nyquist (highest) frequency that we are 
    # using
    energy= np.sum(D, axis = 1)
   
    threshold = 20
    # edge condition 
    if len(np.where(energy<threshold)[0])!=0:
        # 2048 is the default for N pts DFT in librosa, we can change this
        freq = (sr*np.where(energy<threshold)[0][0])/2048
    else:
        # in this case use everything hence 0 best quality
        return 0
    # percentage of nyquist freq
    return 1-np.minimum(freq/(sr/2), 1)


'''
Description: 
    This module predicts the amount of none interrupted speech in the audio in percentatge. It used a pre-trained
    SVM model (clf_SVM.p). The audio will be preprocessed into desired dimension before passing to the model.

Inputs:
    audio: audio waveform in floating point
    sr: sampling rate of the audio

Outputs:
    The amount of none interrupted speech in percentage. 
    (1.0 for zero interrupted speech in the audio, and 0.0 for all interrupted speech in the audio)

'''
def latency_detector(audio, sr, clf_SVM=None):

    # make sure it is the correct format, expect float
    if max(audio) > 1:
        audio = (audio/32767).astype(float)

    # make sure the input audio has the correct shape
    # linear interpo
    audio = np.interp(np.arange(0,len(audio), float(sr/22000)), np.arange(0, len(audio)), audio)
    fs = 22000

    # pre-process the audio into 5 second segemnts
    X = []
    seg_num = math.ceil(len(audio)/fs/5)
    for i in range(seg_num-1):
        X.append(audio[i*fs*5:(i+1)*fs*5])

    # handle the last segment
    last_seg = [0]*5*fs
    last_seg[:len(audio[(seg_num-1)*fs*5:])] = audio[(seg_num-1)*fs*5:]
    X.append(last_seg)
    
    # process each 5s segments into 10 0.5s segments
    X_ready = []
    for j in range(len(X)):
        for i in range(10):
            s = int((i/2)*fs)
            e = int((i/2+0.5)*fs)
            X_ready.append(X[j][s:e])


    
    # SVM prediction
    if clf_SVM == None:
        clf_SVM = pickle.load( open( "clf_SVM.p", "rb" ) )
    pred = clf_SVM.predict(X_ready)
    actual_seg = round(len(audio)/sr)*2
    result = pred[:actual_seg]

    return 1 - sum(result)/len(result)

