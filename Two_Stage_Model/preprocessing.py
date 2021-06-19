import numpy as np
import math
from scipy import signal



time_profiles = {
                'Even': [(2,5),(10,13), (18,21), (26,29), (34,37), (40,43),(48,51),(56, 61),(66,69),(74,77),(82,85),(90,93),(98,99) ],
                'Even_Occasional':[(2,5),(20,23),(38,41),(56,59),(62,65),(80,83),(98,99)] , 
                'Even_Seldom': [(2,5),(35,38),(68,71)],
                'Start_Only':[(2,3),(8,10),(13,18),(21,23),(28,29) ] ,
                'Start' :[(2,3),(8,10),(13,18),(21,23),(28,29), (38,39), (50,52) ] ,
                'Middle_Only': [(32,33),(38,40),(43,48),(61,63),(68,69) ],
                'Middle' : [(15,17),(24,25),(32,33),(38,40),(43,48),(61,63),(68,69), (76,77),(84,86) ],
                'End_Only': [(67,68),(73,75),(78,83),(86,88),(93,94) ],
                'End': [(45,47),(58,59),(67,68),(73,75),(78,83),(86,88),(93,94) ],
                'Start_Long':[(5,20)],
                'Middle_Long':[(42,57)],
                'End_Long': [(70,85)],
                'Entire' : [(0,100)]
}



def helper_stereo_check(audio):
# '''
#     Description:
#         Helper to check how many channels
#     Input: 
#         audio: a buffer, two channels or mono channel. Note that audio read in with 2 channels
#             have a chanle of (len(tot num samples in 1 channel), num_channels)
#             Also note that this has to be a numpy array, given .T requries a numpy array
#             We will assume input is a numpy array
                    
#     Output: 
#         True if 2 channels, False if single

#     Side effect:

# '''
    # take the transpose, does not affect 
    audio = audio.T
    if len(audio) == 2:
        return True
    else:
        return False



def percentofaudio_to_absolute_sample(profile,sr, sample_length ):
# '''
#     Description:
#         helper function to covert a time profile to absolute sample for each internval
#     Input:
#         profile: one of the profile defined in the document or above of this file
#         sr: sampling rate
#         sample_length: length of audio in number of samples, ie 44100 samples for 1 second of 44100 Hz audio
#     Output:
#         time: a list of start and end in absolute samples
#         time_seconds: time in seconds
#     Side Effect:
#         None

# '''
    time = []
    time_seconds = []
    for per in profile:
        st = int((per[0]/100)*sample_length)
        ed = int((per[1]/100)*sample_length)
        time.append((st,ed))
        time_seconds.append((st/sr, ed/sr))
    return time,time_seconds







def wav2mstft(x, N, S, L):
# '''
#     Description:
#         Chop a waveform into frames, and compute magnitude FFT of each frame.
#     Input: 
#         x = the signal (a np array)
#         N = length of the FFT
#         S = frameskip
#         L = framelength         
#     Output: 
#         the positive-frequency components of the magnitude FFT of each frame (nparray),
#         which is an array of size T by (N+1)/2.

#     Side effect:

# '''
    # T = number of frames
    T = 1+int(math.ceil((len(x)-L)/S))
    X = np.zeros((T, 1+int(N/2)))
    for t in range(T):
        window = signal.hamming(len(x[(t*S):min(t*S+L,len(x))]),sym=False)
        X[t,:] = np.absolute(np.fft.rfft(window*x[(t*S):min(t*S+L,len(x))], n=N))
    return(X)




def mstft2filterbank(X, Fs, Nm):
# '''
#     Description:
#         Convert MSTFT into Filterbank coefficients in each frame
#     Input: 
#         X = the MSTFT of each frame (an nparray)
#         Fs = the sampling frequency
#         Nm = the number of mel-frequency bandpass filters to use    
#     Output: 
#         the Filterbank coefficients in each frame (nparray)
#     Side effect:

# '''
    # First, create the triangle filters
    max_bin = X.shape[1]
    max_hertz = Fs/2
    max_mel = 1127*np.log(1+max_hertz/700.0)
    triangle_filters = np.zeros((max_bin, Nm))
    for m in range(Nm):
        corners=[int((max_bin*700.0/max_hertz)*(np.exp(max_mel*(m+n)/(1127*(Nm+1)))-1)) for n in range(3)]
        triangle_filters[corners[0]:corners[1],m]=np.linspace(0,1,corners[1]-corners[0],endpoint=False)
        triangle_filters[corners[1]:corners[2],m]=np.linspace(1,0,corners[2]-corners[1],endpoint=False)

    magnitudefilterbank = np.matmul(X,triangle_filters)
    maxcof = np.amax(magnitudefilterbank)
   
   # check for the edge case if everything is 0
    if (np.maximum(1e-6*maxcof,magnitudefilterbank)).all() == 0:
        return np.maximum(1e-6*maxcof,magnitudefilterbank)
    return(np.log(np.maximum(1e-6*maxcof,magnitudefilterbank)))



def znorm(X):
    return((X-np.average(X,axis=0))/np.std(X,axis=0))



def log_mel_spectrogram(audio,fs,L,S,dft_length,n_bands):
# '''
#     Description:
#         Create log mel spectrogram
#     Input: 
#         audio 
#         fs: sampling rate
#         L: framelegth in time (seconds)
#         S: frame overlap in time (seconds)
#         n_bands: how many triangle filters for mel 
        
#     Output: 
#         the Filterbank coefficients in each frame (nparray)
#     Side effect:

# '''
    
    # calling wav2mstft by chopping audio into 80ms chunks
    # L is framelength, 0.08 is 80 ms

    X = wav2mstft(audio,dft_length,S,L)
    log_mel = mstft2filterbank(X,fs,n_bands)
    #log_mel = znorm(log_mel)
    return X,log_mel




def normal_spectrogram(X,dbrange):
# '''
#     Description:
#         Create the spectrogram by calculating in the decibel range

#         This different from how we ususally do spectrograms, since in this case when we call log_mel_spectrogram it returns X which the absolute has been taken care of 

#         We call this using the return value X from log_mel for our model to avoid calculating STFT again which is time consuming
#     Input: 
#         X: stfted audio
#         dbrange: db range of this spectrogram
#     Output: 
#         X_normal_spectrogram: absolute, adjucted to the db scale
#     Side effect:
# 
# '''
    dbrange = dbrange
    X = np.abs(X)
    maxval = np.max(X)
    
    # check for the edge cases of X and maxval all 0
    if maxval == 0:
        maxval = 1
    if X.all() == 0:
        X_normal_spectrogram = np.zeros((X.shape[0],X.shape[1]))
    else:
        X_normal_spectrogram = 20*np.log10(np.abs(X)/maxval)
        X_normal_spectrogram = np.maximum(-1*dbrange,X_normal_spectrogram)

    return X_normal_spectrogram





def filterbank2mfcc(X, D):
# '''
#     Description:
#         Get the MFCC 
#     Input: 
#         X: stfted audio
#         D: number of features in the output
#     Output: 
#         X_normal_spectrogram: absolute, adjucted to the db scale
#     Side effect:

# '''
    T = X.shape[0]
    Nm = X.shape[1]
    
    mfcc = np.zeros((T, D))  
    
    # Create the DCT transform matrix
    #Dover3 = int(np.ceil(D/3))
    #D = int(3*Dover3) # Make sure it's a multiple of 3
    dct_transform = np.zeros((Nm, D))
    for m in range(Nm):
        dct_transform[m,:] = np.cos(np.pi*(m+0.5)*np.arange(D)/Nm)
            
    # Create the MFCC
    mfcc = np.zeros((T, D))  
    mfcc = np.matmul(X,dct_transform)   # static MFCC
    #mfcc[1:T,Dover3:2*Dover3] = mfcc[1:T,0:Dover3] - mfcc[0:(T-1),0:Dover3]          # Delta-MFCC
    #mfcc[1:T,2*Dover3:D] = mfcc[1:T,Dover3:2*Dover3] - mfcc[0:(T-1),Dover3:2*Dover3]  # Delta-delta-MFCC
    return(mfcc)
    


def helper_check_blip_exist(seg_st_ed,time_segments):
# '''
#     Description:
#         Helper function check if there is a blip in this segment of the audio
#     Input: 
#         seg_st_ed: st is index into original audio of this segment, same for end a tuple (st,ed)
#         fs: sampling rate 
#         time_segments: a list [(a1,a2)] of tuples, (st,ed) where they are samples
#         audio_length_samples: the length of the audio in number of samples
#     Output: 
#         has_blip: True/false
#         has_blip_when: [] if no blip, [(st,ed)] where st and ed are adjusted for current segment

#     Side effect:
#         None
# '''
    seg_st = seg_st_ed[0]
    seg_ed = seg_st_ed[1]
    
    has_blip = False
    has_blip_when = []
    
    for blip_seg in time_segments:
        b_st = blip_seg[0]
        b_ed = blip_seg[1]
        
        if b_st >= seg_st and b_st <= seg_ed:
            has_blip = True
            # if the blip goes out of this segment, say that it goes to the end of the segment
            if b_ed > seg_ed:
                b_ed = seg_ed
            has_blip_when.append((b_st-seg_st,b_ed-seg_st))
        # started before this segment
        elif b_st < seg_st and b_ed >= seg_st:
            has_blip = True
            if b_ed > seg_ed:
                b_ed = seg_ed
            has_blip_when.append((0,b_ed-seg_st))
            
    return has_blip, has_blip_when



def helper_fpt_convert(audio_single_channel):
# '''
#     Description:
#         Helper function to convert all audio streams to [1.0,-1.0] float32 if not already in this format
#     Input: 
#         audio: a buffer of (len(1 channel of audio),)
                    
#     Output: 
#         audio_single_channel: converted audio

#     Side effect:

# '''
    if type(audio_single_channel[0]) == np.int16:
        # 32768 = 2^(16-1) since we are doing signed
        audio_single_channel = audio_single_channel.astype(np.float32, order='C') / 32768.0

        return audio_single_channel
    else:
        
        return audio_single_channel



def helper_convert_to_segments_dataset(audio, fs,t,b_t_samples):
# '''
#     Description:
#         Helper function to convert an audio sample in 3 (t) seconds segments
#         ****This function has been tested and checked, should be working fine for larger than one segment****
#     Input: 
#         audio: a buffer of (len(1 channel of audio),)
#         fs: sampling rate of audio
#         t: time segment in seconds of the audio we want per spectrogram
#         b_t_samples: a list of when blips happen, if entire it is [-1] in absolute samples
#     Output: 
#         audio: a np array of shape (number of 3 seconds segment, len of the 3 second segment)

#     Side effect:

# '''
    # assumes no blip happens after the audio (there could be edge cases)
    num_sample_per = int(t*fs)
    
    # take the floor, if theres anything left, we will deal with later
    num_segments = len(audio)/num_sample_per
    
    wrap_around = False
    # meaning there are more left
    if num_segments > int(num_segments):
        wrap_around = True
        num_segments = int(num_segments)
        num_segments += 1
    num_segments = int(num_segments)
    
    
    
    wrap_around = False
    # if audio is shorter than 3 seocnds, wrap around
    if num_segments ==0:
        new_audio = np.zeros(fs*3)
        new_audio[0:len(audio)] = audio
        st = 0
        for i in range(len(audio),fs*3):
            new_audio[i] = audio[st]
            st += 1
    
        return num_segments
    
    """
    now we do the work
    """
    seg_hv_blip = []
    new_audio = np.zeros((num_segments,num_sample_per))
    for i in range(num_segments):
        # if the last segment
        if (i+1) == num_segments: 
            # check for when blips happen
            rec_b_t = []
            has_blip, has_blip_t_samples =  helper_check_blip_exist((i*num_sample_per,len(audio)-1),b_t_samples)
            rec_b_t = has_blip_t_samples
                                                                    
            # put the audios in 
            new_audio[i][0:(len(audio)-i*num_sample_per)] = audio[i*num_sample_per:len(audio)]    
            num_padded = len(audio[i*num_sample_per:len(audio)])
            num_sam_left = (i+1)*num_sample_per - len(audio)
            
            # edge case
            if(num_sam_left ==1):
                new_audio[i][num_sample_per-1] = audio[i*num_sample_per]
            st = 0
            # recurrent blips
            for j in range(0,num_sam_left):
                if st <num_padded:
                    new_audio[i][num_padded+j] = new_audio[i][st]
                else:
                    st = 0
                    new_audio[i][num_padded+j] = new_audio[i][st]
                st += 1        
                                                                    
                # add the recurrent blip market back , for all the blips in this segment, check if a blip exist here
                for z in has_blip_t_samples:
                
                    if j == z[0]:
                        end_rec = num_padded+z[1]
                        if end_rec > len(audio):
                            end_rec = len(audio)-1
                        rec_b_t.append((num_padded+j,end_rec))
                        
            # combine the lists again
            # combine the two
            seg_hv_blip = seg_hv_blip+[rec_b_t]
        else:
            has_blip, has_blip_t_samples =  helper_check_blip_exist((i*num_sample_per,(i+1)*num_sample_per-1),b_t_samples)
            if has_blip:
                seg_hv_blip.append(has_blip_t_samples)
            else: 
                seg_hv_blip.append([])
            new_audio[i] = audio[i*num_sample_per:(i+1)*num_sample_per]

    return new_audio,seg_hv_blip


'''
Description:
    define a class for preprocessing
'''

class Preprocessing:

    def __init__(self, audio_in,fs_audio,dbrange, L_time,S_time, dft_length,n_bands):
        '''
        All internal variables and macros

        audio_in: assumed to be a 3 (or other legth segment) of audio
        '''
        self.audio = np.zeros((audio_in.T).shape,dtype=np.float32)
        self.fs = fs_audio
        self.duo_channel = False
        self.dbrange = dbrange
        self.L = int(self.fs*L_time)
        self.S = int(self.fs*S_time)
        self.dft_length = dft_length
        self.n_bands = n_bands
        # X_spec is the normal spectrogram
        self.X_spec = None
        # log mel is the mel spectrogram
        self.log_mel = None

        '''
        Initialisation code
        '''
        self.duo_channel = helper_stereo_check(audio_in)
        # put audio into right format, shape after .T if duo is (2, len(audio per channel))
        audio_in = audio_in.T
        
        # convert to ft32, call per channel if duo
        if self.duo_channel:
            self.audio[0] = helper_fpt_convert(audio_in[0])
            #self.audio[0].astype(np.float32)
            
            self.audio[1] = helper_fpt_convert(audio_in[1])
        else:
            self.audio = helper_fpt_convert(audio_in)

        # put audio into a 2d array of 3 seconds 
        '''
        Right now we only care about one channel, so I am going to ignore the other one
        '''

        if self.duo_channel:
            # get the spectrograms
            # ignore the other channel
            X, self.log_mel = log_mel_spectrogram(self.audio[0], self.fs,self.L,self.S,dft_length,self.n_bands)
            self.X_spec = normal_spectrogram(X,self.dbrange)
        else:
            # get the spectrograms
            X, self.log_mel = log_mel_spectrogram(self.audio, self.fs,self.L,self.S,self.dft_length,self.n_bands)
            self.X_spec = normal_spectrogram(X,self.dbrange)
        
        
        

    


    
    








