from model_1_modules import zero_frame_detection, packetloss_intensity,bandwidth_est, detectEcho, latency_detector
from preprocessing import Preprocessing
import numpy as np
import pickle


'''
These are the macros used for spectrograms
'''
# dbrange is the range of audio decibal for the normal spectorgrams
dbrange_spectrogram = 120

# L_time is the time of the frame for STFT
L_time = 0.04
# S_time is the time for overlapping frame for STFT
S_time = 0.02

# here is an exmaple of how to use them
# L = int(fs_audio*0.04)
# S = int(fs_audio*0.02)

# dft length for a given frame in STFT
dft_length = 2048
# the number of frequency bands for mel spectrogram
n_bands = 128

# buffer the input audio into 3 seconds segments

'''
Description:
    define a class for input audio
'''
class InputAudio:

    def __init__(self, signal, sr, path=None):

        self.sr = sr
        self.path = path
        self.original_signal = signal
        # load audio as single channel int16 and float32
        self.preprocessor = Preprocessing(signal,sr,dbrange_spectrogram, L_time, S_time, dft_length,n_bands)
        self.latency_model = pickle.load( open( "clf_SVM.p", "rb" ) )  # load latency pre-trained model

        if self.preprocessor.duo_channel:
            self.wave = self.preprocessor.audio[0]
            self.wave_float = self.preprocessor.audio[0]
        else:
            self.wave = self.preprocessor.audio
            self.wave_float = self.preprocessor.audio

        # DSP metrics and detectors, _norm are for metrics that would be passed to model 2
        self.zero_frames = []
        self.zero_frames_norm = []
        self.echo = []
        self.latency = None


    # detector helper functions
    def get_zero_frames(self, signal=None, sr=None, delay=0, threshold=0):

        if signal == None:
            signal=self.wave_float
        if sr == None:
            sr=self.sr



        zero_frames,ct_bin,max_vel= zero_frame_detection(signal, sr, delay=0, threshold=0)
        zero_length = []
        for start, end in zero_frames:
            zero_length.append((end-start)/self.sr)     # calculate start and end in seconds

        #self.zero_frames = zero_length
        self.zero_frames = zero_frames

        ploss_inten = packetloss_intensity(zero_frames , len(signal), np.array(ct_bin), max_vel, self.sr, num_seg=3)
        self.zero_frames_norm = ploss_inten

        return self.zero_frames


    def get_bandwidth(self):
        return bandwidth_est(self.wave_float, self.sr)

    def get_echo(self, signal=None):

        if signal == None:
            signal=self.wave

        self.echo = detectEcho(signal)
        return self.echo
    
    def get_latency(self, signal=None, sr=None):
        if signal == None:
            signal=self.wave
        if sr == None:
            sr=self.sr

        self.latency = latency_detector(signal, sr, self.latency_model)
        return self.latency


    def get_module_results(self):
        results = {
            "zero_frames" : self.get_zero_frames(),
            "zero_frames_norm" : self.zero_frames_norm,
            "Echo": self.get_echo(),
            "Bandwidth": self.get_bandwidth(),
            "Latency": self.get_latency()        
            }
        return results

