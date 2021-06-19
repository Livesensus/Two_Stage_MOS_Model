import model_1_pipline
import audio_extraction
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt



model_2_pickle_fname = "finalized_model.sav"


class Audio_Model:
    '''
        Pipleline for the whole model 1 and 2 audio model. Supports singal and duo channel audio
    '''

    def __init__(self, fname, input_signal=False, sec_seg=3, x=None, sr=None, video=False, video_fortran=False): 
        '''
            Init function
            @params:
                input_signal: If true then make sure
                                                    1) x
                                                    2) sr 
                            are in the input
                              If False then make sure 
                                                    1) fname
                            are in the input
                              This allows the class to take in filanme and load the audio in the class or have it in the inptut

                fname: If input_signal = False, and video = True then it is the path/name to the video. If not then it is the name if the
                    audio
                sec_seg: seconds per segment. For example audio file is 6 seconds we would do 6/3 = 2 seconds of inference. In this case if
                    audio file is 10 seconds we just discard the last 1 seconds since 10/3 = 3 so only 3 segments
                x: If input_signal is true, then it is the input. It can be 1D array or 2d array float or int since they are handled
                    by the model 
                sr: sampling rate of the audio if input_signal is true.
                video: If true, then we will extract audio from the video
                video_fortran: apply np.asfortranarray( . ) to signal, error for some reason sometimes when extracting audio from video
        '''
        self.sec_seg = sec_seg

        if input_signal == False:
            if video == False:
                self.signal, self.sr = librosa.load(fname, dtype = np.float32)
            else:
                self.sr, self.signal = audio_extraction.extract_audio(fname)
                if video_fortran:
                    self.signal = np.asfortranarray(self.signal)
                # make sure audio in (2, num samples) if it is 2d
            
        self.duo_channel = self.helper_stereo_check(self.signal)

        if self.duo_channel:
            self.signal = self.signal.T
            '''
            if self.signal.shape[0]!= 2 and len(self.signal.shape[0]) == 2:
                self.signal = self.signal.T
            '''


        if self.duo_channel:
            num_seg = int( len(self.signal[0]) /(sec_seg*self.sr) )
        else:
            num_seg = int( len(self.signal) /(sec_seg*self.sr) )


        # get model 2 file
        self.loaded_model = pickle.load(open(model_2_pickle_fname, 'rb'))


        '''
        IMPORTANT: for now we only keep one channel if duo channel
        '''
        if self.duo_channel:
            self.signal = self.signal[0]


    def inference(self, st, ed, verbose=True):
        '''
            Inference model 1 and model 2
            @params:
                st: st of the audio in sec
                ed: same as above in sec
                verbose: whether print the meta information 
        '''
        model = model_1_pipline.InputAudio(self.signal[st*self.sr:ed*self.sr], self.sr)
        model_1_output = model.get_module_results()

        model_1_output_all = []
        echo_0,echo_1 = self.norm_echo(model_1_output['Echo'][0],model_1_output['Echo'][1])

        model_1_output_all.append(echo_0)
        model_1_output_all.append(echo_1)
        model_1_output_all.append(model_1_output['zero_frames_norm'][0])
        model_1_output_all.append(model_1_output['zero_frames_norm'][1])
        model_1_output_all.append(model_1_output['Bandwidth'])
        model_1_output_all.append(model_1_output['Latency'])

        

        mos_predicted = self.loaded_model.predict([model_1_output_all])

        if verbose:
            print('input: [echo_0, echo_1, zero_norm_0, zero_norm_1, bandwidth, latency]',model_1_output_all)
            print('Predicted MOS', mos_predicted)
            print('\n')


        return mos_predicted[0]


    def inference_all(self, verbose=True):
        self.mos_time = []
        end = int(len(self.signal)/(self.sr*self.sec_seg))*self.sec_seg
        for i in range(0,end,self.sec_seg):
            if verbose:
                print('time st:', i, 'ed', i+self.sec_seg)
            mos_predicted = self.inference(i,i+self.sec_seg, verbose)
           
            self.mos_time.append(mos_predicted)

        if verbose:
            plt.figure(figsize=(10,5))
            plt.title('MOS over Time')
            plt.plot(np.linspace(0,len(self.signal)/self.sr,len(self.mos_time)),self.mos_time)

        return 


    '''
    ################################################################################################################################
                                                                Normalisation functions 
    ################################################################################################################################
    '''
    def norm_echo(self,x_0,x_1):
        cur = min(x_0, 150)
        x_0 = 1-cur/150
        cur = x_1
        x_1 = 1 - cur
        return x_0, x_1



    '''
    ################################################################################################################################
                                                                HELPER FUNCTIONS
    ################################################################################################################################
    '''
    def helper_stereo_check(self,audio):
        '''
            Description:
                Helper to check how many channels
            Input: 
                audio: a buffer, two channels or mono channel. Note that audio read in with 2 channels
                    have a chanle of (len(tot num samples in 1 channel), num_channels)
                    Also note that this has to be a numpy array, given .T requries a numpy array
                    We will assume input is a numpy array
                            
            Output: 
                True if 2 channels, False if single

            Side effect:
                take the transpose, does not affect 
        '''
        
        audio = audio.T
        if len(audio) == 2:
            return True
        else:
            return False


