
"""
Main function to call when blipping audios

"""

import scipy.io.wavfile
from audio_blips import audio_blips
import audio_extraction
import moviepy.editor
import numpy as np
import ffmpeg
import ast
from tqdm import tqdm
import os
import pandas as pd
import librosa


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
                'Start_1.2': [(15,25)],
                'Middle_1.2': [(45,55)],
                'End_1.2': [(62,82)],
                'Start_Long_1.2': [(15,45)],
                'Middle_Long_1.2': [(35,65)],
                'End_Long_1.2': [(51,81)],
                'BMC_1.2': [(0,40),(40,60),(60,100)],
                'BH_1.2': [(0,50),(50,100)],
                'Survey_Training_Long_1.2': [(25,70)],
                'Survey_Training_1.2': [(25,45)],
                'Middle_1.2_Short': [(45,54)],
                'Start_Short_1.2': [(15,20)],
                'End_Short_1.2': [(76,81)]

}



def video_with_audio():

    # blip ID
    PACKETDROP  = 0  # audio_packet_drop(ori_audio, start=0, end=0, times = 5)
    PACKETLOSS  = 1  # audio_packetLoss(audio, start=0, end=0, sampling_rate = 44100)
    LAGGING     = 2  # audio_lag(raw_audio, start=0, end=0, sampling_rate=44100)
    WHITE_NOISE = 3  # white_noise(audio, start=0, end=0,level=20). ATTENSION! this output file is super loud. 
    COMPRESSION = 4  # compression(audio, start=0, end=0, level=20)
    REVERBERATION = 5 #reverberations(audio, start=0, end=0, meta_data=(10,44100)):

    # path to folder
    PATH = '/Users/michaellau/Documents/UIUC/Alchemy/Videos/GTA_Blipped'
    LD_AUDIO = True


    # add for each file in the folder

    #filename = 'original_audio.wav' # WAV file only
    filename = '/Users/michaellau/Documents/UIUC/Alchemy/Videos/GTA_Blipped/GTA_1_P2.mp4'


    video = None
    if LD_AUDIO == True:
        # checked, it is getting a buffer correctly and I checked it it gives the right length and it does
        raw_rate, raw_audio = audio_extraction.extract_audio(filename)
        # raw_audio is (num samples, num channels), transpose get the (num channels, samples)
        raw_audio = raw_audio.T
        
    else:
        # load the file
        raw_rate, raw_audio = scipy.io.wavfile.read(filename)

 


    # blip the audio
    blip_id = PACKETLOSS
    meta_data = raw_rate
    # each entry is a tuple (start time in seconds, end time)
    time = [(1,1.25), (1.5,1.75), (1.8, 1.85), (1.88, 1.9), (1.95, 2.05), (6.06, 6.10), (6.12, 6.15), (6.18,6.24),(6.5, 6.7)]

    blipped_audio = np.copy(raw_audio)

    for t in time:
        blipped_audio[0] = (audio_blips(blip_id, blipped_audio[0], int(raw_rate*t[0]), int(raw_rate*t[1]), meta_data))
        blipped_audio[1] = (audio_blips(blip_id, blipped_audio[1], int(raw_rate*t[0]), int(raw_rate*t[1]), meta_data))

  


    # write to the file
    # expect (data, num channel) for the last argument
    scipy.io.wavfile.write("blipped_audio.wav",raw_rate, np.asarray(blipped_audio).T)




    # combine the audio with video, use VideoClip.set_audio(audio)

    """
    Doesnt work

    video_with_new_audio = video.set_audio(AudioFileClip("blipped_audio.wav"))
    video_with_new_audio.write_videofile("Test.mp4",audio=True)
    """


    # this works beautifully finally
    # https://stackoverflow.com/questions/56973205/how-to-combine-the-video-and-audio-files-in-ffmpeg-python
    input_video = ffmpeg.input(filename)
    input_audio = ffmpeg.input("blipped_audio.wav")
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output('finished_video.mp4').run()
    return




'''
Description:
    helper funcion to merge audio with video after blip induction

Input:
    vid_filename: file name of video 
        example: filename = '/Users/michaellau/Documents/UIUC/Alchemy/Videos/GTA_Blipped/GTA_1_P2.mp4'
    audio_filename:file name of audio
        example: "blipped_audio.wav"
    output_name:
        name of the output video combined with the new audio
        example: 'finished_video.mp4'
Output:
    None

Side Effect:
    Writes the combined media into the current folder if no path, or the destination + file name

'''
def merge_audio_video(vid_filename, audio_filename, output_name):
    input_video = ffmpeg.input(vid_filename)
    input_audio = ffmpeg.input(audio_filename)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_name).run()
    return


'''
Description:
    calculates the intensity of the given blip and returns it
    For Compression and White noise, metadata is already the intensity, so we are just returning it 

Input:
    blip: an int that specifies which blip it is
    audio_duration: a float that specifies the length of the audio
    time_list: a [] that contains the time stamp
    meta_data: meta_data for this give audio clip, dependent on the blip type

Output:
    intensity: int ranging from 0 - 100

Side Effect:
    Writes the combined media into the current folder if no path, or the destination + file name

'''
def intensity_to_metadata_calculator(blip, audio_duration, time_list, meta_data):
    PACKETDROP  = 0  # audio_packet_drop(ori_audio, start=0, end=0, times = 5)
    PACKETLOSS  = 1  # audio_packetLoss(audio, start=0, end=0, sampling_rate = 44100)
    LAGGING     = 2  # audio_lag(raw_audio, start=0, end=0, sampling_rate=44100)
    WHITE_NOISE = 3  # white_noise(audio, start=0, end=0,level=20). ATTENSION! this output file is super loud. 
    COMPRESSION = 4  # compression(audio, start=0, end=0, level=20)
    REVERBERATION = 5 #reverberations(audio, start=0, end=0, meta_data=(10,44100)):

    # for these the meta_data is the intensity
    if blip == WHITE_NOISE or blip == COMPRESSION:
        return meta_data
    intensity = 0

    

    if blip == PACKETDROP:
        """
        intensity = total time dropped / audio duration  [seconds]
        """ 
        tot_time = 0   
        for b_t in time_list:
            for t in b_t:
                tot_time += (t[1]-t[0])
        intensity = tot_time/audio_duration
       
    if blip == PACKETLOSS:
    
        """
        intensity = total time dropped / audio duration  [seconds]
        tot_time = 0   
        for t in time_list:
            tot_time += (t[1]-t[0])
        intensity = tot_time/audio_duration
       
        """  
        intensity = 0
        


    # normalise to from 0 to 100
    return intensity*100


    

'''
Description:
    helper function to covert a time profile to absolute sample for each internval
Input:
    profile: one of the profile defined in the document or above of this file
    sr: sampling rate
    sample_length: length of audio in number of samples, ie 44100 samples for 1 second of 44100 Hz audio
Output:
    time: a list of start and end in absolute samples
    time_seconds: time in seconds
Side Effect:
    None

'''
def percentofaudio_to_absolute_sample(profile,sr, sample_length ):
    time = []
    time_seconds = []
    for per in profile:
        st = int((per[0]/100)*sample_length)
        ed = int((per[1]/100)*sample_length)
        time.append((st,ed))
        time_seconds.append((st/sr, ed/sr))
    return time,time_seconds



'''
Description:
    helper function override metadata
Input:
    meta_data: current meta data
    blip numer: what we are blipping now bro
    raw_rate: sampling rate
Output:
    meta_data: modified metadata

Side Effect:
    None

'''
def meta_data_check(meta_data, b, raw_rate, filename):
    # discon
    if b == 1:
        meta_data = raw_rate
    # packet loss
    elif b == 2:
        
        temp = meta_data
        meta_data = (meta_data[0], meta_data[1],meta_data[2],raw_rate)
    # echo
    elif b ==7:
        temp = meta_data
        meta_data = (raw_rate, meta_data[1],meta_data[2])
    # compression
    elif b ==5:
        temp = meta_data
        meta_data = (meta_data[0], meta_data[1],meta_data[2],raw_rate)
    # if lagging
    elif b==3:
        temp = meta_data
        meta_data = (meta_data[0],raw_rate)
    # bandwidth limited
    elif b == 8:
        temp = meta_data
        meta_data = (meta_data[0], meta_data[1],meta_data[2],raw_rate, filename)

    return meta_data


'''
Description:
    helper function to get timing information in samples
Input:
    profile: profile, a string
    sr: raw_rate
    audio: itself
Output:
    time: location in samples
    time_seconds: actual time

Side Effect:
    None

'''
def helper_timing_info(profile, raw_rate, raw_audio ,duo_channel):
    time = []
    time_seconds = []
    # each entry is a tuple (start time in seconds, end time)
    for i in profile:
        if i != 'Entire':
            if duo_channel == True:
                time_i, time_seconds_i = percentofaudio_to_absolute_sample(time_profiles[i],raw_rate,len(raw_audio[0]))
                time.append(time_i)
                time_seconds.append(time_seconds_i)
            else:
                time_i, time_seconds_i = percentofaudio_to_absolute_sample(time_profiles[i],raw_rate,len(raw_audio))
                time.append(time_i)
                time_seconds.append(time_seconds_i)
        else:
            # if entire audio, take the entire audio's length as end so we can blip the whole thing
            if duo_channel == True:
                time.append([(0,len(raw_audio))])
                time_seconds.append([])
            else:
                time.append([(0,len(raw_audio))])
                time_seconds.append([])
    return time, time_seconds



    

'''
Description:
    source is a wav file
    extract the audio from the given video source to blip and save it to the desired location

Input:
    path: Path to the video
    name: filename of the source video
    blip_id_list: a list [integers] of ID of the blips we want
    time_list: 
    output_name: file destimation + output file name that we want
    meta_data: meta data
Output:
    Intensity: integer that represents the intensity of the given blip, a list

Side Effect:
    write the blipped audio in .wav to the desired location with the name 

'''
def blip(path,name,blip_id_list, profile, output_name, meta_data,filetype, noise_path = None):
    # blip ID
    PACKETDROP  = 0  # audio_packet_drop(ori_audio, start=0, end=0, times = 5)
    PACKETLOSS  = 1  # audio_packetLoss(audio, start=0, end=0, sampling_rate = 44100)
    LAGGING     = 2  # audio_lag(raw_audio, start=0, end=0, sampling_rate=44100)
    WHITE_NOISE = 3  # white_noise(audio, start=0, end=0,level=20). ATTENSION! this output file is super loud. 
    COMPRESSION = 4  # compression(audio, start=0, end=0, level=20)
    REVERBERATION = 5 #reverberations(audio, start=0, end=0, meta_data=(10,44100)):
    ECHO = 6

    # path to folder
    PATH = path

    # add for each file in the folder
    filename = PATH + name +'.'+filetype

    '''
    LD the audio
    '''
    video = None
    duo_channel = False
    raw_rate = 0
    duo_channel_noise = False

    # load the file
    if filetype == 'mp4':
        raw_rate, raw_audio = audio_extraction.extract_audio(filename)
        # raw_audio is (num samples, num channels), transpose get the (num channels, samples)
        raw_audio = raw_audio.T
        if len(raw_audio) == 2:
            duo_channel = True       
            raw_audio = raw_audio[0]
            duo_channel = False
        if noise_path != None:
            #noise_rate, noise = scipy.io.wavfile.read(noise_path)
            noise ,noise_rate= librosa.load(noise_path, sr=raw_rate,dtype=np.float32, mono=True)
            noise = noise.T
            if len(noise) == 2 :
                duo_channel_noise = True   
     
    elif filetype == 'wav':
        #raw_rate, raw_audio = scipy.io.wavfile.read(filename)
        raw_audio,raw_rate = librosa.load(filename, sr=None,dtype=np.float32, mono = True)
        if len(raw_audio) == 2:
            duo_channel = True
            raw_audio = raw_audio[0]
        if noise_path != None:
            #noise_rate, noise = scipy.io.wavfile.read(noise_path)
            noise ,noise_rate= librosa.load(noise_path, sr=raw_rate,dtype=np.float32, mono=True)
            noise = noise.T
            if len(noise) == 2 :
                duo_channel_noise = True
               
        raw_audio = raw_audio.T
  
    # adjust the length, note we have now switched to using mono
    if noise_path is not None :
        if len(noise) >= len(raw_audio)   :
            noise = noise[0:len(raw_audio)]
        else:
            remindar = len(raw_audio) // len(noise)
            for i in range(remindar-1):
                noise = np.concatenate((noise, noise))
            diff = len(raw_audio) - len(noise)
            noise = np.concatenate((noise, noise[0:diff]))
   
    '''
    Find when the blips happens
    '''
    time, time_seconds  = helper_timing_info(profile,raw_rate,raw_audio,duo_channel)


    '''
    IMPORTANT, this is added for noise, we are stacking them together
    '''
    if noise_path != None:
        new_audio = []
        #only take 1 channl regardless
        if duo_channel:
            new_audio.append(raw_audio[0])
        else: 
            new_audio.append(raw_audio)

        if duo_channel_noise:
            new_audio.append(noise[0])
        else: 
            new_audio.append(noise)
        raw_audio = new_audio
        
    blipped_audio = np.copy(raw_audio)
    # list of intensity, follows the same index
    intensity = []

    for i,b in enumerate(blip_id_list):
        for t in time[i]:
            curr_meta = meta_data_check(meta_data[i],b,raw_rate, filename)
            if duo_channel == True:
                blipped_audio = (audio_blips(b-1, blipped_audio, int(t[0]), int(t[1]), curr_meta))
                #blipped_audio[0] = (audio_blips(b-1, blipped_audio[0], int(t[0]), int(t[1]), curr_meta))
                #blipped_audio[1] = (audio_blips(b-1, blipped_audio[1], int(t[0]), int(t[1]), curr_meta))
                #intensity.append(intensity_to_metadata_calculator(b-1, len(blipped_audio[0])/raw_rate ,time_seconds, curr_meta ))
            else:
             
                blipped_audio = (audio_blips(b-1, blipped_audio, int(t[0]), int(t[1]), curr_meta))
                #intensity.append(intensity_to_metadata_calculator(b-1, len(blipped_audio)/raw_rate ,time_seconds, curr_meta ))
                intensity.append(0)

    # check for if there is no blip
    if len(blip_id_list) == 0:
        intensity.append(0)

    # write to the file
    # expect (data, num channel) for the last argument
    # note that we are using a standard of float 32, since float64 does not seem to work in chrome
    
    if duo_channel:
        if type(blipped_audio[0][0])==np.int16:
            scipy.io.wavfile.write(output_name,raw_rate,np.int16(np.asarray(blipped_audio.T)))
        else:
            scipy.io.wavfile.write(output_name,raw_rate,np.float32( np.asarray(blipped_audio).T))
    else:
        if type(blipped_audio[0])==np.int16:
            scipy.io.wavfile.write(output_name,raw_rate,np.int16(np.asarray(blipped_audio)))
        else:
            scipy.io.wavfile.write(output_name,raw_rate,np.float32( np.asarray(blipped_audio)))
    return intensity





def blip_final(filename,blip_id_list, profile, output_name, meta_data,filetype, noise_path = None):
    '''
        This is the function used in the deliverable for calling blips


        extract the audio from the given video source to blip and save it to the desired location
    @params:
        path: path to the audio or video
        name: filename of the source video
        blip_id_list: a list [integers] of ID of the blips we want
        time_list: 
        output_name: file destimation + output file name that we want
        meta_data: meta data
        noise_path: path to the background noise
    @Returns:
        Intensity: integer that represents the intensity of the given blip, a list
    @Side Effect:
        write the blipped audio in .wav to the desired location with the name 

    '''
    
    
    
    # blip ID
    PACKETDROP  = 0  # audio_packet_drop(ori_audio, start=0, end=0, times = 5)
    PACKETLOSS  = 1  # audio_packetLoss(audio, start=0, end=0, sampling_rate = 44100)
    LAGGING     = 2  # audio_lag(raw_audio, start=0, end=0, sampling_rate=44100)
    WHITE_NOISE = 3  # white_noise(audio, start=0, end=0,level=20). ATTENSION! this output file is super loud. 
    COMPRESSION = 4  # compression(audio, start=0, end=0, level=20)
    REVERBERATION = 5 #reverberations(audio, start=0, end=0, meta_data=(10,44100)):
    ECHO = 6


    # add for each file in the folder
    filename = filename

    '''
    LD the audio
    '''
    video = None
    duo_channel = False
    raw_rate = 0
    duo_channel_noise = False

    # load the file
    if filetype == 'mp4':
        raw_rate, raw_audio = audio_extraction.extract_audio(filename)
        # raw_audio is (num samples, num channels), transpose get the (num channels, samples)
        raw_audio = raw_audio.T
        if len(raw_audio) == 2:
            duo_channel = True       
            raw_audio = raw_audio[0]
            duo_channel = False
        if noise_path != None:
            #noise_rate, noise = scipy.io.wavfile.read(noise_path)
            noise ,noise_rate= librosa.load(noise_path, sr=raw_rate,dtype=np.float32, mono=True)
            noise = noise.T
            if len(noise) == 2 :
                duo_channel_noise = True   
     
    elif filetype == 'wav':
        #raw_rate, raw_audio = scipy.io.wavfile.read(filename)
        raw_audio,raw_rate = librosa.load(filename, sr=None,dtype=np.float32, mono = True)
        if len(raw_audio) == 2:
            duo_channel = True
            raw_audio = raw_audio[0]
        if noise_path != None:
            #noise_rate, noise = scipy.io.wavfile.read(noise_path)
            noise ,noise_rate= librosa.load(noise_path, sr=raw_rate,dtype=np.float32, mono=True)
            noise = noise.T
            if len(noise) == 2 :
                duo_channel_noise = True
               
        raw_audio = raw_audio.T
  
    # adjust the length, note we have now switched to using mono
    if noise_path is not None :
        if len(noise) >= len(raw_audio)   :
            noise = noise[0:len(raw_audio)]
        else:
            remindar = len(raw_audio) // len(noise)
            for i in range(remindar-1):
                noise = np.concatenate((noise, noise))
            diff = len(raw_audio) - len(noise)
            noise = np.concatenate((noise, noise[0:diff]))
   
    '''
    Find when the blips happens
    '''
    time, time_seconds  = helper_timing_info(profile,raw_rate,raw_audio,duo_channel)


    '''
    IMPORTANT, this is added for noise, we are stacking them together
    '''
    if noise_path != None:
        new_audio = []
        #only take 1 channl regardless
        if duo_channel:
            new_audio.append(raw_audio[0])
        else: 
            new_audio.append(raw_audio)

        if duo_channel_noise:
            new_audio.append(noise[0])
        else: 
            new_audio.append(noise)
        raw_audio = new_audio
        
    blipped_audio = np.copy(raw_audio)
    # list of intensity, follows the same index
    intensity = []

    for i,b in enumerate(blip_id_list):
        for t in time[i]:
            curr_meta = meta_data_check(meta_data[i],b,raw_rate, filename)
            if duo_channel == True:
                blipped_audio = (audio_blips(b-1, blipped_audio, int(t[0]), int(t[1]), curr_meta))
                #blipped_audio[0] = (audio_blips(b-1, blipped_audio[0], int(t[0]), int(t[1]), curr_meta))
                #blipped_audio[1] = (audio_blips(b-1, blipped_audio[1], int(t[0]), int(t[1]), curr_meta))
                #intensity.append(intensity_to_metadata_calculator(b-1, len(blipped_audio[0])/raw_rate ,time_seconds, curr_meta ))
            else:
             
                blipped_audio = (audio_blips(b-1, blipped_audio, int(t[0]), int(t[1]), curr_meta))
                #intensity.append(intensity_to_metadata_calculator(b-1, len(blipped_audio)/raw_rate ,time_seconds, curr_meta ))
                intensity.append(0)

    # check for if there is no blip
    if len(blip_id_list) == 0:
        intensity.append(0)

    # write to the file
    # expect (data, num channel) for the last argument
    # note that we are using a standard of float 32, since float64 does not seem to work in chrome
    
    if duo_channel:
        if type(blipped_audio[0][0])==np.int16:
            scipy.io.wavfile.write(output_name,raw_rate,np.int16(np.asarray(blipped_audio.T)))
        else:
            scipy.io.wavfile.write(output_name,raw_rate,np.float32( np.asarray(blipped_audio).T))
    else:
        if type(blipped_audio[0])==np.int16:
            scipy.io.wavfile.write(output_name,raw_rate,np.int16(np.asarray(blipped_audio)))
        else:
            scipy.io.wavfile.write(output_name,raw_rate,np.float32( np.asarray(blipped_audio)))
    return intensity











class ABlipEngine:
    def __init__(self, input_path, output_path,noise_path , data_source,noise_source,profiles_csv, continue_df = None):
        '''
        @params: continue_df = path to DF from  last time
            say we have version 1.2 DF from last time and want to append new rows fro each given type use this
        @params: new_df = just a boolean for if we are starting fresh with indexing 
        '''

        # note that these numbers are differnet from what we have in the code else where, these numbers are what we have on the macro
        self.blipid = { 1:'PACKETDROP'  ,
                        2:'PACKETLOSS',
                        3:'LAGGING', 
                        4:'WHITE_NOISE',
                        5:'COMPRESSION',
                        6:'REVERBERATION' ,
                        7:'ECHO',
                        8: 'BANDWIDTH',
                        9:'FR',
                        10:'Overlap',
                        11:'Speaker_issues',
                        12:'Background_noise'}


        self.noiseid = {1:'air_conditioner'  ,
                        2:'car_horn',
                        3:'children_playing', 
                        4:'dog_bark',
                        5:'drilling',
                        6:'engine_idling' ,
                        7:'jackhammer',
                        8:'siren',
                        9:'street_music',
                        10:'wind'}
        self.noiseint = { "AC_0":{0: 0.45,1:0.2,2:0.1}  ,
                        "AC_0_STAT400":{0: 0.45,1:0.2,2:0.1},
                        "AC_1":{0: 0.45,1:0.2,2:0.1},
                        "AC_1_STAT400":{0: 0.45,1:0.2,2:0.1},
                        'bark_0':{0: 0.9,1:0.6,2:0.125},
                        'bark_0_STAT400':{0: 0.9,1:0.6,2:0.125},
                        'bark_1':{0: 0.8 ,1:0.45,2:0.15 },
                        'bark_1_STAT400':{0: 0.8 ,1:0.45,2:0.15 },
                        'car_0':{0: 0.50,1:0.25,2:0.1},
                        'car_0_STAT400':{0: 0.30,1:0.15,2:0.05},
                        'Siren_0':{0: 0.8,1:0.35,2:0.15},
                        'Siren_0_STAT400':{0: 0.8,1:0.35,2:0.15},
                        'Drill_0':{0: 0.95,1:0.55,2:0.15},
                        'Drill_0_STAT400':{0: 0.75,1:0.35,2:0.08},
                        'Drill_1':{0: 0.95,1:0.4,2:0.2},
                        'Drill_1_STAT400':{0: 0.75,1:0.25,2:0.08},
                        'children_1':{0: 0.8,1:0.5,2:0.25},
                        'children_1_STAT400':{0: 0.8,1:0.5,2:0.25},
                        'children_0':{0: 0.9,1:0.6,2:0.3},
                        'children_0_STAT400':{0: 0.9,1:0.6,2:0.3},
                        'wind_0':{0: 0.8,1:0.25,2:0.075},
                        'wind_0_STAT400':{0: 0.8,1:0.25,2:0.075}
                        }

        self.in_path = input_path
        self.out_path = output_path
        self.noise_path = noise_path
        self.data_source = data_source
        self.noise_source = noise_source
        self.BLIP_DATA = pd.read_csv(profiles_csv)
        self.data_source_num = self.get_num_files(input_path,data_source)
        self.f_type = self.get_f_type(input_path,data_source)

        self.blips, self.blip_time_profile, self.meta = self.get_blip_data(self.BLIP_DATA)


        # make new frames 
        # for cont_df, we are doign one df each
        self.df = self.make_new_frame()
        self.cont_df = None
        self.cont_blip_num = {}

        if continue_df is not None:
            self.cont_df = pd.read_csv(continue_df)

            # remove the unamed: 0 thing if exist
            if 'Unnamed: 0' in self.cont_df.columns:
                del self.cont_df['Unnamed: 0']

            # now self.df is blip name |-> df for given blip
            self.df = {}   
            for i in self.blipid.values():
                self.df[i] = self.make_new_frame()


        


       
    def blip_engine(self):
        '''
        Description:
            Welcome to the main blip loop 

        Input:
            For this, checkout the notebook
            num_profiles,data_source, data_source_num,output_folder_path,input_folder_path,df
        Output:
            df: dataframe with the data we need  
        Side Effect:
        
        '''
        # print statements are intentional
        num_profiles = (self.BLIP_DATA.shape)[0]
        print('num_profiles',num_profiles)

        intensity_list = []

      
        # curr num is used to keep track of the current output number
        # start with the first profile, append to same list
        # bkkp bookkeeping, we have num_profiles (ie 44 rows) and total number of files. each file would append to the row 

        # Blip type id -> current output number for given type
        self.curr_num = {}
        # 
        self.bkkp = {}
        version = 1.2

        # Blip type id -> current output number for given type if cont_df
        if self.cont_df is not None:
            self.cont_blip_num = self.get_num_blipped_contdf()
            print('Yay dude', self.cont_blip_num)

        # initialise curr_num
        for i in self.blipid.keys():
            self.curr_num[self.blipid[i]] = 0
            # if we keep using the same df
           
            if self.cont_df is not None and i in self.cont_blip_num :
                # note that self.blipid[i] is 'PACKETLOSS for example'
                self.curr_num[self.blipid[i]] = int(self.cont_blip_num[i]) +1
            self.bkkp[self.blipid[i]] = []



        for i in range(len(self.data_source)):
            num_source = self.data_source_num[i] 
            curr_f_type = self.f_type[i]
            


            for j in tqdm(range(1,num_source+1)):
                source_file = self.data_source[i]+"_"+str(j)+'_'+str(version)
                source_name = self.data_source[i]
                
                for z in range(num_profiles):
                    blip_list = self.blips[z]
                    # gets u to 'Lagging' entry of the dict for ex, then we have a list of len(num profiles)
                    curr_nth = self.curr_num[self.blipid[blip_list[0]]] 

                    # 12 is backgorund noise 
                    
                    if blip_list[0] != 12:
                       
                        ret_int = self.call_blip(z,source_name,blip_list,curr_nth,source_file,curr_f_type)
                    else:
        
                        ret_int = self.call_blip_noise(z,source_name,blip_list,curr_nth,source_file,curr_f_type)
                    intensity_list.append(ret_int)
                    

        

        # write the data into dataframe
        for i in self.blipid.values():

            b_ith = sorted(self.bkkp[i], key = lambda x: x[0])

            for j in b_ith:
                entry = j
                new_row = {'Output Audio ID (.wav)':entry[1] , 'Video/Audio Source (.wav/.mp4)':entry[2] , 'Metadata':entry[3], 'List of Blip IDs':entry[4],'Time Profile':entry[5],'Intensity':entry[6]}
                
                if self.cont_df is None:
                    self.df = self.df.append(new_row, ignore_index=True)
                else:
                    if i in self.df:
                        self.df[i] = self.df[i].append(new_row, ignore_index=True)

        # combine the old with the new df if needed 
        if self.cont_df is not None:
            self.combine_df()


        return 



    def call_blip(self,z,source_name,blip_list,curr_nth,source_file,curr_f_type):
        blip_time = self.blip_time_profile[z]
        meta_data = self.meta[z]
        output_name = self.out_path+ source_name +"_Blipped" + "/"+self.blipid[blip_list[0]]  +"_"+str(curr_nth)+'.wav'

        # this is is used to keep check of current file name, blip_list[0] so the first blip is more important
    
        intensity = blip(self.in_path + source_name +"/" , source_file,list(blip_list),blip_time, output_name,meta_data,curr_f_type )
    
        # update CSV
        new_entry = np.asarray([curr_nth,self.blipid[blip_list[0]] +"_"+str(curr_nth), source_file, meta_data,blip_list , blip_time,intensity[0]])

        # dont append yet, save to vector
        self.bkkp[self.blipid[blip_list[0]]].append(new_entry)
        self.curr_num[self.blipid[blip_list[0]]]  += 1
        return intensity[0]




    def call_blip_noise(self,z,source_name,blip_list,curr_nth,source_file,curr_f_type):

        # make a list consisting of paths to the file itself
       

        names_path, names = self.get_file_names(self.noise_path,self.noise_source)
        blip_time = self.blip_time_profile[z]
        
        meta_data_int = self.meta[z][0]
        

       
        for i,(curr_path,curr_noisesrc) in enumerate(zip(names_path,names)):
            #output_name = self.out_path+ source_name +"_Blipped" + "/"+self.blipid[blip_list[0]]  +"_"+str(curr_nth)+"_"+self.get_name_wo_type(curr_noisesrc)+'.wav'

            noise_src = self.get_name_wo_type(curr_noisesrc)

            
            meta_data = self.noiseint[noise_src][meta_data_int]
            
           
            if source_name == "STAT400" or source_name == "NASA" :
                meta_data = self.noiseint[noise_src+"_"+"STAT400"][meta_data_int]
            
            

            output_name = self.out_path+ source_name +"_Blipped" + "/"+self.blipid[blip_list[0]]  +"_"+str(self.curr_num[self.blipid[blip_list[0]]] )+'.wav'
        
            intensity = blip(self.in_path + source_name +"/" , source_file,list(blip_list),blip_time, output_name,[meta_data],curr_f_type , curr_path)
            # this is is used to keep check of current file name, blip_list[0] so the first blip is more important
    
        
    
            # update CSV
            #new_entry = np.asarray([curr_nth,self.blipid[blip_list[0]] +"_"+str(curr_nth)+"_"+self.get_name_wo_type(curr_noisesrc), source_file, meta_data,blip_list , blip_time,intensity[0]])
            new_entry = np.asarray([curr_nth,self.blipid[blip_list[0]] +"_"+str(self.curr_num[self.blipid[blip_list[0]]]), source_file+"_"+self.get_name_wo_type(curr_noisesrc), meta_data,blip_list , blip_time,intensity[0]])

            # dont append yet, save to vector
            self.bkkp[self.blipid[blip_list[0]]].append(new_entry)
            self.curr_num[self.blipid[blip_list[0]]]  += 1

        return intensity[0]
        

   
        


    def df_to_csv(self,filename):
        self.df.to_csv(filename)
        return




    def get_blip_data(self,BLIP_DATA):
    
        '''
        Description:
            Called to convert data in dataframe to a list

            This is dependent on how the CSV is laid out
        Input:
            BLIP_DATA: pandas dataframe
        Output:
            blips: a [] with IDs for blips, ie [1,2]
            blip time profile: a str specifying which time profile to use
            meta: 
        Side Effect:
        
        '''
        # put dataframe in a list 
        blips = []
        blip_time_profile = []
        meta = []
        for j in range((BLIP_DATA.shape)[0]):
            blips.append(ast.literal_eval(BLIP_DATA["List of Blip IDs"][j]))
            blip_time_profile.append(str(BLIP_DATA["Time Profile"][j]).replace(' ','').split(','))
            meta.append(ast.literal_eval(BLIP_DATA["Metadata"][j]))
            #meta.append(5)
        return blips, blip_time_profile,meta





    '''
    ***************************************************************   Functions that are trivial helpers *********************************************************************

        For both cases, data_sources a list
    '''
    def make_new_frame(self):
        col_list = ['Output Audio ID (.wav)', 'Video/Audio Source (.wav/.mp4)', 'Metadata', 'List of Blip IDs', 'Time Profile', 'Huamn Observed Blips', 'Intensity']
        df = pd.DataFrame(columns=col_list)
        return df
        
    def append_to_current(self):
        csv_filename = "Survey Database - Blip Audio Profule Version 1.1.csv"
        df = pd.read_csv(csv_filename)
        return df

    def get_num_files(self,path, data_source):
        # path is the actual path to folder, ending with /
        filenums = []
        for i in data_source:
            filenums.append(len([name for name in os.listdir(path+i+'/') if not os.path.isfile(name) and name != '.DS_Store' ])) 
        return filenums





    def get_file_names(self,path, data_source):
        # path is the actual path to folder, ending with /
        names_path = np.asarray([])
        names = np.asarray([])
        for i in data_source:
            names_path = np.concatenate((names_path, [path+ i +'/'+ name for name in os.listdir(path+i+'/') if not os.path.isfile(name) and name != '.DS_Store' ] ) )
            names = np.concatenate((names, [name for name in os.listdir(path+i+'/') if not os.path.isfile(name)] ) )
        return names_path,names



    def get_name_wo_type(self,name):
        return name.split('.')[0]
    

    def get_f_type(self,path, data_source):
        f_type = []
        
        for i in data_source:
            files = [name for name in os.listdir(path+i+'/') if  os.path.isfile(path+i+'/'+name) and name != '.DS_Store']
            

            if '.DS_Store' in files:
                files.remove('.DS_Store')
            
           
            if 'wav' in files[0]:
                f_type.append('wav')
            elif  'mp4' in files[0]:
                f_type.append('mp4')
        return f_type

    

    def get_num_blipped_contdf(self):
        '''
        Get the number of blipped per blip type, ret as a dict. blip type id |-> number. IE in the df PACKLETLOSS goes up to PACKETLOSS_1000, blipid_num[2] = 1000 where 2 is the blip type id for 
        packetloss

        Also fill in the current df

        Used for continuing wiritng to the old df
        '''

        blipid_num_strid = {}
        blipid_num = {}

        num_clips_tot = self.cont_df['Output Audio ID (.wav)'].shape[0]
        
        for i in range(num_clips_tot):
            '''
            Counting
            '''

            # ie curr_blip_type_str = 'ECHO'
            curr_blip_type_str = self.cont_df['Output Audio ID (.wav)'][i].split('_')[0]

            # ie curr_blip_type_str = 1000
            curr_blip_type_num = self.cont_df['Output Audio ID (.wav)'][i].split('_')[1]

            #handle exception for white noise
            if curr_blip_type_str == 'WHITE':
                curr_blip_type_str = 'WHITE_NOISE'
                curr_blip_type_num = self.cont_df['Output Audio ID (.wav)'][i].split('_')[2]


            #type cast for comparsion
            curr_blip_type_num = int(curr_blip_type_num)
            if curr_blip_type_str in blipid_num_strid:
                if curr_blip_type_num > blipid_num_strid[curr_blip_type_str]:
                    blipid_num_strid[curr_blip_type_str] = curr_blip_type_num
            else:
                blipid_num_strid[curr_blip_type_str] = curr_blip_type_num

            '''
            Moving
            '''
            curr_row = {'Output Audio ID (.wav)':self.cont_df['Output Audio ID (.wav)'][i] , 'Video/Audio Source (.wav/.mp4)': self.cont_df['Video/Audio Source (.wav/.mp4)'][i] , 'Metadata': self.cont_df['Metadata'][i], 'List of Blip IDs': self.cont_df['List of Blip IDs'][i],'Time Profile': self.cont_df['Time Profile'][i],'Intensity': self.cont_df['Intensity'][i]}
            
            self.df[curr_blip_type_str] = self.df[curr_blip_type_str].append(curr_row, ignore_index=True)
            
        print("HERRRRRRRRO", blipid_num_strid)

        for blip_id, blip_type in self.blipid.items():
            # safety check to see for example if PACKETLOSS is in blipid_num_strid
            if blip_type in blipid_num_strid:
                val = blipid_num_strid[blip_type] 
                blipid_num[blip_id] = val

        return blipid_num
            

    def combine_df(self):
        # beautiful references
        frames = []
        for blip_type in self.df.keys():
            frames.append(self.df[blip_type])
        self.df = pd.concat(frames)

        return 

