import numpy as np
import random
import os
import soundfile as sf
import subprocess
import librosa
import random
import scipy.io.wavfile
import pandas as pd


"""
Terms:

Absolute samples: 1 second of audio with 44100Hz of sampling rate has 44100 samples per second, absolute means the 12th sample in that respect


"""




'''
Description:
    Drops blocks of audio in given time frame, number of blocks are detemined by intensity calculated and passed in as meta_data
    blip the audio for the given start and end times
Input: 
    audio: the input audio 
    start: the start of the blip in absolute samples  (default=0)
    end: the end of the blip in absolute samples (default=0, if 0, then blip entire audio)
    meta_data: how many blocks to be dropped (default = 5) 
                
Output: 
    the blipped audio

Side effect:

'''
def audio_packet_drop(ori_audio, start=0, end=0, meta_data = 5):
    #print("Packet Drop")
    # if no end time
    if end == 0:
        end = len(ori_audio)

    # induce times number of 0s between start and end with equal spacing
    duration = end - start   # in terms of number of samples
    block_size = int(duration/(meta_data))

    audio = ori_audio.copy()
    """
    for i in range(1,meta_data+1):
        
        if i %2 == 0:
            curr_st = start + block_size*(i-1)
            curr_end = start + block_size*i
            audio[curr_st:curr_end] = 0
    """
    audio[start:end] = 0
    return audio





'''
Description:
    simulate audio effect of packetLoss
Input: 
    audio: the input audio 
    start: the start of the blip in sampling frequncy (default=0)
    end: the end of the blip in sampling frequncy (default=0, if 0, then blip entire audio)
    meta_data: (packet loss mode, percentage of the packet loss, frame size of a packet in time, sampling rate)
                
                Examples:   percentage of packet loss, 0.3 means 30 percent
                            frame size: 0.02 means 20 ms frames, ideally we would have 10,20, and 40 as frame size
                            in some cases it is possible to have 120ms packets

Output: 
    the blipped audio

Side effect:

'''
def audio_packetLoss(audio, start=0, end=0, meta_data = (0,0.3,0.02,44100)):
    #print("PacketLoss")
    # if end==0, blip entire audio
    if end==0:
        end = len(audio)

    sr = meta_data[3]
    frame = meta_data[2]
    mode = meta_data[0]

    samples_per_frame = int(sr*frame) 
    percentage = meta_data[1] # 30 % of the audio would have packet loss

    percent_dropped = int(len(audio)*percentage)
    num_packets_dropped = int(percent_dropped/samples_per_frame)

    data = audio.copy()

    '''
    This is for mode 0, zero filling, simply replacing them with zeros
    '''
    if mode == 0:
        for i in range(num_packets_dropped):
            st = random.randrange(len(audio)-samples_per_frame-1)
            data[st:st+samples_per_frame] = 0

    '''
    This is for mode 1, we are now doing LPC for the missing packets, which is used in Opus
    '''
    return data






'''
description: 
    lag the audio for the given time and then fast playback 

input:
    file_name: the name of the WAV file
    start: start of the lagging in second
    end: end of the lagging in second
    metadata: the speed of the playback data (default = 1.5)

output:
    audio: the blipped audio, contains both lagging and playback

side effect:
    output file saved locally

'''

def audio_lag(audio, start=0, end=0, metadata = (1.5,44100)):
   
    input_length = len(audio)
    raw_sr = metadata[1]
    file_name = 'placeholder.wav'
    sf.write(file_name,audio,raw_sr)

    # convert back to seconds because i am lazy
    start = int(start/raw_sr)
    end = int(end/raw_sr)


    speed = metadata[0]
    output_name = "out.wav"
    cmd = "ffmpeg -i " + file_name + " -y -filter:a atempo=" + str(speed) + " " + output_name
    subprocess.run(cmd, shell=True)

    audio_r, sr = librosa.core.load(file_name, sr=raw_sr, mono=True)
    audio_c, sr = librosa.core.load(output_name, sr=raw_sr, mono=True)
    audio = audio_r.copy()
    audio[start*sr:end*sr] = 0
    playback = audio_c.copy()
    playback = playback[int(start*sr/speed):int(end*sr/speed)]
    audio = np.concatenate([audio[:end*sr], playback, audio[end*sr:]])
    cmd = "rm " + output_name
    subprocess.run(cmd, shell=True)
    librosa.output.write_wav(output_name, audio, sr)

    
    audio_done, sr = librosa.core.load(output_name, sr=None, mono=True)
    
    
    os.remove(file_name)
    os.remove(output_name)

   
    if len(audio_done) > input_length:
        return audio_done[0:input_length]
    else:
        return audio_done









'''
Description:
    blip the audio by adding white noise
Input: 
    audio: the input audio 
    start: not used
    end: not used
    meta_data: level of the artifact, from levels 0 - 100 -> higher level, more noise
            (default=20)

Output: 
    the blipped audio

Side effect:

'''
def white_noise(audio, start=0, end=0, meta_data=20): #from levels 0 - 100 -> higher level, more noise
    #print("White Noise")
    level = meta_data
    starting_w=0.0001
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * np.random.random_sample(len(audio[start:end]))
    if type(audio[0]) == np.int16:
        noise = noise_level * np.random.random_sample(len(audio[start:end]))*32767

    audio[start:end] += type(audio[0])(noise)

    
    return audio



'''
Description:
    blip the audio by simulate the compression artifacts
Input: 
    audio: the input audio 
    start: not used
    end: not used
    meta_data: (int1,int2,int3,int4)   int 1 is codec, 0 for AAC-LC, 1 for OPUS, 2 for AAC but ffmepg built in
                int2 is bitrate, refer to the survey database sheet 
                int3 is complexility. This only applies for OPUS 
                int4 is the sampling rate of this audio, this will be overrided by the audio's sampling rate

    THIS IS THE OLD ONEmeta_data: level of the artifact, from levels 0 - 100 -> higher level, more noise
            (default=20)

Output: 
    the blipped audio

Side effect:
    some of the compresson data will be printed in terminal

'''
def compression(audio, start=0, end=0, meta_data=(0,16,0,44100)): #from levels 0 - 100 -> higher level, more noise
   
    input_length = len(audio)
    codec_used = int(meta_data[0])
    bitrate = str(meta_data[1])
    complexity = int(meta_data[2])
    sampling_rate = int(meta_data[3])

    usage = "meta"
    curr = "TEMP"+bitrate+str(codec_used)+str(complexity)
    os.mkdir(curr)

    codec = None
    if codec_used == 0:
        codec = "libfdk_aac -profile:a aac_low -eld_sbr 1 -y -b:a "+bitrate+"k "
        #codec = "libfdk_aac -profile:a aac_low -eld_sbr 1 -y -vbr 1"
    elif codec_used == 1:  
        codec = "libopus -application:a voip -y -compression_level "+str(complexity)+ " -packet_loss 100 -frame_duration 60 -vbr off -b:a "+bitrate+"k "
    else:
        codec = "aac -profile:a aac_low -y  -b:a "+bitrate+"k "
    
    
    sf.write(curr+'/3.wav',audio,sampling_rate)

   
    """
    ffmpeg usage:
    -vn: blocks all video stream if any
    -ar: sets the sampling freq
    """
    outname = "opus"+bitrate+"k_LC"
    
    bashCommand = None
    if codec_used == 1:
        #bashCommand = "ffmpeg -i "+curr+"/3.wav"+" -ar "+str(16000) + "  -c:a "+codec+ curr+"/"+outname+".opus"
        bashCommand = "ffmpeg -i "+curr+"/3.wav"+ "  -c:a "+codec+ curr+"/"+outname+".opus"
    else:
        bashCommand = "ffmpeg -i "+curr+"/3.wav"+" -c:a "+codec+ curr+"/"+outname+".m4a"
    
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


    bashCommand_reverse = None
    if codec_used == 1:
        bashCommand_reverse = "ffmpeg  -c:a libopus -i "+curr+"/"+outname+".opus "+ " -ar "+str(sampling_rate) +" -y -c:a pcm_f32le "+curr+"/"+outname+".wav"
    else:
        bashCommand_reverse = "ffmpeg  -c:a libfdk_aac -i "+curr+"/"+outname+".m4a "+" -y -c:a pcm_f32le "+curr+"/"+outname+".wav"


    process = subprocess.Popen(bashCommand_reverse.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    audio,sr = sf.read(curr+"/"+outname+".wav")
    os.remove(curr+"/3.wav")
    if codec_used == 1:
        os.remove(curr+"/"+outname+".opus")
    else:
        os.remove(curr+"/"+outname+".m4a")
    os.remove(curr+"/"+outname+".wav")
    os.rmdir(curr)
    
   

    if len(audio) > input_length:
        return audio[0:input_length]
    else:
        return audio






'''
    Reverberation in principle should be applied to the whole audio

Description:
   introduce reverberations, start and end are not used
   

Input: 
    audio: the input audio 
    start: not used
    end: not used
    meta_data: (sampling rate, time delay, decay constant)

Output: 
    the blipped audio

Side effect:
    None

'''
def reverberation(audio,start = 0, end = 0, meta_data = (44100,0.07,0.5) ):
    reverb = audio.copy()
    SAMPLING_RATE = meta_data[0]
    reverb_time = meta_data[1]
    decay_constant = meta_data[2]
    
    
    delay = int(reverb_time*SAMPLING_RATE)
    delay1 = int(reverb_time*2*SAMPLING_RATE)
    delay2 = int(reverb_time*3*SAMPLING_RATE)
    delay3 = int(reverb_time*4*SAMPLING_RATE)
    delay4 = int(reverb_time*5*SAMPLING_RATE)
    delay5 = int(reverb_time*6*SAMPLING_RATE)
    delay6 = int(reverb_time*7*SAMPLING_RATE)
    for i in range(delay, len(audio)):
        reverb[i] =  audio[i] + audio[i-delay]*0.9
        decay = decay_constant*0.9
        reverb[i] =  reverb[i] + audio[i-delay1]*decay
        decay = decay*decay_constant
        reverb[i] =  reverb[i] + audio[i-delay2]*decay
        decay = decay*decay_constant
        reverb[i] =  reverb[i] + audio[i-delay3]*decay
        decay = decay*decay_constant
        reverb[i] =  reverb[i] + audio[i-delay4]*decay
        decay = decay*decay_constant
        reverb[i] =  reverb[i] + audio[i-delay5]*decay
        decay = decay*decay_constant
        reverb[i] =  reverb[i] + audio[i-delay6]*decay

    return reverb



'''
    Echo generation

Description:
   introduce echo, start and end are not used
   

Input: 
    audio: the input audio 
    start: not used
    end: not used
    meta_data: (sampling rate, time delay, decay constant)

Output: 
    the blipped audio

Side effect:
    None

'''
def echo(audio,start = 0, end = 0, meta_data = (44100,0.1,0.9) ):
    time_delay = meta_data[1]
    decay_constant = meta_data[2]

    echoed = audio.copy()
    delay = int(time_delay*meta_data[0])
    for i in range(delay, len(echoed)):
        echoed[i] =  audio[i] + audio[i-delay]*decay_constant


    return echoed



'''
Description:
   Adding background noise with different volume
   

Input: 
    audio: the input audio 
    start: not used
    end: not used
    meta_data: (sampling rate, not used, volume from 0-1 float)

Output: 
    the blipped audio

Side effect:
    None

'''
def background_noise(audio,start = 0, end = 0, meta_data = 0.2 ): 
    
    volume = meta_data
    return audio[0]+volume*audio[1]


'''
Description:
   Produce a 'muffled' effect on the audio 
   
Input: 
    audio: the input audio 
    start: not used
    end: not used
    meta_data: (codec, cut off frequency, complexity, sampling rate, filename)
        - codec: 0 for AAC-LC, 1 for OPUS, 2 for AAC but ffmepg built in
        - cut off frequency: for opus only accepts 4000,6000, 8000, 12000, 20000. 
        - complexity: complexility. This only applies for OPUS 
        - sampling rate of this audio, this will be overrided by the audio's sampling rate
        - filename: path to the audio

Output: 
    the blipped audio

Side effect:
    None

'''
def bandwidth_limited(audio,start = 0, end = 0, meta_data =(0,16,4000,44100,'')):

    input_length = len(audio)
    codec_used = int(meta_data[0])
    cut_off = str(meta_data[1])
    complexity = int(meta_data[2])
    sampling_rate = int(meta_data[3])
    file_name = str(meta_data[4])

    
    curr = "TEMP"+cut_off+str(codec_used)+str(complexity)
    os.mkdir(curr)

    codec = None
    if codec_used == 0:
        codec = "libfdk_aac -profile:a aac_low -eld_sbr 0 -y -b:a 50k -cutoff "+cut_off+" -vbr 5 "
       
    elif codec_used == 1:  
        codec = "libopus -application:a voip -y -compression_level "+str(complexity)+ " -packet_loss 0 -frame_duration 60 -vbr off -b:a 50k -cutoff "+cut_off +" "
    else:
        codec = "aac -profile:a aac_low -y  -b:a "+cut_off+"k "
    
    
   
    """
    ffmpeg usage:
    -vn: blocks all video stream if any
    -ar: sets the sampling freq
    """
    outname = "opus"+cut_off+"k_LC"
    
    encode_command = None
    if codec_used == 1:
        encode_command = "ffmpeg -i "+file_name+"  -acodec "+codec+ curr+"/"+outname+".opus"
    else:
        encode_command = "ffmpeg -i "+file_name+" -acodec "+codec+ curr+"/"+outname+".m4a"
    
    process = subprocess.Popen(encode_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    
    decode_command = None
    if codec_used == 1:
        decode_command = "ffmpeg  -c:a libopus -i "+curr+"/"+outname+".opus "+ " -ar "+str(sampling_rate) +" -y -c:a pcm_f32le "+curr+"/"+outname+".wav"
    else:
        decode_command = "ffmpeg  -c:a libfdk_aac -i "+curr+"/"+outname+".m4a "+ " -ar "+str(sampling_rate) +" -y -c:a pcm_f32le "+curr+"/"+outname+".wav"
    process = subprocess.Popen(decode_command.split(), stdout=subprocess.PIPE)
   
    output, error = process.communicate()


    audio,sr = librosa.load(curr+"/"+outname+".wav", sr = sampling_rate)
    
    if codec_used == 1:
        os.remove(curr+"/"+outname+".opus")
    else:
        os.remove(curr+"/"+outname+".m4a")
    os.remove(curr+"/"+outname+".wav")
    os.rmdir(curr)
    
   

    if len(audio) > input_length:
        return audio[0:input_length]
    else:
        return audio

    return


def freq_range():
    return


def overlap_speaker():
    return
def Speaker_issues():
    return
'''
Description:
    helper funcion that calls the audio blip function based on the given input

Input:
    audio_blip_id: the ID of the audio blip
    audio: the input wavefile array, always a 1D array
    start: the start frame number of the audio want to be blipped. (ie. start time in seocnd * sampling rate)
            (default=0, start at 0 second)
    end: the end frame number of the audio want to be blipped. 
            (default=0, end at 0 second)
    meta_data: different parameters for different blips. eg, for compresson is 'level'
                (default=sampling frequncy=44100)

Output:
    None

Side Effect:
    call the audio blip funtion based on the given ID

'''
def audio_blips(audio_blip_id, audio, start=0, end=0, meta_data=44100):
    '''
    This following list follows the documentation in Survey Database Google Sheets. They should not be changed. 

    0 = audio_packet_drop
    1 = audio_packetLoss
    2 = audio_lag
    3 = white_noise
    4 = compression
    5 = reverberation
    6 = echo
    7 = Bandwidth Limited
    8 = Frequency Range
    9 = Overlap Speaker
    10 = Speaker Isses
    11 = Background Noise
    '''
    audio_blip_list = [audio_packet_drop, audio_packetLoss, audio_lag, white_noise, compression, reverberation, echo,bandwidth_limited ,freq_range,overlap_speaker,Speaker_issues, background_noise]
   
    return audio_blip_list[audio_blip_id](audio, start, end, meta_data)




"""
ARCHIVE




def compression(audio, start=0, end=0, meta_data=(0,16)): #from levels 0 - 100 -> higher level, more noise
    #print("Compression")
    starting_w=300.0
    ending_w = 1.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=meta_data
    noise_level = starting_w * (F1 ** (jk))
    noise_level=round(noise_level,4)
    
    usage = "meta"
    
    os.mkdir(usage)
    
    sf.write(os.path.join(usage,'3.wav'),audio,44100)
    
   
    bashCommand = "ffmpeg -i "+usage+"/3.wav"+" -vn -ar 44100 -b:a " + str(noise_level)+"k "+usage+"/3.mp3"
    # print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand_reverse = "ffmpeg -i "+usage+"/3.mp3 "+usage+"/4.wav"
    # print(bashCommand_reverse)
    process = subprocess.Popen(bashCommand_reverse.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    audio,sr =sf.read(usage+"/4.wav")
    os.remove(usage+"/3.mp3")
    os.remove(usage+"/3.wav")
    os.remove(usage+"/4.wav")
    
    os.rmdir(usage)



"""