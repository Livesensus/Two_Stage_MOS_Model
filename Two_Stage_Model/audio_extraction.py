import moviepy.editor

'''
Description:
    extract audio from video file

Input:
    video_filename: the name of the video file (eg. video_file1.mp4)
    path_for_audio: path to save the audio file 
                        (default = "", save to the same folder as the code. Or, eg. "audio_files_folder/")

Output:
    sampling rate, Audio: we converted to an array like from an AudioFileClip Object
                        Important***, sampling rate audio.fps should be the natural sampling rate of the audio, no need for modication
    (num samples, num channels), ie (5000, 2) so 2 channels
                        Checkout the documentation: https://zulko.github.io/moviepy/ref/AudioClip.html#moviepy.audio.io.AudioFileClip.AudioFileClip
    

Side effect:
    the audio of the video will be saved locally, with the name of the video + ".wav"
'''
def extract_audio(video_filename, path_for_audio="", get_audio = True):
    # read the audio from the video file
    #https://zulko.github.io/moviepy/getting_started/audioclips.html
    # this is an AudioFileClip object in moviepy
    video = moviepy.editor.VideoFileClip(video_filename)
    audio = video.audio

    if get_audio == False:
        # save the audio file
        audio_name = path_for_audio + video_filename[:-4] + ".wav"
        audio.write_audiofile(audio_name)
        return
    else:
        # writing a wav like buffe
        return audio.fps,audio.to_soundarray()

#extract_audio("out1.mp4", "./")
