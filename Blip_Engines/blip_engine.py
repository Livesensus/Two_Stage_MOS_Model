import sys
import argparse
from audio_blips import audio_blips
import audio_extraction
import audio_blip_engine
import librosa
import os




def parse_arguments():
    parser = argparse.ArgumentParser()
    # input paths
    parser.add_argument("-i", "--input_folder_path", help="path to input folder", required=True)
    parser.add_argument("-n", "--noise_folder_path", help="Path to folder with noise contents", required=True)
    parser.add_argument("-o", "--output_folder_path", help="Path to noise", required=True)
    parser.add_argument("-p", "--csv_filename", help="Path to csv used as profile", required=True)


    
    return parser.parse_args()


def blip_engine(filename,blip_id_list, profile, output_name, meta_data,filetype, noise_path = None):
    '''
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
    audio_blip_engine.blip_engine_final(filename,blip_id_list, profile, output_name, meta_data,filetype, noise_path = None)
    print("  ss")
    return


if __name__ == "__main__":

    args = parse_arguments()
   
    input_folder_path = args.input_folder_path
    output_folder_path = args.output_folder_path
    noise_folder_path = args.noise_folder_path
    #csv for blip profiles
    csv_filename = args.csv_filename


    data_source = [name for name in os.listdir(input_folder_path) if not os.path.isfile(name)]
    noise_source = [name for name in os.listdir(noise_folder_path) if not os.path.isfile(name)]


    engine = audio_blip_engine.ABlipEngine( input_folder_path, output_folder_path,noise_folder_path, data_source,noise_source,csv_filename, csv_continue)
    engine.blip_engine()

    engine.df_to_csv('outputs.csv')
   
