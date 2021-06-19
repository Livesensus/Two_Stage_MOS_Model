import sys
import model_1_2_pipeline

def livesensus(filename):
    a_fname = filename
    model = model_1_2_pipeline.Audio_Model(fname=a_fname, input_signal=False, sec_seg = 1, video = False, video_fortran = False)
    model.inference_all(verbose=False)
    print(model.mos_time)
    print("average MOS: ", sum(model.mos_time)/len(model.mos_time))



if __name__ == "__main__":
    # make sure the audio file name is provided in command line
    if len(sys.argv) < 2:
        raise Exception("Please provide filename")
    else:
        livesensus(sys.argv[1])
