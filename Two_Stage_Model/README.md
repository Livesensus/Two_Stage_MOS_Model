# Livesensus Two-Stage Model

Livesensus two-stage model assesses the quality of an audio based on human perception by predicting the Mean Opinion Score (MOS) of a given audio. The first stage (stage 1) of the model will preprocess the audio and detect the evidence of echo, packetloss, dynamic range compression, and vocal interruption. The outputs of the stage 1, an array of artifact detectors' outputs, will then be passed to stage 2. The stage 2 is a pre-trained Support Vector Machine (SVM) model that takes the output array of stage 1 and predicts MOS score of the audio in the range of 1 to 5 (1 being the poorest qualitym and 5 being the best quality).

## Installation

Create a virtual environment and install required packages.

```bash
$ virtualenv LS_env

$ source LS_env/bin/activate

$ pip install -r requirements.txt
```

## Usage

Use the following command line to run the program. 

```bash
$ python livesensus.py VCC_sample_audio.wav 
```

Alternatively, the program can also be excuted in Jupiter Notebook. An example is shown in the ```sample_livesensus.ipynb``` file.


## File Description 
### livesensus.py

This file is the top level file that executes the stage 1 and stage 2 program. By default, it takes an audio file name as input from command line and returns the MOS score for every "sec_seg".

``` python
model = model_1_2_pipeline.Audio_Model(fname=a_fname, input_signal=False, sec_seg=1, video=False, video_fortran=False)

model.inference_all(verbose=False)

print(model.mos_time)
```
```model_1_2_pipeline.Audio_Model``` declares the stage 1 and 2 model.

**Parameters:**
| Parameter | Description|
| ------------- |:-------------:| 
| input_signal| If true then make sure 1) x 2) sr are in the input. If False then make sure 1) fname are in the input. This allows the class to take in filanme and load the audio in the class or have it in the input|
| fname | If input_signal = False, and video = True then it is the path/name to the video. If not then it is the name if the audio |
|sec_seg| seconds per segment. For example, audio file is 6 seconds we would do 6/3 = 2 seconds of inference. In this case if audio file is 10 seconds we just discard the last 1 seconds since 10/3 = 3 so only 3 segments|
|x| If input_signal is true, then it is the input. It can be 1D array or 2D array float or int since they are handled by the model |
|sr| sampling rate of the audio if input_signal is true|
|video| If true, then we will extract audio from the video|
|video_fortran| apply np.asfortranarray( . ) to signal, error for some reason sometimes when extracting audio from video|


### model_1_2_pipeline.py

Pipeline for the whole model 1 and 2 audio model. Supports singal and duo channel audio.

### model_1_pipeline.py
Pipeline for the model 1 which takes an audio waveform as input and runs the various blip detectors over the audio. It returns the detectors' outputs to the model 2.


### model_1_modules.py
This file contains all of the blip detectors.
| Detector | Description|
| ------------- |:-------------:| 
|zero_frame_detection|return the start and end times of zero-frame intervals in the given audio in a list|
|packetloss_intensity|Intensity module for packetloss|
|detectEcho|performs spectrogram autocorrelation, searches for peaksthat would correlate to echo. This can detect echo that has at least a 0.1 volume multiplier (or higher) and has at least a delay of 30 ms|
|bandwidth_est|This module predicts the frequency bandwidth that is being transmitted. While Opus and other codecs sometimes restruct to 4000, 6000 and others, this predictor will not return one of these numbers|
|latency_detector|This module predicts the amount of none interrupted speech in the audio in percentatge. It used a pre-trained SVM model (clf_SVM.p). The audio will be preprocessed into desired dimension before passing to the model|


### preprocessing.py
Helper functionns that preprocess the audio from waveform to desired format.

### audio_extraction.py
Helper function that extract audio from video file.


### requirements.txt
All the required packages to execute the program.

### clf_SVM.p
Pre-trained SVM model for latency detection.

### finalized_model.sav
Pre-trained model 2 for MOS reference.



