# Speech to Text using Deep Learning

This notebook demonstrates how to use DeepSpeech, a state-of-the-art open-source speech recognition engine developed by Mozilla, to transcribe audio files into text. The notebook uses a pre-trained model to perform the transcription.

## Dependencies
To use this notebook, you will need to install DeepSpeech and its pre-trained models. This can be done using the following commands:


- !pip install deepspeech
- !wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
- !wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

Additionally, the following dependencies are required:

- wave
- numpy
- librosa
- matplotlib
- IPython

### Loading the DeepSpeech Model
The pre-trained model and its scorer are loaded using the following code:

model_file_path = '/kaggle/working/deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_file_path)

scorer_file_path = '/kaggle/working/deepspeech-0.9.3-models.scorer'
model.enableExternalScorer(scorer_file_path)

### Helper Functions
The following helper functions are defined in this notebook:

- transcribe_audio_file(): loads an audio file, transcribes it using the pre-trained model, and prints the transcribed text and the original transcription (if available).


- play_audio_file(): loads an audio file and plays it using IPython's Audio module.


- show_results(): combines the transcribe_audio_file() and play_audio_file() functions to display the original audio file and the transcribed text.


- show_graphs(): loads an audio file, computes its spectrogram, and displays the waveform and spectrogram using librosa and matplotlib.
Testing out the model


The notebook includes two audio files (audio_file and audio_file_2) from the **LJSpeech dataset**, which are used to test the performance of the model. The show_results() and show_graphs() functions are called on each file to display the original audio, the transcribed text, and the spectrogram.




