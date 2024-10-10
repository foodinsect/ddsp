import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings and info messages

import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import librosa

def get_audio(sample_rate=44100, record_or_upload="Upload (.mp3 or .wav)", record_seconds=30):
    if record_or_upload == "Record":
        print("Recording is not supported in this script.")
        return None, None
    else:
        print("=== Debug: Starting file upload process ===")
        Tk().withdraw()  # Hide the root window
        file_path = askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        print(f"Selected file: {file_path}")

        if file_path:
            try:
                audio, sr = librosa.load(file_path, sr=sample_rate)
                print(f"Audio loaded successfully: shape = {audio.shape}, sample rate = {sr}")
                return audio[np.newaxis, :], sr
            except Exception as e:
                print(f"Error loading audio file: {e}")
                return None, None
        else:
            print("No file selected.")
            return None, None
