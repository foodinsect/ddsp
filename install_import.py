import warnings
warnings.filterwarnings("ignore")

import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings and info messages

import time

import crepe
import ddsp
import ddsp.training
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# 대체된 기능 (로컬 환경에서 파일 업로드 및 오디오 재생)
import sounddevice as sd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

DEFAULT_SAMPLE_RATE = 16000  # 16000

sample_rate = DEFAULT_SAMPLE_RATE

# 대체된 오디오 재생 함수
def play(audio, sample_rate=DEFAULT_SAMPLE_RATE):
    print("=== Debug: Playing Audio ===")
    print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}")
    try:
        sd.play(audio.flatten(), samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error during audio playback: {e}")

# 대체된 파일 업로드 함수
def upload():
    Tk().withdraw()
    print("=== Debug: Uploading File ===")
    file_path = askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        print(f"File selected: {file_path}")
        try:
            audio, _ = librosa.load(file_path, sr=sample_rate)
            print(f"Loaded audio shape: {audio.shape}")
            return [file_path], [audio]
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return [], []
    else:
        print("No file selected.")
        return [], []

# 대체된 오디오 시각화 함수
def specplot(audio, sample_rate=DEFAULT_SAMPLE_RATE):
    print("=== Debug: Plotting Spectrogram ===")
    try:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio.flatten()), ref=np.max),
                                 sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()
    except Exception as e:
        print(f"Error during spectrogram plotting: {e}")
