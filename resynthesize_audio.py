import os
import ddsp
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf

# 고품질 오디오 저장 함수
def save_audio(audio, sample_rate=16000, filename="output.wav"):
    try:
        # EagerTensor를 numpy 배열로 변환 후 flatten
        audio = np.array(audio).flatten()
        sf.write(filename, audio, sample_rate, subtype='PCM_24')  # 고품질 인코딩 사용
        print(f"Audio saved as {filename}")
    except Exception as e:
        print(f"Error during audio saving: {e}")

# 향상된 스펙트로그램 시각화 및 저장 함수
def specplot(audio, sample_rate=16000, title="Spectrogram", filename=None):
    try:
        # EagerTensor를 numpy 배열로 변환 후 flatten
        audio = np.array(audio).flatten()
        print(f"Generating spectrogram for audio with shape: {audio.shape}, sample rate: {sample_rate}")
        plt.figure(figsize=(14, 6))
        # 더 높은 FFT 크기와 해상도 적용
        D = librosa.amplitude_to_db(librosa.stft(audio, n_fft=2048, hop_length=512), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()

        # 파일 저장
        if filename:
            plt.savefig(filename)
            print(f"Spectrogram saved as {filename}")

        plt.show()
    except Exception as e:
        print(f"Error during spectrogram generation: {e}")

def resynthesize_audio(model, audio_features, audio_features_mod=None, segment_index=None, model_name="Model"):
    af = audio_features if audio_features_mod is None else audio_features_mod

    try:
        print("Generating audio using the model...")
        outputs = model(af, training=False)
        audio_gen = model.get_audio_from_outputs(outputs)
    except Exception as e:
        print(f"Error during audio synthesis: {e}")
        return

    # 파일 이름에 세그먼트 인덱스와 모델 이름 추가
    segment_info = f"segment_{segment_index}" if segment_index is not None else "full"
    original_filename = f"original_{model_name}_{segment_info}.wav"
    resynthesized_filename = f"resynthesized_{model_name}_{segment_info}.wav"
    spec_original_filename = f"spec_original_{model_name}_{segment_info}.png"
    spec_resynthesized_filename = f"spec_resynthesized_{model_name}_{segment_info}.png"

    print('Original')
    save_audio(audio_features['audio'], filename=original_filename)

    print('Resynthesis')
    save_audio(audio_gen, filename=resynthesized_filename)

    print("Generating spectrograms...")
    specplot(audio_features['audio'], title=f"Original {model_name} {segment_info}", filename=spec_original_filename)
    specplot(audio_gen, title=f"Resynthesis {model_name} {segment_info}", filename=spec_resynthesized_filename)
