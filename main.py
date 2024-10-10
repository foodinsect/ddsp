import numpy as np
import soundfile as sf
import ddsp
import math  # math 모듈 임포트 
from tensorflow.python.ops.numpy_ops import np_config

def split_audio(audio, sr, segment_duration=4.0):
    """4초 단위로 오디오를 분할합니다."""
    # 오디오가 2차원 배열일 경우 1차원으로 변환
    if len(audio.shape) > 1:
        audio = audio.flatten()

    segment_length = int(segment_duration * sr)
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    print(f"Total segments created: {len(segments)} with each segment duration: {segment_duration} seconds.")
    return segments

def concatenate_audio(segments):
    """분할된 오디오 조각들을 순서대로 이어붙입니다."""
    return np.concatenate(segments)

def process_segments(model, audio_segments, sr, dataset_stats, model_name):
    """각 오디오 세그먼트를 처리하여 합친 결과를 반환합니다."""
    processed_segments = []

    for i, segment in enumerate(audio_segments):
        print(f"=== Processing segment {i + 1} / {len(audio_segments)} ===")
        # Compute features
        ddsp.spectral_ops.reset_crepe()
        audio_features = ddsp.training.metrics.compute_audio_features(segment)

        # Convert loudness to float32
        audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)

        # Modify conditioning
        audio_features_mod = modify_conditioning(audio_features, dataset_stats)
        if audio_features_mod is None:
            print("Error in modify_conditioning. Skipping this segment.")
            continue

        # Resynthesize segment
        try:
            outputs = model(audio_features_mod, training=False)
            audio_gen = model.get_audio_from_outputs(outputs)
            # 원래 세그먼트 길이에 맞게 출력 조정
            audio_gen = np.pad(audio_gen.numpy().flatten(), (0, max(0, len(segment) - len(audio_gen))), 'constant')
            processed_segments.append(audio_gen[:len(segment)])  # 세그먼트 길이에 맞춤
        except Exception as e:
            print(f"Error in segment processing: {e}")

    return processed_segments

if __name__ == "__main__":
    from install_import import *
    from upload import get_audio
    from load_model import load_model
    from resynthesize_audio import resynthesize_audio
    from modify_conditioning import modify_conditioning
    from combine import adjust_volume, combine_audio_files  
    from tensorflow.python.ops.numpy_ops import np_config

    
    # TensorFlow 넘파이 호환 모드 활성화
    np_config.enable_numpy_behavior()

    # 모델 맵핑: 사용 가능한 모델 목록
    available_models = ['Violin', 'Flute', 'Tenor_Saxophone', 'Trumpet', 'Flute2']

    # 모델 선택
    print("Available models:")
    for idx, model_name in enumerate(available_models):
        print(f"{idx + 1}. {model_name}")

    try:
        selected_idx = int(input("Select a model by entering the corresponding number: ")) - 1
        if selected_idx < 0 or selected_idx >= len(available_models):
            raise ValueError("Invalid selection.")
        selected_model = available_models[selected_idx]
    except ValueError:
        print("Invalid input. Please enter a valid number corresponding to the model.")
        exit()

    # Record or Upload Audio
    print("=== Debug: Start recording or uploading audio ===")
    audio, sr = get_audio(sample_rate=16000)
    if audio is None:
        print("No audio loaded.")
        exit()

    # 오디오가 2차원 배열인지 확인 후 1차원으로 변환
    if len(audio.shape) > 1:
        audio = audio.flatten()

    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")

    # Split audio into 4-second segments if longer than 4 seconds
    segment_duration = 4.0
    audio_segments = split_audio(audio, sr, segment_duration=segment_duration)

    # 각 세그먼트 길이 출력
    for idx, segment in enumerate(audio_segments):
        print(f"Segment {idx + 1}: Length = {len(segment) / sr:.2f} seconds")

    # Load model
    print("=== Debug: Loading model ===")
    model_dir, dataset_stats = load_model(model_name=selected_model)
    if model_dir is None or dataset_stats is None:
        print("Model loading failed.")
        exit()

    print(f"Model directory: {model_dir}, Dataset stats loaded: {dataset_stats is not None}")

    # Resynthesize Audio
    print("=== Debug: Resynthesizing audio ===")
    model = ddsp.training.models.Autoencoder()
    try:
        model.restore(model_dir)
        print(f"Model restored from {model_dir}")
    except Exception as e:
        print(f"Error restoring model: {e}")
        exit()

    # Process each audio segment and concatenate results
    processed_segments = process_segments(model, audio_segments, sr, dataset_stats, selected_model)
    if not processed_segments:
        print("No segments were successfully processed.")
        exit()

    # Concatenate all processed segments into one audio
    final_audio = concatenate_audio(processed_segments)

    # Save the final audio output with the selected model name
    output_filename = f"{selected_model}.wav"
    sf.write(output_filename, final_audio, sr)
    print(f"Final audio saved as {output_filename}")

    # 음량 퍼센트 설정 및 데시벨로 변환
    volume_percent = 20
    volume_db = 20 * math.log10(volume_percent / 100)  # 데시벨 계산

    # Step 3: Adjust volume of the generated audio file
    adjusted_file_path = adjust_volume(output_filename, volume_db)
    if not adjusted_file_path:
        print("Failed to adjust volume, exiting.")
        exit()

    # Step 4: Combine the adjusted audio with other audio files
    # 합칠 파일 목록 설정
    file_paths = ['vocals.wav', 'drums.wav', adjusted_file_path, 'bass.wav']
    combine_audio_files(file_paths, output_path=f"Combine_{selected_model}_output.wav")