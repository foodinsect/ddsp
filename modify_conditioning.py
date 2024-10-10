import gc
import tensorflow as tf
import numpy as np
import ddsp
from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
from scipy.signal import medfilt

def smooth_pitch(pitch, kernel_size=3):
    """피치 값을 평활화하기 위해 Median Filter 적용"""
    # 디버깅 코드
    print(f"smooth_pitch - pitch type: {type(pitch)}, shape: {pitch.shape}")
    return medfilt(pitch, kernel_size=kernel_size)

def reduce_unwanted_harmonics(audio_features_mod, confidence_threshold=0.5):
    """신뢰도가 낮은 배음을 억제하기 위한 라우드니스 감소"""
    # 디버깅 코드
    print(f"reduce_unwanted_harmonics - audio_features_mod type: {type(audio_features_mod)}, keys: {audio_features_mod.keys()}")
    print(f"reduce_unwanted_harmonics - f0_confidence type: {type(audio_features_mod['f0_confidence'])}, shape: {audio_features_mod['f0_confidence'].shape}")
    mask = audio_features_mod['f0_confidence'] < confidence_threshold
    audio_features_mod['loudness_db'][mask] -= 15.0  # 신뢰도가 낮은 음의 라우드니스 감소

def modify_conditioning(audio_features, dataset_stats, adjust=True, quiet=20.0, autotune=0.0):
    # 메모리 정리 시도
    gc.collect()
    tf.keras.backend.clear_session()

    # EagerTensor 객체를 numpy 배열로 변환 후 작업 수행
    audio_features_mod = {k: (v.numpy() if isinstance(v, tf.Tensor) else v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in audio_features.items()}

    # 디버깅 코드
    print(f"modify_conditioning - audio_features type: {type(audio_features)}, keys: {audio_features.keys()}")
    for k, v in audio_features.items():
        if isinstance(v, tf.Tensor):
            print(f"modify_conditioning - audio_features[{k}] Tensor shape: {v.shape}")
        elif isinstance(v, np.ndarray):
            print(f"modify_conditioning - audio_features[{k}] numpy shape: {v.shape}")
        else:
            print(f"modify_conditioning - audio_features[{k}] type: {type(v)}")

    if adjust and dataset_stats is not None:
        try:
            # detect_notes 함수에서 threshold 인자를 제거
            mask_on, note_on_value = detect_notes(audio_features_mod['loudness_db'], audio_features_mod['f0_confidence'])
            confidence_threshold = 0.5
            mask_on = mask_on & (audio_features_mod['f0_confidence'] > confidence_threshold)
            
            if np.any(mask_on):
                pitch = ddsp.core.hz_to_midi(audio_features_mod['f0_hz'])
                pitch = smooth_pitch(pitch, kernel_size=5)  # Median Filter를 사용한 피치 평활화
                
                target_mean_pitch = dataset_stats['mean_pitch']
                mean_pitch = np.mean(pitch[mask_on])
                p_diff = target_mean_pitch - mean_pitch
                p_diff_octave = p_diff / 12.0
                round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
                p_diff_octave = round_fn(p_diff_octave)
                audio_features_mod['f0_hz'] *= 2.0 ** (p_diff_octave)

                _, loudness_norm = fit_quantile_transform(
                    audio_features_mod['loudness_db'],
                    mask_on,
                    inv_quantile=dataset_stats['quantile_transform']
                )
                mask_off = np.logical_not(mask_on)
                loudness_norm[mask_off] -= quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
                audio_features_mod['loudness_db'] = loudness_norm

                reduce_unwanted_harmonics(audio_features_mod, confidence_threshold=0.6)

                if autotune:
                    f0_midi = ddsp.core.hz_to_midi(audio_features_mod['f0_hz'])
                    tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
                    f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
                    audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
        except Exception as e:
            print(f"Error during conditioning adjustment: {e}")
            return None

    audio_features_mod['loudness_db'] += 0.0
    audio_features_mod['f0_hz'] = smooth_pitch(audio_features_mod['f0_hz'], kernel_size=3)
    audio_features_mod['f0_hz'] *= 0.65
    audio_features_mod['f0_hz'] = np.clip(audio_features_mod['f0_hz'], 0.0, 5000.0)

    return audio_features_mod


