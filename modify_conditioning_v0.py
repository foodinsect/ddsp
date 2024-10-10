from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
import ddsp
import numpy as np
import tensorflow as tf

def modify_conditioning(audio_features, dataset_stats, adjust=True, quiet=20.0, autotune=0.0):
    # EagerTensor 객체를 numpy 배열로 변환 후 작업 수행
    audio_features_mod = {}
    for k, v in audio_features.items():
        if isinstance(v, tf.Tensor):
            audio_features_mod[k] = v.numpy()  # EagerTensor -> NumPy 배열로 변환
        elif isinstance(v, np.ndarray):
            audio_features_mod[k] = v.copy()  # NumPy 배열 복사
        else:
            audio_features_mod[k] = v  # 그 외는 그대로 사용

    # Dataset statistics를 기반으로 특징 조정
    if adjust and dataset_stats is not None:
        try:
            mask_on, note_on_value = detect_notes(audio_features_mod['loudness_db'], audio_features_mod['f0_confidence'])

            if np.any(mask_on):
                target_mean_pitch = dataset_stats['mean_pitch']
                pitch = ddsp.core.hz_to_midi(audio_features_mod['f0_hz'])  # ddsp.core를 이용한 pitch 변환
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

                if autotune:
                    f0_midi = ddsp.core.hz_to_midi(audio_features_mod['f0_hz'])
                    tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
                    f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
                    audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
        except Exception as e:
            print(f"Error during conditioning adjustment: {e}")
            return None  # 오류 발생 시 None 반환

    # Pitch와 loudness를 조정 (기본값: 변경 없음)
    audio_features_mod['loudness_db'] += 0.0  # 기본값으로 loudness_shift 0.0 추가
    audio_features_mod['f0_hz'] *= 1.00  # 기본값으로 pitch_shift 1.0 추가
    audio_features_mod['f0_hz'] = np.clip(audio_features_mod['f0_hz'], 0.0, 5000.0)

    return audio_features_mod
