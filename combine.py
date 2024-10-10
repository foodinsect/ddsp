from pydub import AudioSegment
import math

# 조정할 음량 퍼센트
volume_percent = 20

# 퍼센트를 데시벨로 변환 (예: 10% 음량 감소 = 약 -20dB)
volume_db = 20 * math.log10(volume_percent / 100)

def adjust_volume(file_path, volume_db):
    """파일의 음량을 조정하여 새로운 파일로 저장합니다."""
    try:
        # 오디오 파일 로드
        audio = AudioSegment.from_file(file_path)

        # 음량을 퍼센트에 맞춰 조절
        adjusted_audio = audio.apply_gain(volume_db)
        adjusted_file_path = f"adjusted_{file_path}"
        adjusted_audio.export(adjusted_file_path, format="wav")
        print(f"{file_path} 파일의 음량이 {volume_percent}%로 조절되었습니다. (약 {volume_db:.2f} dB 감소)")
        print(f"음량이 조정된 파일이 '{adjusted_file_path}'로 저장되었습니다.")
        return adjusted_file_path
    except Exception as e:
        print(f"파일을 로드하거나 음량을 조절하는 중 오류가 발생했습니다: {e}")
        return None

def combine_audio_files(file_paths, output_path="combined_output.wav"):
    """여러 오디오 파일을 하나로 합칩니다."""
    try:
        # 오디오 트랙들을 저장할 리스트
        tracks = [AudioSegment.from_file(file_path) for file_path in file_paths]

        # 첫 번째 트랙을 기준으로 복사본을 만듭니다.
        combined = tracks[0]

        # 첫 번째 트랙 이외의 트랙들을 첫 번째 트랙에 겹칩니다.
        for track in tracks[1:]:
            # 오디오 트랙을 같은 시작점에서 재생되도록 겹칩니다.
            combined = combined.overlay(track)

        # 합쳐진 오디오 파일을 저장합니다.
        combined.export(output_path, format="wav")
        print(f"오디오 파일들이 성공적으로 합쳐져 '{output_path}'로 저장되었습니다.")
    except Exception as e:
        print(f"오디오 파일을 합치는 중 오류가 발생했습니다: {e}")