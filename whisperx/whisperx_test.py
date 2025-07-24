import json
import os
from dotenv import load_dotenv
import whisperx

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수에서 토큰 읽기
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN 환경변수가 설정되어 있지 않습니다. .env 파일을 확인하세요.")


def transcribe_with_speakers(audio_path: str, model_size="medium", device="cuda", save_json=False):
    """
    WhisperX를 이용해 음성에서 텍스트 + 화자 분리까지 자동 수행하는 함수

    Parameters:
    - audio_path (str): 오디오 파일 경로
    - hf_token (str): HuggingFace 토큰 (pyannote.audio 용)
    - model_size (str): whisper 모델 사이즈 (base, medium, large 등)
    - device (str): "cuda" or "cpu"

    Returns:
    - List[dict]: 화자, 시작/종료 시간, 텍스트 포함된 세그먼트 리스트
    """
    # 1. WhisperX ASR 모델 로드
    print("[1] Load WhisperX model...")
    model = whisperx.load_model(model_size, device)

    # 2. 오디오 로드 및 텍스트 추출
    print("[2] Transcribe audio...")
    audio = whisperx.load_audio(audio_path)
    asr_result = model.transcribe(audio)

    # 3. 정렬 모델 로딩 및 정밀 타임스탬프 정렬
    print("[3] Align timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=asr_result["language"], device=device)
    asr_result_aligned = whisperx.align(asr_result["segments"], model_a, metadata, audio, device)

    # 4. 화자 분리
    print("[4] Diarization (speaker separation)...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token)
    diarize_segments = diarize_model(audio_path)

    # 5. 텍스트 + 화자 병합
    print("[5] Merging ASR + diarization results...")
    merged = whisperx.merge_text_diarization(asr_result_aligned["segments"], diarize_segments)

    # 6. 결과 출력
    print("\n--- 결과 ---")
    for seg in merged:
        print(f"Speaker {seg['speaker']} [{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}")

    if save_json:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base_name}_whisperx.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 저장 완료: {output_path}")

    return merged


if __name__ == "__main__":
    audio_path = "your_audio_file.wav"
    transcribe_with_speakers(audio_path)