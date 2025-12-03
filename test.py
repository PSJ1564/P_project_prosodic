import os
import json
import time
from prosody_analysis import ProsodyAnalyzerLight

# ==========================================
# [설정] 테스트할 파일 경로
# ==========================================
# 테스트할 파일명을 입력하세요. (training_data 폴더 안의 파일 권장)
TARGET_FILE = "test4.wav" 
DATA_FOLDER = "training_data"

def find_any_audio_file():
    if os.path.exists(DATA_FOLDER):
        for f in os.listdir(DATA_FOLDER):
            if f.lower().endswith(('.wav', '.mp3', '.mp4', '.m4a')):
                return os.path.join(DATA_FOLDER, f)
    return None

def run_test():
    # 1. 테스트 파일 확인
    file_path = TARGET_FILE
    if not os.path.exists(file_path):
        print(f"⚠️ 지정된 파일({TARGET_FILE})이 없습니다.")
        found_file = find_any_audio_file()
        if found_file:
            file_path = found_file
            print(f"🔍 대체 파일로 테스트합니다: {file_path}")
        else:
            print("❌ 테스트할 오디오/비디오 파일을 찾을 수 없습니다.")
            return

    # 2. 모듈 초기화
    print("\n🚀 [Step 1] 분석 모듈 초기화...")
    analyzer = ProsodyAnalyzerLight()
    
    # 3. 분석 실행 및 Latency 측정 (Test Code 레벨에서 측정)
    print("🚀 [Step 2] 분석 시작...")
    start_time = time.time()
    
    try:
        result = analyzer.analyze(file_path)
    except Exception as e:
        print(f"❌ 모듈 실행 중 에러 발생: {e}")
        return

    end_time = time.time()
    latency = end_time - start_time  # 소요 시간 계산

    if result is None:
        print("❌ 분석 실패 (None 반환 - 파일 경로 또는 FFmpeg 확인 필요)")
        return

    # 4. 전체 결과 출력 (JSON)
    # 가독성을 위해 Latency 정보를 테스트 결과 딕셔너리에 잠시 추가해서 출력
    display_result = result.copy()
    display_result['test_latency_sec'] = round(latency, 4)

    print("\n" + "="*60)
    print("📦 [Step 3] Full Return Dictionary (JSON Format)")
    print("="*60)
    print(json.dumps(display_result, indent=4, ensure_ascii=False))

    # 5. 요약 결과 출력 (Summary)
    # 데이터 구조 변경 반영: scores['Overall']['score'] -> scores['Overall']
    meta = result['metadata']
    scores = result['scores']
    raw = result['raw_features']

    print("\n" + "="*60)
    print("📝 [Step 4] Analysis Summary")
    print("="*60)
    print(f"📂 분석 파일      : {os.path.basename(file_path)}")
    print(f"⏱️ 분석 소요 시간 : {latency:.4f} sec")
    print("-" * 60)
    print(f"👤 감지된 성별    : {meta['gender']}")
    print(f"🎼 평균 피치      : {meta['mean_pitch']} Hz")
    print("-" * 60)
    # [수정됨] 점수 접근 방식 변경 (Scalar 값 직접 접근)
    print(f"🏆 종합 점수 (Overall)        : {scores['Overall']:.4f}")
    print(f"💼 고용 추천 (Hiring Score)   : {scores['RecommendedHiring']:.4f}")
    print("-" * 60)
    print("📊 핵심 지표 (Raw Features):")
    print(f"   • F1 대역폭 (명료도)       : {raw['avgBand1']:.2f} Hz")
    print(f"   • 평균 성량 (Energy)       : {raw['intensityMean']:.2f} dB")
    print(f"   • 무성음 비율 (떨림/쉼)    : {raw['percentUnvoiced']*100:.1f} %")
    print(f"   • 평균 침묵 길이 (Pause)   : {raw['avgDurPause']:.3f} sec")
    print("="*60)

if __name__ == "__main__":
    run_test()