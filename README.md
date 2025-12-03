Prosody Analysis Module (AI Interview Coach)

이 저장소는 면접 영상 및 음성 데이터를 분석하여 발화자의음성 특징(Prosody)을 추출하고,
면접 역량 점수(Overall, Hiring)를 산출하는 Python 모듈입니다.

Praat(Parselmouth) 엔진을 기반으로 하며, 실시간 서비스에 최적화된 4대 핵심 Feature[F1대역폭, 평균멈춤길이, 무성음비율, 평균강도] 분석을 기본으로 제공합니다.

📂 파일 구조 (File Structure)
Plaintext

P_prosody/

├── prosody_analysis.py             # [메인] 고속 분석 모듈 (Core 4 Features)

├── prosody_analysis_all_feature.py # [연구용] 정밀 분석 모듈 (All Features)

├── test.py                         # 모듈 실행 예시

└── requirements.txt                # 의존성 패키지 목록

🛠️ 설치 및 환경 설정 (Installation)

1. Python 라이브러리 설치
프로젝트 루트에서 다음 명령어를 실행하여 필요한 패키지를 설치합니다.
Bash
pip install -r requirements.txt

3. FFmpeg 설치 (필수 ⭐)
이 모듈은 미디어 변환을 위해 시스템의 FFmpeg를 직접 호출합니다. 따라서 OS에 FFmpeg가 설치되어 있고, 환경 변수(PATH)에 등록되어 있어야 합니다.

Windows: ffmpeg.org에서 다운로드 후 bin 폴더를 시스템 환경 변수 Path에 추가.

Mac: brew install ffmpeg

Linux: sudo apt install ffmpeg

확인 방법: 터미널에 ffmpeg -version을 입력했을 때 버전 정보가 나와야 합니다.


🚀 사용 방법 (Usage)
Python

# main.py 예시
from prosody_analysis import ProsodyAnalyzerLight

# 1. 분석기 초기화
analyzer = ProsodyAnalyzerLight()

# 2. 파일 분석 (영상 또는 음성 파일 경로)
input_file = "user_upload/interview.mp4"
result = analyzer.analyze(input_file)

if result:

    print(f"성별: {result['metadata']['gender']}")
    print(f"종합 점수: {result['scores']['Overall']}")
    print(f"고용 추천 점수: {result['scores']['RecommendedHiring']}")
else:

    print("분석 실패 (파일 손상 또는 FFmpeg 오류)")
    
# all feature 분석이 필요한 경우
모든 음향 지표(Shimmer, Jitter, Formant Ratio 등)가 필요한 경우 prosody_analysis_all_feature 모듈을 사용합니다.

Python

from prosody_analysis_all_feature import ProsodyAnalyzer

analyzer = ProsodyAnalyzer()

result = analyzer.analyze("interview.mp4")
사용법은 위와 동일

# analyze메서드 반환형태 

analyze() 함수는 다음과 같은 Dictionary 형태의 데이터를 반환합니다.

Type : dict(Dictionary)
json

{

          "metadata": {
          
                    "gender": "Male",           // 자동 감지된 성별 ("Male" | "Female")
                    
                    "mean_pitch": 131.5         // 성별 감지 기준이 된 평균 피치 (Hz)
                    
          },
          
          "scores": {
          
                    "Overall": {
                    
                      "score": 0.85             // 종합 점수 (Z-Score 기반, 양수일수록 좋음)
                      
                    },
                    
                    "RecommendedHiring": {
                    
                      "score": 1.20             // 고용 추천 점수
                      
                    }
            
          },
          
          "raw_features": {             // [참고용] 실제 측정된 4대 핵심 지표 값
          
                    "mean pitch": 131.5,        // 평균 높낮이 (Hz)
                    
                    "avgBand1": 380.2,          // F1 대역폭 (Hz) - 목소리 명료도/공명
                    
                    "intensityMean": 65.4,      // 평균 성량 (dB)
                    
                    "percentUnvoiced": 0.45,    // 무성음 비율 (0.0 ~ 1.0) - 목소리 쉼/떨림
                    
                    "avgDurPause": 0.65         // 평균 휴지기 길이 (sec)
            
          }
  
}

## 분석 로직 상세 (Technical Details)

본 모듈은 일관성 있는 분석 결과를 위해 전처리(Normalization) → 특징 추출 → 성별 감지 → 점수 산출의 파이프라인을 따릅니다.

## 1. 전처리 (Preprocessing)
녹음 환경(마이크 거리, 입력 게인)에 따른 편차를 제거하기 위해 분석 전 오디오를 정규화합니다.
* **Peak Normalization (-1dB):** 입력된 오디오의 최대 진폭을 **-1dB (약 0.89)**로 통일합니다.
* **효과:** * 작게 녹음된 목소리도 표준 크기로 보정됨
  
### 2. 분석 Feature (Light Version)
속도와 정확도의 균형을 위해 다음 4가지 핵심 지표를 추출합니다.
* **F1 Bandwidth:** 목소리의 공명과 명료도를 측정 (Formant 분석).
* **Mean Intensity:** 목소리의 크기 및 에너지 (Normalization 이후의 밀도 측정).
* **Unvoiced Rate:** 순수 발화 시간 중 성대가 울리지 않는(무성음) 구간의 비율.
* **Mean Pause Duration:** 발화 사이의 침묵 길이 (관련 연구 논문에 기반하여 **0.5초 이상** 지속된 침묵만 감지).

### 3. 성별 감지 (Gender Detection)
발화자의 성별에 따라 다른 기준 분포(Baseline)를 적용하기 위해 피치를 분석합니다.
* **기준:** 평균 피치(Mean Pitch) **175Hz**
* **근거:** AI Hub 한국어 음성 데이터 모집단 분포 분석 결과에 기반.
    * `< 175Hz`: **남성(Male)** 기준 데이터 적용
    * `>= 175Hz`: **여성(Female)** 기준 데이터 적용

### 4. 점수 산출 (Scoring)
* **Z-Score Normalization:** 추출된 Raw Data를 성별 기준 분포(Mean, Std)를 이용해 표준화합니다.
* **Weighted Scoring:** 사전에 정의된 가중치(Weights)를 각 지표에 곱하여 최종 **종합 점수(Overall)**와 **고용 추천 점수(Recommended Hiring)**를 도출합니다.

## 기반 연구
본 모듈의 분석 로직과 가중치는 다음 논문을 기반으로 구현 및 튜닝되었습니다.
* Naim, I., Tanveer, M. I., Gildea, D., & Hoque, M. E. (2018). **Automated Analysis and Prediction of Job Interview Performance.** IEEE Transactions on Affective Computing.
