import os
import subprocess
import numpy as np
import parselmouth
from parselmouth.praat import call

# FFmpeg setup (static_ffmpeg 사용 시)
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip

class ProsodyAnalyzer:
    def __init__(self):
        # Weight Table from Request (All Features)
        self.weights = {
            "Overall": {
                "avgBand1": -0.120, "intensityMean": 0.065, "F1STD": -0.071,
                "f3STD": -0.058, "f3meanf1": 0.073, "f2meanf1": 0.059, "f2STDf1": 0.067,
                "percentUnvoiced": -0.076, "avgDurPause": -0.090, "maxDurPause": -0.076,
                "PercentBreaks": -0.064
            },
            "RecommendedHiring": {
                "avgBand1": -0.132, "avgBand2": -0.082, "intensityMax": 0.124, "intensityMean": 0.086,
                "intensitySD": 0.103, "F1STD": -0.082, "f3meanf1": 0.079,
                "percentUnvoiced": -0.111, "avgDurPause": -0.094, "maxDurPause": -0.074,
                "PercentBreaks": -0.090
            },
            "Excited": {
                "avgBand1": -0.159, "avgBand2": -0.119, "intensityMax": 0.174, "intensityMean": 0.115,
                "diffIntMaxMin": 0.132, "intensitySD": 0.093, "mean pitch": 0.113,
                "F1STD": -0.101, "f3STD": -0.115, "f2STDf1": 0.089,
                "percentUnvoiced": -0.105, "PercentBreaks": -0.100
            },
            "EngagingTone": {
                "avgBand1": -0.171, "intensityMax": 0.094, "intensityMean": 0.146,
                "diffIntMaxMin": 0.151, "intensitySD": 0.061, "max pitch": 0.070,
                "F1STD": -0.103, "f3STD": -0.091, "f3meanf1": 0.095, "f2STDf1": 0.096,
                "percentUnvoiced": -0.078, "PercentBreaks": -0.068
            },
            "Friendly": {
                "avgBand1": -0.070, "intensityMean": 0.090, "diffIntMaxMin": 0.089,
                "mean pitch": 0.136, "F1STD": -0.087, "f3STD": -0.106,
                "f2meanf1": 0.077, "fmean3": 0.075,
                "percentUnvoiced": -0.069, "PercentBreaks": -0.068, "shimmer": -0.073
            }
        }

        # Baseline Statistics (Male)
        
        self.baseline_male = {
            'F1STD': {'mean': 401.0287, 'std': 59.2318},
            'PercentBreaks': {'mean': 0.2478, 'std': 0.1045},
            'avgBand1': {'mean': 374.8122, 'std': 81.9690},
            'avgBand2': {'mean': 494.6753, 'std': 80.4747},
            'avgDurPause': {'mean': 0.9950, 'std': 0.1761},
            'diffIntMaxMin': {'mean': 384.7958, 'std': 1.6064},
            'f2STDf1': {'mean': 1.8720, 'std': 0.4161},
            'f2meanf1': {'mean': 3.1860, 'std': 0.3620},
            'f3STD': {'mean': 466.7625, 'std': 59.5747},
            'f3meanf1': {'mean': 5.4319, 'std': 0.5938},
            'fmean3': {'mean': 2917.3232, 'std': 109.8644},
            'intensityMax': {'mean': 84.7958, 'std': 1.6064},
            'intensityMean': {'mean': 45.4446, 'std': 9.0125},
            'intensitySD': {'mean': 56.0788, 'std': 14.4263},
            'max pitch': {'mean': 439.1715, 'std': 85.7272},
            'maxDurPause': {'mean': 2.1721, 'std': 0.7638},
            'mean pitch': {'mean': 130.4102, 'std': 15.4041},
            'percentUnvoiced': {'mean': 0.3488, 'std': 0.0624},
            'shimmer': {'mean': 0.1176, 'std': 0.0184},
        }
        # Baseline Statistics (Female)
        self.baseline_female = {
            'F1STD': {'mean': 367.7468, 'std': 49.5547},
            'PercentBreaks': {'mean': 0.1835, 'std': 0.1035},
            'avgBand1': {'mean': 338.1834, 'std': 53.5852},
            'avgBand2': {'mean': 447.2607, 'std': 70.0854},
            'avgDurPause': {'mean': 1.0560, 'std': 0.3177},
            'diffIntMaxMin': {'mean': 386.1339, 'std': 1.3178},
            'f2STDf1': {'mean': 1.9504, 'std': 0.4173},
            'f2meanf1': {'mean': 3.2692, 'std': 0.3836},
            'f3STD': {'mean': 457.9493, 'std': 74.9516},
            'f3meanf1': {'mean': 5.6357, 'std': 0.7248},
            'fmean3': {'mean': 3018.9404, 'std': 101.4008},
            'intensityMax': {'mean': 86.1339, 'std': 1.3178},
            'intensityMean': {'mean': 50.3322, 'std': 5.2173},
            'intensitySD': {'mean': 53.2378, 'std': 10.7447},
            'max pitch': {'mean': 461.1523, 'std': 43.5000},
            'maxDurPause': {'mean': 2.3965, 'std': 1.2574},
            'mean pitch': {'mean': 218.5473, 'std': 21.1296},
            'percentUnvoiced': {'mean': 0.2823, 'std': 0.0439},
            'shimmer': {'mean': 0.1040, 'std': 0.0159},
        }

    def _convert_to_wav_ffmpeg(self, input_path):
        """FFmpeg를 이용해 미디어 파일을 16kHz Mono WAV로 변환"""
        wav_path = "temp_all_analysis.wav"
        if os.path.exists(wav_path): os.remove(wav_path)
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn", "-f", "wav", wav_path
        ]
        try:
            # stdout/stderr를 숨겨 로그를 깔끔하게 유지
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return wav_path
        except:
            return None

    def _extract_features(self, sound):
        """Feature Extraction (Praat) - Normalized Sound 사용"""
        duration = sound.get_total_duration()

        # 1. Pitch (50Hz Floor 적용 - 튜닝된 파라미터 사용)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=50.0, pitch_ceiling=500.0)
        pitch_vals = pitch.selected_array['frequency']
        # 유효 피치 (50Hz 이상)
        pitch_vals_valid = pitch_vals[pitch_vals >= 50.0]
        
        mean_pitch = np.mean(pitch_vals_valid) if len(pitch_vals_valid) > 0 else 0
        max_pitch = np.max(pitch_vals_valid) if len(pitch_vals_valid) > 0 else 0

        # 2. Shimmer
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 50, 500)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer = 0

        # 3. Intensity
        intensity = sound.to_intensity()
        int_vals = intensity.values[0]
        mean_int = np.mean(int_vals)
        max_int = np.max(int_vals)
        int_range = max_int - np.min(int_vals)
        int_std = np.std(int_vals)

        # 4. Formants & Ratios
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
        times = np.arange(0, duration, 0.01)

        f1_list, f3_list = [], []
        f1_bw_list, f2_bw_list = [], []
        f2_f1_ratio, f3_f1_ratio = [], []

        for t in times:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            f1_bw = formant.get_bandwidth_at_time(1, t)
            f2_bw = formant.get_bandwidth_at_time(2, t)

            if not np.isnan(f1): f1_list.append(f1)
            if not np.isnan(f3): f3_list.append(f3)
            if not np.isnan(f1_bw): f1_bw_list.append(f1_bw)
            if not np.isnan(f2_bw): f2_bw_list.append(f2_bw)
            
            # Ratio calculation
            if not np.isnan(f1) and f1 > 0:
                if not np.isnan(f2): f2_f1_ratio.append(f2/f1)
                if not np.isnan(f3): f3_f1_ratio.append(f3/f1)

        # 5. Pauses & Unvoiced (튜닝된 로직 적용)
        # Normalization 되었으므로 Silence Threshold -35dB 사용
        silence_tier = call(sound, "To TextGrid (silences)", 50.0, 0.0, -35.0, 0.5, 0.1, "silent", "sounding")
        num_intervals = call(silence_tier, "Get number of intervals", 1)
        
        pause_durs = []
        total_silence_dur = 0
        
        for i in range(1, num_intervals + 1):
            label = call(silence_tier, "Get label of interval", 1, i)
            if label == "silent":
                start = call(silence_tier, "Get start time of interval", 1, i)
                end = call(silence_tier, "Get end time of interval", 1, i)
                dur = end - start
                pause_durs.append(dur)
                total_silence_dur += dur

        # Unvoiced Rate Correction (Silence 제외)
        speaking_duration = duration - total_silence_dur
        voiced_duration = len(pitch_vals_valid) * 0.01 # Time step 0.01
        
        if speaking_duration > 0:
            unvoiced_duration = max(0, speaking_duration - voiced_duration)
            percent_unvoiced = unvoiced_duration / speaking_duration
        else:
            percent_unvoiced = 0
        
        percent_unvoiced = min(1.0, percent_unvoiced)

        # Return dict with keys matching the weight table
        return {
            "avgBand1": np.mean(f1_bw_list) if f1_bw_list else 0,
            "avgBand2": np.mean(f2_bw_list) if f2_bw_list else 0,
            "intensityMax": max_int,
            "intensityMean": mean_int,
            "diffIntMaxMin": int_range,
            "intensitySD": int_std,
            "mean pitch": mean_pitch,
            "max pitch": max_pitch,
            "F1STD": np.std(f1_list) if f1_list else 0,
            "f3STD": np.std(f3_list) if f3_list else 0,
            "f3meanf1": np.mean(f3_f1_ratio) if f3_f1_ratio else 0,
            "f2meanf1": np.mean(f2_f1_ratio) if f2_f1_ratio else 0,
            "f2STDf1": np.std(f2_f1_ratio) if f2_f1_ratio else 0,
            "fmean3": np.mean(f3_list) if f3_list else 0,
            "percentUnvoiced": percent_unvoiced,
            "avgDurPause": np.mean(pause_durs) if pause_durs else 0,
            "maxDurPause": np.max(pause_durs) if pause_durs else 0,
            "PercentBreaks": total_silence_dur / duration if duration > 0 else 0,
            "shimmer": shimmer
        }

    def analyze(self, file_path):
        """Process: Convert -> Peak Norm -> Extract -> Normalize -> Score"""
        wav_path = self._convert_to_wav_ffmpeg(file_path)
        if not wav_path: return None

        try:
            sound = parselmouth.Sound(wav_path)
            
            # ========================================================
            # [Normalization] -1dB Peak Normalization 적용
            # ========================================================
            sound.scale_peak(0.89125)
            
            raw_features = self._extract_features(sound)
            
        except Exception as e:
            print(f"Extraction Failed: {e}")
            if os.path.exists(wav_path): os.remove(wav_path)
            return None
        finally:
            if os.path.exists(wav_path): os.remove(wav_path)

        # Gender Detection
        pitch_val = raw_features["mean pitch"]
        if pitch_val < 175.0:
            baseline = self.baseline_male
            gender = "Male"
        else:
            baseline = self.baseline_female
            gender = "Female"

        # Z-Score Normalization
        normalized = {}
        for key, val in raw_features.items():
            if key in baseline:
                stat = baseline[key]
                mu, sigma = stat['mean'], stat['std']
                if sigma == 0: sigma = 1
                normalized[key] = (val - mu) / sigma

        # Scoring Logic
        scores = {}
        for category, weights in self.weights.items():
            total = 0
            details = {}
            for feat, weight in weights.items():
                if weight == 0: continue
                z_val = normalized.get(feat, 0)
                contrib = z_val * weight
                total += contrib
                details[feat] = round(contrib, 4)
            scores[category] = {"score": round(total, 4), "details": details}

        return {
            "metadata": {"gender": gender, "mean_pitch": round(pitch_val, 2)},
            "raw_features": raw_features,
            "scores": scores
        }

if __name__ == "__main__":
    analyzer = ProsodyAnalyzer()
    
    # 현재 폴더에 있는 첫 번째 wav/mp4 파일로 테스트
    test_files = [f for f in os.listdir('.') if f.lower().endswith(('.wav', '.mp4'))]
    if test_files:
        print(f"Testing with: {test_files[0]}")
        res = analyzer.analyze(test_files[0])
        if res:
            print(f"Gender: {res['metadata']['gender']}")
            print("Scores:")
            for cat, val in res['scores'].items():
                print(f"  {cat}: {val['score']}")
    else:
        print("No test file found.")