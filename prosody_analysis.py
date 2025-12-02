import os
import subprocess
import numpy as np
import parselmouth
from parselmouth.praat import call

class ProsodyAnalyzerLight:
    def __init__(self):
        # 1. 가중치 (Top 4 Feature 전용)
        self.weights = {
            "Overall": {
                "avgBand1": -0.120, "intensityMean": 0.065,
                "percentUnvoiced": -0.076, "avgDurPause": -0.090
            },
            "RecommendedHiring": {
                "avgBand1": -0.132, "intensityMean": 0.086,
                "percentUnvoiced": -0.111, "avgDurPause": -0.094
            }
        }

        # 2. 기준 분포 (Top 4 Feature + Pitch)
        self.baseline_male = {
            'avgBand1': {'mean': 374.8, 'std': 81.9},
            'intensityMean': {'mean': 44.6, 'std': 8.9},
            'percentUnvoiced': {'mean': 0.51, 'std': 0.07},
            'avgDurPause': {'mean': 0.85, 'std': 0.18},
            'mean pitch': {'mean': 131.5, 'std': 14.7}
        }
        
        self.baseline_female = {
            'avgBand1': {'mean': 338.1, 'std': 53.5},
            'intensityMean': {'mean': 49.2, 'std': 5.3},
            'percentUnvoiced': {'mean': 0.42, 'std': 0.07},
            'avgDurPause': {'mean': 0.77, 'std': 0.19},
            'mean pitch': {'mean': 219.1, 'std': 20.7}
        }

    def _convert_to_wav_ffmpeg(self, input_path):
        wav_path = "temp_light_analysis.wav"
        if os.path.exists(wav_path): os.remove(wav_path)
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn", "-f", "wav", wav_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return wav_path
        except:
            return None

    def _extract_features_light(self, sound):
        duration = sound.get_total_duration()

        # [1] Pitch (성별 & 무성음용)
        # Time step을 0.02로 늘려 속도 향상
        pitch = sound.to_pitch(time_step=0.02, pitch_floor=75.0, pitch_ceiling=600.0)
        pitch_vals = pitch.selected_array['frequency']
        
        # 유효 피치 (100Hz 이상)
        pitch_vals_valid = pitch_vals[pitch_vals >= 100.0]
        mean_pitch = np.mean(pitch_vals_valid) if len(pitch_vals_valid) > 0 else 0
        
        # Unvoiced Rate
        total_frames = len(pitch_vals)
        valid_frames = len(pitch_vals_valid)
        percent_unvoiced = 1.0 - (valid_frames / total_frames) if total_frames > 0 else 0

        # [2] Intensity
        intensity = sound.to_intensity()
        mean_int = np.mean(intensity.values[0])

        # [3] Formant (F1용) - Max Freq 3000Hz로 제한하여 속도 향상
        formant = sound.to_formant_burg(time_step=0.02, max_number_of_formants=3, maximum_formant=3000)
        
        f1_bw_list = []
        times = np.arange(0, duration, 0.02)
        
        for t in times:
            bw = formant.get_bandwidth_at_time(1, t)
            if not np.isnan(bw):
                f1_bw_list.append(bw)
        
        avg_band1 = np.mean(f1_bw_list) if f1_bw_list else 0

        # [4] Pause (TextGrid)
        # 조건: Min Pitch 100Hz, Silence Threshold -25dB, Min Interval 0.5s
        silence_tier = call(sound, "To TextGrid (silences)", 100.0, 0.0, -25.0, 0.5, 0.1, "silent", "sounding")
        num_intervals = call(silence_tier, "Get number of intervals", 1)
        
        pause_durs = []
        for i in range(1, num_intervals + 1):
            if call(silence_tier, "Get label of interval", 1, i) == "silent":
                dur = call(silence_tier, "Get end time of interval", 1, i) - call(silence_tier, "Get start time of interval", 1, i)
                pause_durs.append(dur)
        
        avg_dur_pause = np.mean(pause_durs) if pause_durs else 0

        return {
            "mean pitch": mean_pitch,
            "avgBand1": avg_band1,
            "intensityMean": mean_int,
            "percentUnvoiced": percent_unvoiced,
            "avgDurPause": avg_dur_pause
        }

    def analyze(self, file_path):
        wav_path = self._convert_to_wav_ffmpeg(file_path)
        if not wav_path: return None

        try:
            sound = parselmouth.Sound(wav_path)
            raw_features = self._extract_features_light(sound)
        except Exception as e:
            if os.path.exists(wav_path): os.remove(wav_path)
            return None
        finally:
            if os.path.exists(wav_path): os.remove(wav_path)

        # Gender Detection
        if raw_features["mean pitch"] < 175.0:
            baseline = self.baseline_male
            gender = "Male"
        else:
            baseline = self.baseline_female
            gender = "Female"

        # Normalization
        normalized = {}
        for key, val in raw_features.items():
            if key in baseline:
                stat = baseline[key]
                mu, sigma = stat['mean'], stat['std']
                if sigma == 0: sigma = 1
                normalized[key] = (val - mu) / sigma

        # Scoring
        scores = {}
        for category, weights in self.weights.items():
            total = 0
            for feat, weight in weights.items():
                z_val = normalized.get(feat, 0)
                total += z_val * weight
            scores[category] = round(total, 4)

        return {
            "metadata": {
                "gender": gender, 
                "mean_pitch": round(raw_features["mean pitch"], 2)
            },
            "scores": scores,
            "raw_features": raw_features
        }