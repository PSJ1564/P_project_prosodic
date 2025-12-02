import os
import numpy as np
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment

# FFmpeg setup
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
        # Weight Table from Request
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

        # Baseline Statistics (Male) - Needs real data update
        self.baseline_male = {
            'avgBand1': {'mean': 374.8, 'std': 81.9}, 'avgBand2': {'mean': 494.6, 'std': 80.4},
            'intensityMax': {'mean': 83.9, 'std': 2.2}, 'intensityMean': {'mean': 44.6, 'std': 8.9},
            'diffIntMaxMin': {'mean': 38.9, 'std': 2.2}, 'intensitySD': {'mean': 55.9, 'std': 14.4},
            'mean pitch': {'mean': 131.5, 'std': 14.7}, 'max pitch': {'mean': 200.0, 'std': 30.0}, # Dummy
            'F1STD': {'mean': 401.0, 'std': 59.2}, 'f3STD': {'mean': 466.7, 'std': 59.5},
            'f3meanf1': {'mean': 5.0, 'std': 1.0}, 'f2meanf1': {'mean': 3.0, 'std': 0.8}, # Dummy
            'f2STDf1': {'mean': 0.5, 'std': 0.2}, 'fmean3': {'mean': 2917.3, 'std': 109.8},
            'percentUnvoiced': {'mean': 0.51, 'std': 0.07}, 'avgDurPause': {'mean': 0.85, 'std': 0.18},
            'maxDurPause': {'mean': 2.54, 'std': 0.86}, 'PercentBreaks': {'mean': 0.41, 'std': 0.09},
            'shimmer': {'mean': 0.03, 'std': 0.01} # Dummy
        }
        
        # Baseline Statistics (Female) - Needs real data update
        self.baseline_female = {
            'avgBand1': {'mean': 338.1, 'std': 53.5}, 'avgBand2': {'mean': 447.2, 'std': 70.0},
            'intensityMax': {'mean': 84.9, 'std': 2.5}, 'intensityMean': {'mean': 49.2, 'std': 5.3},
            'diffIntMaxMin': {'mean': 38.9, 'std': 2.5}, 'intensitySD': {'mean': 53.0, 'std': 10.5},
            'mean pitch': {'mean': 219.1, 'std': 20.7}, 'max pitch': {'mean': 350.0, 'std': 40.0}, # Dummy
            'F1STD': {'mean': 367.7, 'std': 49.5}, 'f3STD': {'mean': 457.9, 'std': 74.9},
            'f3meanf1': {'mean': 5.5, 'std': 1.2}, 'f2meanf1': {'mean': 3.5, 'std': 0.9}, # Dummy
            'f2STDf1': {'mean': 0.6, 'std': 0.2}, 'fmean3': {'mean': 3018.9, 'std': 101.4},
            'percentUnvoiced': {'mean': 0.42, 'std': 0.07}, 'avgDurPause': {'mean': 0.77, 'std': 0.19},
            'maxDurPause': {'mean': 2.65, 'std': 1.26}, 'PercentBreaks': {'mean': 0.34, 'std': 0.09},
            'shimmer': {'mean': 0.03, 'std': 0.01} # Dummy
        }

    def _convert_to_wav(self, input_path):
        """Media conversion to 16kHz wav"""
        ext = os.path.splitext(input_path)[1].lower()
        wav_path = "temp_analysis.wav"
        try:
            if ext == '.mp4':
                video = VideoFileClip(input_path)
                video.audio.write_audiofile(wav_path, logger=None, fps=16000, nbytes=2, codec='pcm_s16le')
            else:
                sound = AudioSegment.from_file(input_path)
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(wav_path, format="wav")
            return wav_path
        except:
            return None

    def _extract_features(self, sound):
        """Feature Extraction (Praat)"""
        duration = sound.get_total_duration()

        # 1. Pitch
        pitch = sound.to_pitch()
        pitch_vals = pitch.selected_array['frequency']
        pitch_vals_nz = pitch_vals[pitch_vals > 0]
        
        mean_pitch = np.mean(pitch_vals_nz) if len(pitch_vals_nz) > 0 else 0
        max_pitch = np.max(pitch_vals_nz) if len(pitch_vals_nz) > 0 else 0

        # 2. Shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        try:
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

        # 5. Pauses
        silence_tier = call(sound, "To TextGrid (silences)", 100, 0.0, -25.0, 0.3, 0.1, "silent", "sounding")
        num_intervals = call(silence_tier, "Get number of intervals", 1)
        pause_durs = []
        total_silence = 0
        
        for i in range(1, num_intervals + 1):
            if call(silence_tier, "Get label of interval", 1, i) == "silent":
                dur = call(silence_tier, "Get end time of interval", 1, i) - call(silence_tier, "Get start time of interval", 1, i)
                pause_durs.append(dur)
                total_silence += dur

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
            "percentUnvoiced": 1.0 - (len(pitch_vals_nz)/len(pitch_vals)) if len(pitch_vals) > 0 else 0,
            "avgDurPause": np.mean(pause_durs) if pause_durs else 0,
            "maxDurPause": np.max(pause_durs) if pause_durs else 0,
            "PercentBreaks": total_silence / duration if duration > 0 else 0,
            "shimmer": shimmer
        }

    def analyze(self, file_path):
        """Process: Convert -> Extract -> Normalize -> Score"""
        wav_path = self._convert_to_wav(file_path)
        if not wav_path: return None

        try:
            sound = parselmouth.Sound(wav_path)
            raw_features = self._extract_features(sound)
        except Exception as e:
            print(f"Extraction Failed: {e}")
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
            stat = baseline.get(key, {'mean': 0, 'std': 1})
            mu, sigma = stat['mean'], stat['std']
            normalized[key] = (val - mu) / sigma if sigma != 0 else 0

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
    res = analyzer.analyze("aa.wav") # Replace with actual file
    if res:
        print(f"Gender: {res['metadata']['gender']}")
        for cat, val in res['scores'].items():
            print(f"{cat}: {val['score']}")