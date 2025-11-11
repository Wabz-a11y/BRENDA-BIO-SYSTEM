# =========================
# voice.py – Brenda Voice Core
# =========================
import os
import pickle
import threading
import numpy as np
import librosa
import speech_recognition as sr
from io import BytesIO
from typing import Optional, Dict, Any
from pydub import AudioSegment

class BrendaVoice:
    def __init__(self, model_path: str = 'voice_model.pkl'):
        self.model_path = model_path
        self.profiles: Dict[str, np.ndarray] = {}
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.lock = threading.Lock()

        # Load existing profiles
        self._load_profiles()

        # Background listening
        self.is_listening = False
        self.latest_audio: Optional[BytesIO] = None
        self.listening_thread: Optional[threading.Thread] = None

    def _load_profiles(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.profiles = pickle.load(f)
                print(f"[Voice] Loaded {len(self.profiles)} voice profile(s).")
            except Exception as e:
                print(f"[Voice] Failed to load model: {e}")

    def _save_profiles(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.profiles, f)
            print(f"[Voice] Saved {len(self.profiles)} profile(s).")
        except Exception as e:
            print(f"[Voice] Save failed: {e}")

    @staticmethod
    def extract_mfcc(audio_data: bytes, sr: int = 16000) -> np.ndarray:
        """Extract mean MFCC from raw WAV bytes."""
        try:
            y, _ = librosa.load(BytesIO(audio_data), sr=sr, duration=5.0)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return np.mean(mfcc.T, axis=0)
        except Exception as e:
            raise ValueError(f"MFCC extraction failed: {e}")

    def enroll(self, user_name: str, audio_data: bytes) -> Dict[str, Any]:
        """
        Enroll a new speaker from raw WAV bytes.
        """
        if user_name in self.profiles:
            return {"status": "error", "message": f"User '{user_name}' already exists"}

        try:
            mfcc = self.extract_mfcc(audio_data)
            with self.lock:
                self.profiles[user_name] = mfcc
                self._save_profiles()
            return {"status": "success", "user": user_name, "message": "Enrolled"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def identify(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Identify speaker from raw WAV bytes.
        """
        if not self.profiles:
            return {"status": "error", "message": "No profiles enrolled"}

        try:
            query_mfcc = self.extract_mfcc(audio_data)
            best_user = None
            best_dist = float('inf')
            scores = {}

            with self.lock:
                for user, profile in self.profiles.items():
                    dist = np.linalg.norm(profile - query_mfcc)
                    scores[user] = round(float(dist), 4)
                    if dist < best_dist:
                        best_dist = dist
                        best_user = user

            # Simple threshold: reject if too far
            threshold = 15.0  # Tune based on your data
            if best_dist > threshold:
                return {
                    "status": "unknown",
                    "closest": best_user,
                    "distance": best_dist,
                    "scores": scores
                }

            return {
                "status": "identified",
                "user": best_user,
                "confidence": round(1.0 / (1.0 + best_dist), 3),
                "distance": best_dist,
                "scores": scores
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ————————————————————————
    # Live microphone streaming
    # ————————————————————————
    def start_listening(self, duration: float = 5.0):
        """Start background recording."""
        if self.is_listening:
            return {"status": "error", "message": "Already listening"}

        self.is_listening = True
        self.latest_audio = None

        def _listen():
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                print("[Voice] Listening... (speak now)")
                audio = self.recognizer.listen(source, phrase_time_limit=duration)
                wav_data = audio.get_wav_data()
                self.latest_audio = BytesIO(wav_data)
                print(f"[Voice] Captured {len(wav_data)} bytes")

        self.listening_thread = threading.Thread(target=_listen, daemon=True)
        self.listening_thread.start()
        return {"status": "listening"}

    def stop_listening(self) -> Optional[bytes]:
        """Stop and return latest audio."""
        if not self.is_listening:
            return None

        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=6.0)

        if self.latest_audio:
            audio_bytes = self.latest_audio.getvalue()
            self.latest_audio = None
            return audio_bytes
        return None

    def record_and_identify(self, duration: float = 5.0) -> Dict[str, Any]:
        """One-shot: record → identify."""
        self.start_listening(duration)
        audio = self.stop_listening()
        if not audio:
            return {"status": "error", "message": "No audio captured"}
        return self.identify(audio)


# =========================
# Flask Integration Example
# =========================
if __name__ == "__main__":
    from flask import Flask, request, jsonify
    import wave

    app = Flask(__name__)
    voice = BrendaVoice()

    @app.route('/voice/enroll', methods=['POST'])
    def enroll_endpoint():
        user = request.form.get('user')
        if not user:
            return jsonify({"error": "Missing 'user'"}), 400
        if 'audio' not in request.files:
            return jsonify({"error": "Missing audio file"}), 400

        file = request.files['audio']
        audio_bytes = file.read()

        # Ensure WAV format
        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()
        except:
            return jsonify({"error": "Invalid audio format"}), 400

        result = voice.enroll(user, wav_bytes)
        return jsonify(result)

    @app.route('/voice/identify', methods=['POST'])
    def identify_endpoint():
        if 'audio' not in request.files:
            return jsonify({"error": "Missing audio file"}), 400

        file = request.files['audio']
        audio_bytes = file.read()

        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()
        except:
            return jsonify({"error": "Invalid audio format"}), 400

        result = voice.identify(wav_bytes)
        return jsonify(result)

    @app.route('/voice/record', methods=['POST'])
    def record_endpoint():
        duration = float(request.json.get('duration', 5))
        result = voice.record_and_identify(duration)
        return jsonify(result)

    print("Brenda Voice API running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)