import os
import io
import subprocess
import tempfile
import hashlib
import wave
from flask import Flask, request, Response
import speech_recognition as sr

# ─── Piper Python API لوڈ کریں ───
from piper.voice import PiperVoice

app = Flask(__name__)

CACHE_DIR = "/tmp/luma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── وائسز ایک بار لوڈ کریں ───
urdu_voice = None
english_voice = None

def load_piper_voices():
    global urdu_voice, english_voice
    try:
        if os.path.exists("ur_PK-aegis_female-medium.onnx"):
            urdu_voice = PiperVoice.load("ur_PK-aegis_female-medium.onnx")
            print("[STATUS] Urdu voice loaded")
    except Exception as e:
        print(f"[ERROR] Urdu voice: {e}")
    try:
        if os.path.exists("en_US-lessac-medium.onnx"):
            english_voice = PiperVoice.load("en_US-lessac-medium.onnx")
            print("[STATUS] English voice loaded")
    except Exception as e:
        print(f"[ERROR] English voice: {e}")

load_piper_voices()


@app.route('/')
def home():
    return "Luma Voice Server (Piper) is Running!"


@app.route('/stt', methods=['POST'])
def speech_to_text():
    try:
        audio_data = request.data
        if not audio_data:
            return "Error: No audio data", 400

        wav_header = io.BytesIO()
        sample_rate = 16000
        bits = 16
        channels = 1
        data_len = len(audio_data)

        wav_header.write(b'RIFF')
        wav_header.write((data_len + 36).to_bytes(4, 'little'))
        wav_header.write(b'WAVEfmt ')
        wav_header.write((16).to_bytes(4, 'little'))
        wav_header.write((1).to_bytes(2, 'little'))
        wav_header.write((channels).to_bytes(2, 'little'))
        wav_header.write((sample_rate).to_bytes(4, 'little'))
        wav_header.write((sample_rate * channels * bits // 8).to_bytes(4, 'little'))
        wav_header.write((channels * bits // 8).to_bytes(2, 'little'))
        wav_header.write((bits).to_bytes(2, 'little'))
        wav_header.write(b'data')
        wav_header.write((data_len).to_bytes(4, 'little'))
        wav_header.write(audio_data)
        wav_header.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_header) as source:
            audio_file_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_file_data, language="ur-PK")
        except sr.UnknownValueError:
            wav_header.seek(0)
            with sr.AudioFile(wav_header) as source:
                audio_file_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_file_data, language="en-US")

        print(f"[STT] {text}")
        return Response(text, mimetype="text/plain")

    except sr.UnknownValueError:
        return "معذرت، آواز سمجھ نہیں آئی", 200
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return f"STT Error: {e}", 500


@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur')

    if not user_text:
        return "Error: No text", 400

    cache_key = hashlib.md5((user_text + user_lang).encode('utf-8')).hexdigest()
    cached_mp3 = os.path.join(CACHE_DIR, cache_key + ".mp3")

    if os.path.exists(cached_mp3) and os.path.getsize(cached_mp3) > 0:
        print(f"[CACHE] {user_text[:30]}")
        with open(cached_mp3, "rb") as f:
            return Response(f.read(), mimetype="audio/mpeg")

    voice = urdu_voice if user_lang == 'ur' else english_voice
    if voice is None:
        return "Voice not loaded", 500

    wav_path = tempfile.mktemp(suffix=".wav")
    mp3_path = tempfile.mktemp(suffix=".mp3")

    try:
        with wave.open(wav_path, "wb") as wav_file:
            voice.synthesize(user_text, wav_file)

        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "32k", "-ar", "16000", "-ac", "1", mp3_path],
            capture_output=True, timeout=30
        )

        if os.path.exists(wav_path):
            os.remove(wav_path)

        if ffmpeg_result.returncode == 0 and os.path.getsize(mp3_path) > 0:
            with open(mp3_path, "rb") as f:
                audio_bytes = f.read()

            with open(cached_mp3, "wb") as f:
                f.write(audio_bytes)

            os.remove(mp3_path)
            print(f"[PIPER] MP3 sent: {len(audio_bytes)} bytes")
            return Response(audio_bytes, mimetype="audio/mpeg")
        else:
            print(f"[FFMPEG ERROR] {ffmpeg_result.stderr.decode()[:300]}")
            return "FFmpeg failed", 500

    except Exception as e:
        print(f"[PIPER ERROR] {e}")
        return f"Piper Error: {e}", 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
