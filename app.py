import os
import io
import subprocess
import tempfile
import hashlib
from flask import Flask, request, Response
from gtts import gTTS
import speech_recognition as sr

app = Flask(__name__)

# ─── کیش فولڈر بنائیں (تاکہ بار بار ایک ہی جملہ تیز بنے) ───
CACHE_DIR = "/tmp/luma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route('/')
def home():
    return "Luma Voice Server (Piper + Google TTS Backup) is Running!"

# ─── مائیک کی آواز کو ٹیکسٹ میں بدلنے کا سسٹم (STT) ───
@app.route('/stt', methods=['POST'])
def speech_to_text():
    try:
        audio_data = request.data
        if not audio_data:
            return "Error: No audio data received", 400

        wav_header = io.BytesIO()
        sample_rate = 16000
        bits_per_sample = 16
        channels = 1
        data_len = len(audio_data)

        wav_header.write(b'RIFF')
        wav_header.write((data_len + 36).to_bytes(4, 'little'))
        wav_header.write(b'WAVEfmt ')
        wav_header.write((16).to_bytes(4, 'little'))
        wav_header.write((1).to_bytes(2, 'little'))
        wav_header.write((channels).to_bytes(2, 'little'))
        wav_header.write((sample_rate).to_bytes(4, 'little'))
        wav_header.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
        wav_header.write((channels * bits_per_sample // 8).to_bytes(2, 'little'))
        wav_header.write((bits_per_sample).to_bytes(2, 'little'))
        wav_header.write(b'data')
        wav_header.write((data_len).to_bytes(4, 'little'))
        wav_header.write(audio_data)
        wav_header.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_header) as source:
            audio_file_data = recognizer.record(source)

        recognized_text = recognizer.recognize_google(audio_file_data, language="ur-PK")
        print(f"Server Recognized: {recognized_text}")
        return Response(recognized_text, mimetype="text/plain")

    except sr.UnknownValueError:
        return "معذرت، آواز سمجھ نہیں آئی", 200
    except sr.RequestError as e:
        return f"Google STT Service Error: {str(e)}", 500
    except Exception as e:
        return f"STT Server Error: {str(e)}", 500

# ─── TTS سسٹم: Piper (تیز MP3) + gTTS بیک اپ ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur')

    if not user_text:
        return "Error: No text provided", 400

    # کیش چیک کریں: اگر یہی جملہ پہلے بنا ہوا ہے تو فوراً بھیج دیں
    cache_key = hashlib.md5((user_text + user_lang).encode('utf-8')).hexdigest()
    cached_mp3 = os.path.join(CACHE_DIR, cache_key + ".mp3")

    if os.path.exists(cached_mp3) and os.path.getsize(cached_mp3) > 0:
        print(f"[CACHE HIT] Sending cached MP3 for: {user_text[:30]}")
        with open(cached_mp3, "rb") as f:
            return Response(f.read(), mimetype="audio/mpeg")

    # 1. Piper TTS کے ذریعے WAV بنائیں
    try:
        model_name = "ur_PK-aegis_female-medium.onnx" if user_lang == 'ur' else "en_US-lessac-medium.onnx"

        if os.path.exists(model_name):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                wav_path = wav_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
                mp3_path = mp3_file.name

            # اردو ٹیکسٹ کو صحیح طریقے سے بھیجنے کے لیے subprocess کا استعمال
            try:
                piper_process = subprocess.run(
                    ["piper", "--model", model_name, "--output_file", wav_path],
                    input=user_text.encode('utf-8'),
                    capture_output=True,
                    timeout=30
                )
                print(f"[PIPER] Return code: {piper_process.returncode}")

                if piper_process.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    # WAV کو MP3 میں کنورٹ کریں (سائز 4 گنا کم ہو جائے گا)
                    ffmpeg_result = subprocess.run(
                        ["ffmpeg", "-y", "-i", wav_path, "-b:a", "32k", "-ar", "16000", "-ac", "1", mp3_path],
                        capture_output=True,
                        timeout=30
                    )

                    os.remove(wav_path)

                    if ffmpeg_result.returncode == 0 and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                        with open(mp3_path, "rb") as f:
                            audio_bytes = f.read()

                        # کیش میں محفوظ کریں
                        with open(cached_mp3, "wb") as f:
                            f.write(audio_bytes)

                        os.remove(mp3_path)
                        print(f"[PIPER] MP3 sent, size: {len(audio_bytes)} bytes")
                        return Response(audio_bytes, mimetype="audio/mpeg")
                    else:
                        print(f"[FFMPEG ERROR] {ffmpeg_result.stderr.decode()[:200]}")
            except subprocess.TimeoutExpired:
                print("[PIPER] Timeout! Server is too slow.")

    except Exception as piper_err:
        print(f"Piper TTS Error (Falling back to gTTS): {str(piper_err)}")

    # 2. اگر Piper فیل ہو تو gTTS بیک اپ استعمال کریں
    try:
        tts = gTTS(text=user_text, lang='ur' if user_lang == 'ur' else 'en', slow=False)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        audio_bytes = audio_io.read()

        # کیش میں محفوظ کریں
        with open(cached_mp3, "wb") as f:
            f.write(audio_bytes)

        print(f"[gTTS] MP3 sent, size: {len(audio_bytes)} bytes")
        return Response(audio_bytes, mimetype="audio/mpeg")

    except Exception as e:
        print(f"TTS Backup Error: {str(e)}")
        return f"TTS Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
