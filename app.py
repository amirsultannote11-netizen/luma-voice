import os
import io
import hashlib
from flask import Flask, request, Response
from gtts import gTTS
import speech_recognition as sr

app = Flask(__name__)

# کیش فولڈر
CACHE_DIR = "/tmp/luma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route('/')
def home():
    return "Luma Voice Server (Google TTS Simple Version) is Running!"

# ─── 1. Speech to Text (STT) ───
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
    except Exception as e:
        return f"STT Error: {str(e)}", 500

# ─── 2. Google TTS (gTTS) ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur')

    if not user_text:
        return "Error: No text provided", 400

    # کیش چیک کریں
    cache_key = hashlib.md5((user_text + user_lang).encode('utf-8')).hexdigest()
    cached_mp3 = os.path.join(CACHE_DIR, cache_key + ".mp3")

    if os.path.exists(cached_mp3) and os.path.getsize(cached_mp3) > 0:
        with open(cached_mp3, "rb") as f:
            return Response(f.read(), mimetype="audio/mpeg")

    try:
        # Google TTS سے آواز بنانا
        tts_lang = 'ur' if user_lang == 'ur' else 'en'
        tts = gTTS(text=user_text, lang=tts_lang, slow=False)
        
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        audio_bytes = audio_io.read()

        # کیش میں محفوظ کریں
        with open(cached_mp3, "wb") as f:
            f.write(audio_bytes)

        print(f"[gTTS] Audio sent for: {user_text[:20]}")
        return Response(audio_bytes, mimetype="audio/mpeg")

    except Exception as e:
        print(f"gTTS Error: {str(e)}")
        return f"TTS Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
