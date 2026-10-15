import os
import io
import urllib.request
from flask import Flask, request, Response
from piper import PiperVoice
from gtts import gTTS

app = Flask(__name__)

# Piper Model Files
MODEL_FILENAME = "ur_PK-aegis_female-medium.onnx"
CONFIG_FILENAME = "ur_PK-aegis_female-medium.onnx.json"

# Working Direct URLs for Urdu Female Voice
MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ur/ur_PK/aegis/medium/ur_PK-aegis_female-medium.onnx"
CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ur/ur_PK/aegis/medium/ur_PK-aegis_female-medium.onnx.json"

def download_file_if_missing(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully!")

piper_voice = None

try:
    # Auto-download working model & json
    download_file_if_missing(MODEL_FILENAME, MODEL_URL)
    download_file_if_missing(CONFIG_FILENAME, CONFIG_URL)

    if os.path.exists(MODEL_FILENAME) and os.path.exists(CONFIG_FILENAME):
        piper_voice = PiperVoice.load(MODEL_FILENAME, config_path=CONFIG_FILENAME)
        print("Piper Urdu Female Voice Loaded Successfully!")
except Exception as e:
    print(f"Error loading Piper Model: {e}")

@app.route('/')
def home():
    return "Luma Voice Server Active with Piper AI!"

@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    if not user_text:
        return "No text provided", 400

    # 1. Try Piper AI TTS
    if piper_voice:
        try:
            wav_io = io.BytesIO()
            piper_voice.synthesize(user_text, wav_io)
            return Response(wav_io.getvalue(), mimetype="audio/wav")
        except Exception as e:
            print(f"Piper Synthesis Error: {e}")

    # 2. Backup Google TTS
    try:
        print("Using Google TTS Backup...")
        tts = gTTS(text=user_text, lang='ur', slow=False)
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        return Response(mp3_io.getvalue(), mimetype="audio/mpeg")
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
