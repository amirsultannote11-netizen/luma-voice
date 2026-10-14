import os
import subprocess
import urllib.request
from flask import Flask, request, Response
from gtts import gTTS
import speech_recognition as sr
import io

app = Flask(__name__)

# ─── 1. اردو فیمیل ماڈل (Urdu Female: Aegis) ───
URDU_MODEL_ONNX = "ur_PK-aegis_female-medium.onnx"
URDU_MODEL_JSON = "ur_PK-aegis_female-medium.onnx.json"
URDU_ONNX_URL = "https://huggingface.co/mahwizzzz/piper-voice-ur-aegis-female/resolve/main/ur_PK-aegis_female-medium.onnx"
URDU_JSON_URL = "https://huggingface.co/mahwizzzz/piper-voice-ur-aegis-female/resolve/main/ur_PK-aegis_female-medium.onnx.json"

# ─── 2. انگلش فیمیل ماڈل (English Female: Amy) ───
ENG_MODEL_ONNX = "en_US-amy-medium.onnx"
ENG_MODEL_JSON = "en_US-amy-medium.onnx.json"
ENG_ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
ENG_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

def download_file(url, target_path):
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())

def download_models_if_missing():
    try:
        if not os.path.exists(URDU_MODEL_ONNX) or os.path.getsize(URDU_MODEL_ONNX) < 1000000:
            print("Downloading Female Urdu Piper Model (Aegis)...")
            download_file(URDU_ONNX_URL, URDU_MODEL_ONNX)
            download_file(URDU_JSON_URL, URDU_MODEL_JSON)
            print("Urdu Female Model Downloaded!")

        if not os.path.exists(ENG_MODEL_ONNX) or os.path.getsize(ENG_MODEL_ONNX) < 1000000:
            print("Downloading Female English Piper Model (Amy)...")
            download_file(ENG_ONNX_URL, ENG_MODEL_ONNX)
            download_file(ENG_JSON_URL, ENG_MODEL_JSON)
            print("English Female Model Downloaded!")

    except Exception as e:
        print(f"Model Download Warning: {str(e)}")

download_models_if_missing()

@app.route('/')
def home():
    return "Luma Hybrid Server (Urdu & English Female Piper + Google Backup) is Running!"

# ─── Speech To Text (STT) ───
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

# ─── Text To Speech (TTS) ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur').lower()
    user_speed = request.args.get('speed', '1.0')

    if not user_text:
        return "Error: No text provided", 400

    selected_model = ENG_MODEL_ONNX if user_lang == 'en' else URDU_MODEL_ONNX

    if not os.path.exists(selected_model):
        download_models_if_missing()

    output_wav = "output_raw.wav"
    converted_wav = "output_esp32.wav"

    if os.path.exists(selected_model):
        try:
            # Piper TTS ایگزیکیوشن
            cmd = f'echo "{user_text}" | piper --model {selected_model} --length-scale {user_speed} --output_file {output_wav}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
                # FFmpeg کے ذریعے آڈیو کو ESP32 دوست 16kHz PCM WAV میں کنورٹ کرنا
                ffmpeg_cmd = f'ffmpeg -y -i {output_wav} -ar 16000 -ac 1 -c:a pcm_s16le -f wav {converted_wav}'
                ff_res = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)

                target_file = converted_wav if (ff_res.returncode == 0 and os.path.exists(converted_wav)) else output_wav

                with open(target_file, "rb") as f:
                    audio_data = f.read()
                
                print(f"TTS Success: Piper Female Model ({user_lang.upper()}, Speed: {user_speed})")
                return Response(audio_data, mimetype="audio/wav")

        except Exception as e:
            print(f"Piper error, falling back to Google: {str(e)}")

    # 2️⃣ گوگل بیک اپ (gTTS)
    try:
        print(f"TTS Fallback: Google gTTS (Lang: {user_lang})")
        tts_lang = 'en' if user_lang == 'en' else 'ur'
        tts = gTTS(text=user_text, lang=tts_lang, slow=False)
        
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        return Response(audio_io.read(), mimetype="audio/mpeg")
        
    except Exception as e:
        print(f"Both TTS failed: {str(e)}")
        return f"TTS Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
