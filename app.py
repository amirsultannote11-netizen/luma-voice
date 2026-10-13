import os
import subprocess
import urllib.request
from flask import Flask, request, Response
from gtts import gTTS
import speech_recognition as sr
import io

app = Flask(__name__)

# ─── پائپر ماڈل (خاتون کی اردو آواز - Female Urdu Voice) ───
MODEL_ONNX = "ur_PK-medium.onnx"
MODEL_JSON = "ur_PK-medium.onnx.json"

# HuggingFace کے 100% ورکنگ ڈائریکٹ لنکس
ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ur/ur_PK/medium/ur_PK-medium.onnx"
JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ur/ur_PK/medium/ur_PK-medium.onnx.json"

def download_file(url, target_path):
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())

def download_model_if_missing():
    try:
        if not os.path.exists(MODEL_ONNX) or os.path.getsize(MODEL_ONNX) < 1000000:
            print("Female Piper ONNX model downloading...")
            download_file(ONNX_URL, MODEL_ONNX)
            print("ONNX model downloaded successfully!")
            
        if not os.path.exists(MODEL_JSON):
            print("Female Piper JSON config downloading...")
            download_file(JSON_URL, MODEL_JSON)
            print("JSON config downloaded successfully!")
    except Exception as e:
        print(f"Model download error: {str(e)}")

# سرور سٹارٹ ہوتے ہی ماڈل ڈاؤن لوڈ ہوگا
download_model_if_missing()


@app.route('/')
def home():
    return "Luma Hybrid Voice Server (Female Piper + Backup) is Running!"

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

    # 1️⃣ Piper Female Voice (Urdu)
    if user_lang == 'ur':
        if not os.path.exists(MODEL_ONNX) or os.path.getsize(MODEL_ONNX) < 1000000:
            download_model_if_missing()

        if os.path.exists(MODEL_ONNX):
            try:
                output_wav = "output.wav"
                cmd = f'echo "{user_text}" | piper --model {MODEL_ONNX} --length-scale {user_speed} --output_file {output_wav}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
                    with open(output_wav, "rb") as f:
                        audio_data = f.read()
                    print(f"TTS Success: Female Piper TTS (Speed: {user_speed})")
                    return Response(audio_data, mimetype="audio/wav")
            except Exception as e:
                print(f"Piper execution error: {str(e)}")

    # 2️⃣ Google Backup Fallback
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
