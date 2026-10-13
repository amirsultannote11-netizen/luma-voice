import os
import subprocess
import urllib.request
from flask import Flask, request, Response
from gtts import gTTS
import speech_recognition as sr
import io

app = Flask(__name__)

# ─── پائپر ماڈل فائلیں خود بخود ڈاؤن لوڈ کرنے کا فنکشن ───
MODEL_ONNX = "ur_PK-fasih-medium.onnx"
MODEL_JSON = "ur_PK-fasih-medium.onnx.json"

ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx"
JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx.json"

def download_model_if_missing():
    if not os.path.exists(MODEL_ONNX):
        print("Piper ONNX model not found. Downloading from HuggingFace...")
        urllib.request.urlretrieve(ONNX_URL, MODEL_ONNX)
        print("ONNX model downloaded successfully!")
        
    if not os.path.exists(MODEL_JSON):
        print("Piper JSON config not found. Downloading from HuggingFace...")
        urllib.request.urlretrieve(JSON_URL, MODEL_JSON)
        print("JSON config downloaded successfully!")

# سرور شروع ہوتے ہی ماڈل چیک اور ڈاؤن لوڈ ہوگا
download_model_if_missing()


@app.route('/')
def home():
    return "Luma Hybrid Voice Server (Piper + Google Backup) is Running!"

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

# ─── ہائبرڈ TTS سسٹم: پہلے Piper، اگر فیل ہو تو Google gTTS ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur')

    if not user_text:
        return "Error: No text provided", 400

    # 1️⃣ کوشش کریں کہ پہلے Piper (نیورل آواز) سے آڈیو بنے
    try:
        output_wav = "output.wav"
        model_path = MODEL_ONNX
        
        # Piper کمانڈ چلانا
        cmd = f'echo "{user_text}" | piper --model {model_path} --output_file {output_wav}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
            with open(output_wav, "rb") as f:
                audio_data = f.read()
            print("TTS Success: Generated using Piper TTS")
            return Response(audio_data, mimetype="audio/wav")
            
    except Exception as e:
        print(f"Piper failed, switching to Google backup: {str(e)}")

    # 2️⃣ اگر Piper کام نہ کرے تو خود بخود Google gTTS پر شفٹ ہو جائیں (بیک اپ)
    try:
        print("TTS Fallback: Generating using Google gTTS")
        tts = gTTS(text=user_text, lang='ur' if user_lang == 'ur' else 'en', slow=False)
        
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
