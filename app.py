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

# صحیح اور ورکنگ HuggingFace لنکس
ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/medium/ur_PK-medium.onnx"
JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/medium/ur_PK-medium.onnx.json"

def download_model_if_missing():
    try:
        if not os.path.exists(MODEL_ONNX):
            print("Female Piper ONNX model downloading...")
            req = urllib.request.Request(ONNX_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(MODEL_ONNX, 'wb') as out_file:
                out_file.write(response.read())
            print("ONNX model downloaded successfully!")
            
        if not os.path.exists(MODEL_JSON):
            print("Female Piper JSON config downloading...")
            req = urllib.request.Request(JSON_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(MODEL_JSON, 'wb') as out_file:
                out_file.write(response.read())
            print("JSON config downloaded successfully!")
    except Exception as e:
        print(f"Model download warning (using fallback if needed): {str(e)}")

# سرور سٹارٹ ہوتے ہی ماڈل چیک اور ڈاؤن لوڈ ہوگا
download_model_if_missing()


@app.route('/')
def home():
    return "Luma Hybrid Voice Server (Female Piper + Multi-Language) is Running!"

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

# ─── ہائبرڈ TTS سسٹم: اسپیڈ اور زبان (Urdu/English) سلیکشن کے ساتھ ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur').lower()  # 'ur' یا 'en'
    user_speed = request.args.get('speed', '1.0')       # default speed 1.0

    if not user_text:
        return "Error: No text provided", 400

    # 1️⃣ اگر اردو سلیکٹ کی ہو اور ماڈل موجود ہو تو Piper استعمال کریں
    if user_lang == 'ur' and os.path.exists(MODEL_ONNX):
        try:
            output_wav = "output.wav"
            model_path = MODEL_ONNX
            
            # Piper speed control using --length-scale
            cmd = f'echo "{user_text}" | piper --model {model_path} --length-scale {user_speed} --output_file {output_wav}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
                with open(output_wav, "rb") as f:
                    audio_data = f.read()
                print(f"TTS Success: Female Piper Urdu (Speed: {user_speed})")
                return Response(audio_data, mimetype="audio/wav")
                
        except Exception as e:
            print(f"Piper failed, switching to backup: {str(e)}")

    # 2️⃣ اگر انگلش سلیکٹ ہو یا Piper فیل ہو تو Google gTTS استعمال ہوگا
    try:
        print(f"TTS Generating via Google gTTS (Lang: {user_lang})")
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
