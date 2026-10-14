import os
from flask import Flask, request, Response
from gtts import gTTS
import speech_recognition as sr
import io
import subprocess
import tempfile

app = Flask(__name__)

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

# ─── Piper TTS اور Google TTS بیک اپ سسٹم ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_lang = request.args.get('lang', 'ur')

    if not user_text:
        return "Error: No text provided", 400

    # 1. پہلے Piper TTS کے ذریعے خاتون کی خوبصورت آواز بنانے کی کوشش کریں
    try:
        model_name = "ur_PK-aegis_female-medium.onnx" if user_lang == 'ur' else "en_US-lessac-medium.onnx"
        
        if os.path.exists(model_name):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                wav_path = wav_file.name

            cmd = f"echo '{user_text}' | piper --model {model_name} --output_file {wav_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
                os.remove(wav_path)
                return Response(audio_bytes, mimetype="audio/wav")
    except Exception as piper_err:
        print(f"Piper TTS Error (Falling back to gTTS): {str(piper_err)}")

    # 2. اگر Piper میں کوئی مسئلہ ہو تو گوگل gTTS بطورِ بیک اپ استعمال ہوگا
    try:
        tts = gTTS(text=user_text, lang='ur' if user_lang == 'ur' else 'en', slow=False)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return Response(audio_io.read(), mimetype="audio/mpeg")
        
    except Exception as e:
        print(f"TTS Backup Error: {str(e)}")
        return f"TTS Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
