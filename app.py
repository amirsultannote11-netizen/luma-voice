import os
import io
import asyncio
from flask import Flask, request, Response
import speech_recognition as sr

# Google TTS (Backup کے لیے)
from gtts import gTTS

# Piper TTS (Main اعلیٰ ترین وائس کے لیے)
from piper import PiperVoice

app = Flask(__name__)

# ─── PIPER VOICE MODEL SETUP (Main Voice) ───
PIPER_MODEL_PATH = "ur_PK-aegis_female-medium.onnx"
piper_voice = None

try:
    if os.path.exists(PIPER_MODEL_PATH):
        piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
        print("Piper Urdu Female Voice Loaded Successfully!")
    else:
        print("Piper model file not found! Falling back to Google TTS.")
except Exception as e:
    print(f"Error loading Piper model: {e}")


@app.route('/')
def home():
    return "Luma Voice Server (Piper AI + Google TTS Backup) is Running!"


# ─── 1. مائیک کی آواز کو ٹیکسٹ میں بدلنے کا سسٹم (STT) ───
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


# ─── 2. ٹیکسٹ ٹو اسپیچ (Piper Main + Google TTS Backup) ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')

    if not user_text:
        return "Error: No text provided", 400

    # Option A: پہلے Piper TTS (خوبصورت نیورل فی میل وائس)
    if piper_voice:
        try:
            wav_io = io.BytesIO()
            piper_voice.synthesize(user_text, wav_io)
            wav_bytes = wav_io.getvalue()
            
            print("Successfully generated audio using Piper TTS.")
            return Response(wav_bytes, mimetype="audio/wav")
        except Exception as e:
            print(f"Piper TTS Failed: {e}. Switching to Google TTS...")

    # Option B: اگر Piper میں مسئلہ آئے تو خودبخود Google TTS (Backup)
    try:
        tts = gTTS(text=user_text, lang='ur', slow=False)
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_bytes = mp3_io.getvalue()

        print("Successfully generated audio using Google TTS (Backup).")
        return Response(mp3_bytes, mimetype="audio/mpeg")

    except Exception as e:
        print(f"Google TTS Error: {str(e)}")
        return f"TTS Processing Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
