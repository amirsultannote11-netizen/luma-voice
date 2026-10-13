import os
import asyncio
from flask import Flask, request, Response
import edge_tts
import speech_recognition as sr
import io

app = Flask(__name__)

# مائیکروسافٹ کی تمام اصل آوازیں
VOICES = {
    "ur": {
        "female": "ur-PK-UzmaNeural",
        "male": "ur-PK-AsadNeural"
    },
    "en": {
        "female": "en-US-EmmaNeural",
        "male": "en-US-BrianNeural"
    }
}

@app.route('/')
def home():
    return "Luma Original Microsoft Voice Server is Running!"

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

# ─── مائیکروسافٹ ٹیکسٹ ٹو اسپیچ (Pitch & Speed Control کے ساتھ) ───
@app.route('/voice', methods=['GET'])
def text_to_speech():
    user_text = request.args.get('text', '')
    user_rate = request.args.get('rate', '-12%')
    user_pitch = request.args.get('pitch', '0Hz')
    user_lang = request.args.get('lang', 'ur')   
    user_voice = request.args.get('voice', 'female') 

    try:
        selected_voice = VOICES[user_lang][user_voice]
    except KeyError:
        selected_voice = VOICES["ur"]["female"] 
        
    if not user_pitch.endswith('Hz'):
        user_pitch = user_pitch + "Hz"
        
    if not user_pitch.startswith('-') and not user_pitch.startswith('+'):
        user_pitch = "+" + user_pitch
    if not user_rate.startswith('-') and not user_rate.startswith('+'):
        user_rate = "+" + user_rate

    if not user_text:
        return "Error: No text provided", 400

    try:
        async def generate():
            # مائیکروسافٹ کنکشن کے لیے کمیونیکیٹر
            communicate = edge_tts.Communicate(
                text=user_text, 
                voice=selected_voice, 
                rate=user_rate, 
                pitch=user_pitch
            )
            audio_stream_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_stream_bytes.extend(chunk["data"])
            return bytes(audio_stream_bytes)

        audio_output = asyncio.run(generate())

        if not audio_output or len(audio_output) == 0:
            return "Error: Edge TTS returned empty audio", 500

        return Response(audio_output, mimetype="audio/mpeg")
        
    except Exception as e:
        print(f"Microsoft TTS Error: {str(e)}")
        return f"Microsoft TTS Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
