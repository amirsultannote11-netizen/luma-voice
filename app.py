import os
import asyncio
from flask import Flask, request, Response, jsonify
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
    return "Luma Original Voice Server is Running!"

# ─── مائیک کی آواز کو ٹیکسٹ میں بدلنے کا نیا سسٹم (STT) ───
@app.route('/stt', methods=['POST'])
def speech_to_text():
    try:
        # ارڈوینو سے آنے والا خام آڈیو ڈیٹا (Raw Audio) پکڑنا
        audio_data = request.data
        if not audio_data:
            return "Error: No audio data received", 400

        # ارڈوینو 16000Hz، 16-bit، Mono پر ڈیٹا بھیج رہا ہے، اس کو WAV فارمیٹ کی ہیڈر دینا
        wav_header = io.BytesIO()
        sample_rate = 16000
        bits_per_sample = 16
        channels = 1
        data_len = len(audio_data)
        
        # WAV کا پورا ہیڈر بائٹس میں بنانا
        wav_header.write(b'RIFF')
        wav_header.write((data_len + 36).to_bytes(4, 'little'))
        wav_header.write(b'WAVEfmt ')
        wav_header.write((16).to_bytes(4, 'little'))
        wav_header.write((1).to_bytes(2, 'little')) # PCM Linear
        wav_header.write((channels).to_bytes(2, 'little'))
        wav_header.write((sample_rate).to_bytes(4, 'little'))
        wav_header.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
        wav_header.write((channels * bits_per_sample // 8).to_bytes(2, 'little'))
        wav_header.write((bits_per_sample).to_bytes(2, 'little'))
        wav_header.write(b'data')
        wav_header.write((data_len).to_bytes(4, 'little'))
        wav_header.write(audio_data)
        
        wav_header.seek(0)

        # گوگل اسپیچ ریکگنیشن کے ذریعے آڈیو سے ٹیکسٹ نکالنا
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_header) as source:
            audio_file_data = recognizer.record(source)
            
        # اردو زبان میں آواز کو پہچاننا (کیونکہ ہماری لوما اردو بولتی ہے)
        recognized_text = recognizer.recognize_google(audio_file_data, language="ur-PK")
        print(f"Server Recognized: {recognized_text}")
        
        return Response(recognized_text, mimetype="text/plain")

    except sr.UnknownValueError:
        return "معذرت، آواز سمجھ نہیں آئی", 200 # اگر آواز صاف نہ ہو
    except sr.RequestError as e:
        return f"Google STT Service Error: {str(e)}", 500
    except Exception as e:
        return f"STT Server Error: {str(e)}", 500

# ─── آپ کا پرانا والا ٹیکسٹ ٹو اسپیچ سسٹم (بالکل ویسا ہی ہے) ───
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
            communicate = edge_tts.Communicate(
                text=user_text, 
                voice=selected_voice, 
                rate=user_rate, 
                pitch=user_pitch
            )
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        audio_stream = asyncio.run(generate())

        if len(audio_stream) == 0:
            return "Error: Edge TTS returned empty audio", 500

        return Response(audio_stream, mimetype="audio/mpeg")
        
    except Exception as e:
        return f"Microsoft TTS Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)