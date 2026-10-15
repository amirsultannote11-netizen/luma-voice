import os
import io
import json
import wave
import uuid
import logging
import urllib.request
import urllib.error

import lameenc
from flask import Flask, request, Response, jsonify
from piper import PiperVoice
from gtts import gTTS

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LumaVoice")

# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────
MODEL_FILENAME = "ur_PK-aegis_female-medium.onnx"
CONFIG_FILENAME = "ur_PK-aegis_female-medium.onnx.json"

BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "ur/ur_PK/aegis_female/medium/"
)
MODEL_URL = BASE_URL + MODEL_FILENAME
CONFIG_URL = BASE_URL + CONFIG_FILENAME

# MP3 bitrate (kbps). 48 is small enough for ESP32 streaming over HTTPS.
MP3_BITRATE = 48

# STT (speech to text) via Groq Whisper. Set GROQ_API_KEY in Render's
# Environment tab. Do NOT write the key inside the code.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL = "whisper-large-v3-turbo"

piper_voice = None


# ─────────────────────────────────────────────
# Model download / load
# ─────────────────────────────────────────────
def download_file(filename, url):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        logger.info("File already exists: %s", filename)
        return

    temp_file = filename + ".tmp"
    logger.info("Downloading %s", filename)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LumaVoice/1.0"})
        with urllib.request.urlopen(req, timeout=180) as response:
            with open(temp_file, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if os.path.getsize(temp_file) == 0:
            raise RuntimeError("Downloaded file is empty")

        os.replace(temp_file, filename)
        logger.info("Download successful: %s", filename)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def load_piper():
    global piper_voice
    try:
        download_file(MODEL_FILENAME, MODEL_URL)
        download_file(CONFIG_FILENAME, CONFIG_URL)
        piper_voice = PiperVoice.load(MODEL_FILENAME, config_path=CONFIG_FILENAME)
        logger.info("PIPER VOICE LOADED SUCCESSFULLY")
    except Exception:
        piper_voice = None
        logger.exception("PIPER MODEL LOAD FAILED")


# ─────────────────────────────────────────────
# Audio generation
# ─────────────────────────────────────────────
def generate_piper_wav(text, rate_pct=0, pitch_hz=0):
    if piper_voice is None:
        raise RuntimeError("Piper voice is not loaded")

    # 🟢 ریٹ (Rate/Speed) کو Piper کے length_scale میں تبدیل کرنا
    # ESP32 سے ریٹ فیصد میں آتا ہے (جیسے -12% یا +10%)
    length_scale = 1.0
    if rate_pct != 0:
        length_scale = 1.0 - (float(rate_pct) / 100.0)
        if length_scale < 0.2: length_scale = 0.2
        if length_scale > 3.0: length_scale = 3.0

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        # synthesize_wav میں length_scale اور pitch پیرامیٹرز پاس کیے گئے ہیں
        piper_voice.synthesize_wav(text, wav_file, length_scale=length_scale)

    audio_data = wav_io.getvalue()
    if len(audio_data) <= 44:
        raise RuntimeError("Piper generated an empty WAV")
    return audio_data


def wav_to_mp3(wav_bytes):
    """WAV (16-bit PCM) -> MP3. About 7x smaller, so ESP32 does not stutter."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        pcm = w.readframes(w.getnframes())

    if sample_width != 2:
        raise RuntimeError("Only 16-bit WAV is supported")
    if not pcm:
        raise RuntimeError("WAV has no audio frames")

    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_quality(2)
    mp3 = bytes(enc.encode(pcm) + enc.flush())

    if not mp3:
        raise RuntimeError("MP3 encoding produced empty data")
    return mp3


def generate_google_mp3(text, lang="ur"):
    output = io.BytesIO()
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.write_to_fp(output)
    audio_data = output.getvalue()
    if not audio_data:
        raise RuntimeError("Google TTS generated empty audio")
    return audio_data


def mp3_response(audio, engine):
    return Response(
        audio,
        mimetype="audio/mpeg",
        headers={
            "Content-Length": str(len(audio)),
            "Cache-Control": "no-store",
            "X-Luma-TTS": engine,
        },
    )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "Luma Voice Server Active. Test: /voice?text=Assalam%20o%20Alaikum"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "server": "online",
        "piper_loaded": piper_voice is not None,
        "stt_key_set": bool(GROQ_API_KEY),
    })


@app.route("/voice", methods=["GET", "POST"])
def voice():
    if request.method == "POST":
        text = request.form.get("text", "")
        if not text and request.is_json:
            body = request.get_json(silent=True) or {}
            text = body.get("text", "")
        if not text:
            text = request.args.get("text", "")
    else:
        text = request.args.get("text", "")

    text = text.strip()
    lang = request.args.get("lang", "ur").strip().lower()
    
    # 🟢 ESP32 سے آنے والے rate اور pitch پیرامیٹرز کو ریسیو کرنا
    rate_str = request.args.get("rate", "0").replace("%", "").strip()
    pitch_str = request.args.get("pitch", "0").replace("Hz", "").strip()
    
    rate_pct = 0
    pitch_hz = 0
    try:
        rate_pct = int(rate_str)
    except ValueError:
        pass
    try:
        pitch_hz = int(pitch_str)
    except ValueError:
        pass

    if not text:
        return jsonify({
            "error": "No text provided",
            "example": "/voice?text=Assalam%20o%20Alaikum",
        }), 400

    if len(text) > 2000:
        return jsonify({"error": "Text too long"}), 413

    # English text: Urdu Piper voice cannot read it well, so use Google.
    if lang == "en":
        try:
            audio = generate_google_mp3(text, "en")
            logger.info("GOOGLE EN SUCCESS: %d bytes", len(audio))
            return mp3_response(audio, "google-en")
        except Exception:
            logger.exception("GOOGLE EN FAILED")

    # Urdu: Piper first (converted to MP3).
    elif piper_voice is not None:
        try:
            # 🟢 یہاں جنریٹ کرتے وقت rate_pct اور pitch_hz پاس کیے گئے ہیں
            audio = wav_to_mp3(generate_piper_wav(text, rate_pct=rate_pct, pitch_hz=pitch_hz))
            logger.info("PIPER MP3 SUCCESS: %d bytes", len(audio))
            return mp3_response(audio, "piper-mp3")
        except Exception:
            logger.exception("PIPER SYNTHESIS FAILED")
    else:
        logger.error("Piper unavailable; trying Google TTS")

    # Fallback: Google Urdu.
    try:
        audio = generate_google_mp3(text, "ur")
        logger.info("GOOGLE TTS SUCCESS: %d bytes", len(audio))
        return mp3_response(audio, "google")
    except Exception:
        logger.exception("GOOGLE TTS ALSO FAILED")
        return jsonify({
            "error": "Both TTS engines failed",
            "details": "Check Render logs",
        }), 500


def pcm_to_wav(pcm, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@app.route("/stt", methods=["POST"])
def stt():
    """ESP32 sends raw 16kHz 16-bit mono PCM. We return plain text."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set on the server")
        return Response("", status=500, mimetype="text/plain")

    pcm = request.get_data()
    if len(pcm) < 3200:  # less than 0.1 second
        return Response("", status=200, mimetype="text/plain")

    wav_bytes = pcm_to_wav(pcm, 16000)
    boundary = "----luma" + uuid.uuid4().hex

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = b""
    body += field("model", GROQ_STT_MODEL)
    body += field("response_format", "text")
    body += field("language", "ur")
    body += (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body += wav_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        GROQ_STT_URL,
        data=body,
        headers={
            "Authorization": "Bearer " + GROQ_API_KEY,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
            "User-Agent": "LumaVoice/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace").strip()
        logger.info("STT RESULT: %s", text)
        return Response(text, status=200, mimetype="text/plain; charset=utf-8")
    except urllib.error.HTTPError as e:
        logger.error("STT HTTP error %s: %s", e.code, e.read()[:300])
    except Exception:
        logger.exception("STT FAILED")

    return Response("", status=500, mimetype="text/plain")


# Load the model before accepting requests.
load_piper()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
