import os
import io
import re
import time
import json
import threading
import wave
import uuid
import logging
import gc
import urllib.request
import urllib.error

import lameenc
import numpy as np
from flask import Flask, request, Response, jsonify
from piper import PiperVoice

try:
    from piper import SynthesisConfig
except Exception:
    SynthesisConfig = None
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

MP3_BITRATE = 48

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
# Audio generation & Memory Optimization
# ─────────────────────────────────────────────
def generate_piper_wav(text, length_scale=1.0):
    if piper_voice is None:
        raise RuntimeError("Piper voice is not loaded")

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        if SynthesisConfig is not None and abs(length_scale - 1.0) > 0.01:
            syn = SynthesisConfig(length_scale=length_scale)
            piper_voice.synthesize_wav(text, wav_file, syn_config=syn)
        else:
            piper_voice.synthesize_wav(text, wav_file)

    audio_data = wav_io.getvalue()
    wav_io.close()
    if len(audio_data) <= 44:
        raise RuntimeError("Piper generated an empty WAV")
    return audio_data


def shift_pitch_pcm(pcm_bytes, factor):
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    new_n = max(1, int(n / factor))
    positions = np.linspace(0, n - 1, new_n)
    shifted = np.interp(positions, np.arange(n), samples)
    res = np.clip(shifted, -32768, 32767).astype(np.int16).tobytes()
    del samples, shifted, positions
    return res


def wav_to_mp3(wav_bytes, pitch_factor=1.0):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        pcm = w.readframes(w.getnframes())

    if sample_width != 2:
        raise RuntimeError("Only 16-bit WAV is supported")
    if not pcm:
        raise RuntimeError("WAV has no audio frames")

    if abs(pitch_factor - 1.0) > 0.005:
        pcm = shift_pitch_pcm(pcm, pitch_factor)

    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_quality(2)
    mp3 = bytes(enc.encode(pcm) + enc.flush())

    del pcm
    gc.collect()

    if not mp3:
        raise RuntimeError("MP3 encoding produced empty data")
    return mp3


def generate_google_mp3(text, lang="ur"):
    output = io.BytesIO()
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.write_to_fp(output)
    audio_data = output.getvalue()
    output.close()
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


def parse_number(value, default=0.0):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value or "")
    try:
        return float(match.group()) if match else default
    except ValueError:
        return default


def voice_params():
    rate_pct = max(-50.0, min(50.0, parse_number(request.args.get("rate"))))
    pitch_hz = max(-20.0, min(20.0, parse_number(request.args.get("pitch"))))

    speed_scale = 1.0 / (1.0 + rate_pct / 100.0)
    pitch_factor = 2.0 ** (pitch_hz / 100.0)
    length_scale = max(0.4, min(2.5, speed_scale * pitch_factor))
    return rate_pct, pitch_hz, length_scale, pitch_factor


# ─────────────────────────────────────────────
# Routes & Strict Memory Caching
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


AUDIO_CACHE = {}
AUDIO_CACHE_LOCK = threading.Lock()
AUDIO_TTL_SECONDS = 180
AUDIO_CACHE_MAX_ITEMS = 2


def cache_put(audio, engine):
    audio_id = uuid.uuid4().hex[:16]
    now = time.time()
    with AUDIO_CACHE_LOCK:
        for key in [k for k, v in AUDIO_CACHE.items() if now - v["t"] > AUDIO_TTL_SECONDS]:
            del AUDIO_CACHE[key]
        while len(AUDIO_CACHE) >= AUDIO_CACHE_MAX_ITEMS:
            oldest = min(AUDIO_CACHE, key=lambda k: AUDIO_CACHE[k]["t"])
            del AUDIO_CACHE[oldest]
        AUDIO_CACHE[audio_id] = {"audio": audio, "engine": engine, "t": now}

    gc.collect()
    return audio_id


def cache_get(audio_id):
    with AUDIO_CACHE_LOCK:
        item = AUDIO_CACHE.get(audio_id)
        return (item["audio"], item["engine"]) if item else None


def synthesize(text, lang, length_scale, pitch_factor):
    audio = None
    engine = "google"
    if lang == "en":
        try:
            audio = generate_google_mp3(text, "en")
            engine = "google-en"
        except Exception:
            logger.exception("GOOGLE EN FAILED")
    elif piper_voice is not None:
        try:
            wav_data = generate_piper_wav(text, length_scale)
            audio = wav_to_mp3(wav_data, pitch_factor)
            del wav_data
            engine = "piper-mp3"
        except Exception:
            logger.exception("PIPER SYNTHESIS FAILED")

    if audio is None:
        audio = generate_google_mp3(text, "ur")
        engine = "google"

    gc.collect()
    return audio, engine


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
    rate_pct, pitch_hz, length_scale, pitch_factor = voice_params()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long"}), 413

    try:
        audio, engine = synthesize(text, lang, length_scale, pitch_factor)
        resp = mp3_response(audio, engine)
        del audio
        gc.collect()
        return resp
    except Exception:
        logger.exception("ALL TTS ENGINES FAILED")
        return jsonify({"error": "TTS failed"}), 500


@app.route("/tts", methods=["POST"])
def tts():
    text = request.get_data(as_text=True).strip()
    lang = request.args.get("lang", "ur").strip().lower()
    rate_pct, pitch_hz, length_scale, pitch_factor = voice_params()

    if not text or len(text) > 2000:
        return Response("", status=400, mimetype="text/plain")

    try:
        audio, engine = synthesize(text, lang, length_scale, pitch_factor)
    except Exception:
        logger.exception("ALL TTS ENGINES FAILED")
        return Response("", status=500, mimetype="text/plain")

    audio_id = cache_put(audio, engine)
    del audio
    gc.collect()
    return Response(audio_id, status=200, mimetype="text/plain")


@app.route("/audio/<audio_id>.mp3", methods=["GET"])
def audio_file(audio_id):
    found = cache_get(audio_id)
    if not found:
        return Response("Audio expired or not found", status=404, mimetype="text/plain")
    audio, engine = found
    resp = mp3_response(audio, engine)
    gc.collect()
    return resp


def pcm_to_wav(pcm, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    val = buf.getvalue()
    buf.close()
    return val


# ─────────────────────────────────────────────
# 🔹 پرانا ایک-شاٹ STT (برقرار)
# ─────────────────────────────────────────────
def _send_to_groq_stt(pcm, lang="ur"):
    """پورے PCM کو WAV بنا کر Groq Whisper پر بھیجیں اور ٹیکسٹ لوٹائیں۔"""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

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
    body += field("language", lang)
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

    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace").strip()

    del body, wav_bytes
    gc.collect()
    return text


@app.route("/stt", methods=["POST"])
def stt():
    if not GROQ_API_KEY:
        return Response("", status=500, mimetype="text/plain")

    pcm = request.get_data()
    if len(pcm) < 3200:
        return Response("", status=200, mimetype="text/plain")

    try:
        text = _send_to_groq_stt(pcm, lang="ur")
        del pcm
        gc.collect()
        return Response(text, status=200, mimetype="text/plain; charset=utf-8")
    except Exception:
        logger.exception("STT FAILED")
        gc.collect()

    return Response("", status=500, mimetype="text/plain")


# ─────────────────────────────────────────────
# 🔹 نیا اسٹریمنگ STT (start / chunk / end)
#    ریکارڈنگ کے دوران چھوٹے چھوٹے ٹکڑے
#    سرور پر جمع ہوتے ہیں، پھر ایک ساتھ
#    Whisper پر بھیجے جاتے ہیں۔ ESP32 پر
#    کبھی بڑا بفر نہیں بنتا → میموری محفوظ۔
# ─────────────────────────────────────────────
STT_SESSIONS = {}
STT_SESSIONS_LOCK = threading.Lock()
STT_SESSION_TTL = 60
STT_MAX_CHUNKS = 500  # ~500 * 4KB ≈ 2MB (کافی زیادہ)


def cleanup_stt_sessions():
    now = time.time()
    with STT_SESSIONS_LOCK:
        for sid in [k for k, v in STT_SESSIONS.items() if now - v["t"] > STT_SESSION_TTL]:
            del STT_SESSIONS[sid]


@app.route("/stt/start", methods=["POST"])
def stt_start():
    if not GROQ_API_KEY:
        return Response("", status=500, mimetype="text/plain")

    cleanup_stt_sessions()
    session_id = uuid.uuid4().hex[:12]
    with STT_SESSIONS_LOCK:
        STT_SESSIONS[session_id] = {"chunks": [], "t": time.time()}

    logger.info("STT session started: %s", session_id)
    return Response(session_id, status=200, mimetype="text/plain")


@app.route("/stt/chunk", methods=["POST"])
def stt_chunk():
    session_id = request.args.get("id", "").strip()
    if not session_id:
        return Response("", status=400)

    chunk = request.get_data()
    if not chunk:
        return Response("", status=400)

    with STT_SESSIONS_LOCK:
        session = STT_SESSIONS.get(session_id)
        if session is None:
            return Response("session_expired", status=404, mimetype="text/plain")
        if len(session["chunks"]) >= STT_MAX_CHUNKS:
            return Response("too_long", status=413, mimetype="text/plain")
        session["chunks"].append(chunk)
        session["t"] = time.time()

    return Response("OK", status=200, mimetype="text/plain")


@app.route("/stt/end", methods=["POST"])
def stt_end():
    if not GROQ_API_KEY:
        return Response("", status=500, mimetype="text/plain")

    session_id = request.args.get("id", "").strip()
    if not session_id:
        return Response("", status=400)

    with STT_SESSIONS_LOCK:
        session = STT_SESSIONS.pop(session_id, None)

    if session is None:
        return Response("session_expired", status=404, mimetype="text/plain")

    pcm = b"".join(session["chunks"])
    total_chunks = len(session["chunks"])
    del session
    gc.collect()

    logger.info("STT end: id=%s chunks=%d bytes=%d", session_id, total_chunks, len(pcm))

    if len(pcm) < 3200:
        del pcm
        gc.collect()
        return Response("", status=200, mimetype="text/plain")

    try:
        text = _send_to_groq_stt(pcm, lang="ur")
        del pcm
        gc.collect()
        logger.info("STT end → text: %s", text[:80])
        return Response(text, status=200, mimetype="text/plain; charset=utf-8")
    except Exception:
        logger.exception("STT END FAILED")
        gc.collect()
        return Response("", status=500, mimetype="text/plain")


# Load the model before accepting requests.
load_piper()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
