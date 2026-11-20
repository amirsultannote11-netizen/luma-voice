import os
import io
import re
import time
import json
import queue
import hashlib
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
# ONNX Runtime کی میموری کم رکھنا
#   بڑی "arena" میموری بند، تھریڈ PIPER_THREADS (ڈیفالٹ 1)۔ Piper سے پہلے لگنا ضروری ہے۔
# ─────────────────────────────────────────────
PIPER_THREADS = int(os.environ.get("PIPER_THREADS", "1"))
try:
    import onnxruntime as _ort

    _orig_session = _ort.InferenceSession

    def _lean_session(path, sess_options=None, *args, **kwargs):
        so = sess_options or _ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.intra_op_num_threads = PIPER_THREADS
        so.inter_op_num_threads = 1
        return _orig_session(path, so, *args, **kwargs)

    _ort.InferenceSession = _lean_session
    logger.info("ONNX lean session patch applied (threads=%d)", PIPER_THREADS)
except Exception:
    logger.exception("ONNX lean session patch failed (continuing without it)")


def trim_memory():
    """Python اور C دونوں کی خالی میموری سسٹم کو واپس دو۔"""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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

# ── English Female (Amy) ──
EN_MODEL_FILENAME = "en_US-amy-medium.onnx"
EN_CONFIG_FILENAME = "en_US-amy-medium.onnx.json"
EN_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/amy/medium/"
)
EN_MODEL_URL = EN_BASE_URL + EN_MODEL_FILENAME
EN_CONFIG_URL = EN_BASE_URL + EN_CONFIG_FILENAME

MP3_BITRATE = 64

# ایک وقت میں میموری میں کتنی Piper آوازیں رہ سکتی ہیں (1 = کم میموری، 2 = دونوں ساتھ)
MAX_LOADED_VOICES = int(os.environ.get("PIPER_MAX_VOICES", "1"))

# /voice (غیر اسٹریمنگ) کے لیے ٹکڑے کی حد
CHUNK_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "140"))

# ── اسٹریمنگ کے ٹکڑوں کی حدیں ──
# پہلا ٹکڑا بہت چھوٹا تاکہ آواز جلدی شروع ہو، پھر آہستہ آہستہ بڑے۔
STREAM_LIMITS = [
    int(os.environ.get("TTS_FIRST_CHARS", "60")),
    int(os.environ.get("TTS_SECOND_CHARS", "100")),
]
STREAM_QUEUE_MAX = 3          # تیار شدہ ٹکڑے جو ایک وقت میں قطار میں رہ سکتے ہیں (میموری کی حد)
STREAM_IDLE_TIMEOUT = 90      # اتنی دیر کوئی ٹکڑا نہ آئے تو اسٹریم بند
MAX_TEXT_CHARS = 1200         # اس سے لمبا متن رد

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL = "whisper-large-v3-turbo"

VOICE_FILES = {
    "ur": (MODEL_FILENAME, MODEL_URL, CONFIG_FILENAME, CONFIG_URL),
    "en": (EN_MODEL_FILENAME, EN_MODEL_URL, EN_CONFIG_FILENAME, EN_CONFIG_URL),
}
VOICES = {"ur": None, "en": None}

# ایک وقت میں ایک ہی آواز بنے تاکہ میموری/CPU پر دباؤ نہ پڑے
SYNTH_LOCK = threading.Lock()


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


def get_voice(lang):
    """مطلوبہ آواز لوٹاتا ہے۔ ضرورت ہو تو لوڈ کرتا ہے اور حد سے زیادہ ہونے پر پرانی ہٹاتا ہے۔
    اسے SYNTH_LOCK کے اندر ہی بلانا ہے (load_piper کے سوا)۔"""
    key = "en" if lang == "en" else "ur"
    if VOICES[key] is not None:
        return VOICES[key]

    loaded = [k for k, v in VOICES.items() if v is not None]
    while loaded and len(loaded) >= MAX_LOADED_VOICES:
        victim = loaded.pop(0)
        VOICES[victim] = None
        logger.info("Unloaded voice: %s (memory limit)", victim)
    trim_memory()

    model_f, model_u, conf_f, conf_u = VOICE_FILES[key]
    try:
        download_file(model_f, model_u)
        download_file(conf_f, conf_u)
        VOICES[key] = PiperVoice.load(model_f, config_path=conf_f)
        logger.info("PIPER VOICE LOADED: %s", key)
    except Exception:
        VOICES[key] = None
        logger.exception("PIPER %s MODEL LOAD FAILED", key)
    trim_memory()
    return VOICES[key]


def load_piper():
    """شروع میں صرف اردو لوڈ ہوتی ہے، اور ایک چھوٹا سا ٹیسٹ جملہ چلا کر 'گرم' کر دیا جاتا ہے
    تاکہ پہلی اصل درخواست پر ONNX کی پہلی سست رن نہ ہو۔"""
    voice = get_voice("ur")
    if voice is None:
        return
    try:
        t0 = time.time()
        generate_piper_wav("جی ہاں۔", 1.0, voice)
        logger.info("Piper warm-up done in %.2fs", time.time() - t0)
    except Exception:
        logger.exception("Piper warm-up failed (ignored)")
    trim_memory()


# ─────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[۔؟!?.\n])\s+")
_CUT_CHARS = "۔؟!?.\n،,؛;"


def split_text(text, max_len=None):
    """لمبے متن کو جملوں کے حساب سے چھوٹے ٹکڑوں میں توڑتا ہے (/voice کے لیے)۔"""
    max_len = max_len or CHUNK_CHARS
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks = []
    cur = ""
    for s in sentences:
        while len(s) > max_len:
            cut = max(s.rfind("،", 0, max_len), s.rfind(",", 0, max_len), s.rfind(" ", 0, max_len))
            if cut < 40:
                cut = max_len - 1
            head, s = s[:cut + 1].strip(), s[cut + 1:].strip()
            if cur:
                chunks.append(cur)
                cur = ""
            if head:
                chunks.append(head)
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _take(text, limit):
    """متن سے زیادہ سے زیادہ `limit` حروف کا ٹکڑا لو، ترجیحاً جملے/کاما پر کاٹ کر۔
    (ٹکڑا، بچا ہوا متن) لوٹاتا ہے۔"""
    text = text.strip()
    if len(text) <= limit:
        return text, ""
    window = text[:limit]
    cut = max(window.rfind(c) for c in _CUT_CHARS)
    if cut < limit * 0.3:
        sp = window.rfind(" ")
        cut = sp if sp >= limit * 0.3 else limit - 1
    return text[:cut + 1].strip(), text[cut + 1:].strip()


def split_for_stream(text):
    """اسٹریمنگ کے لیے: پہلا ٹکڑا چھوٹا، دوسرا درمیانہ، باقی عام سائز کے۔"""
    out = []
    rest = text.strip()
    i = 0
    while rest:
        limit = STREAM_LIMITS[i] if i < len(STREAM_LIMITS) else CHUNK_CHARS
        head, rest = _take(rest, limit)
        if head:
            out.append(head)
        i += 1
    return out or [text.strip()]


# ─────────────────────────────────────────────
# Audio generation
# ─────────────────────────────────────────────
def generate_piper_wav(text, length_scale=1.0, voice=None):
    """ایک چھوٹے ٹکڑے کی WAV بناتا ہے۔ آواز نہ بنے تو b'' لوٹاتا ہے۔"""
    if voice is None:
        raise RuntimeError("Piper voice is not loaded")

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        if SynthesisConfig is not None and abs(length_scale - 1.0) > 0.01:
            syn = SynthesisConfig(length_scale=length_scale)
            voice.synthesize_wav(text, wav_file, syn_config=syn)
        else:
            voice.synthesize_wav(text, wav_file)

    audio_data = wav_io.getvalue()
    wav_io.close()
    if len(audio_data) <= 44:
        return b""
    return audio_data


def shift_pitch_pcm(pcm_bytes, factor):
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    new_n = max(1, int(n / factor))
    positions = np.linspace(0, n - 1, new_n, dtype=np.float32)
    shifted = np.interp(positions, np.arange(n, dtype=np.float32), samples)
    res = np.clip(shifted, -32768, 32767).astype(np.int16).tobytes()
    del samples, shifted, positions
    return res


def _wav_to_pcm(wav):
    with wave.open(io.BytesIO(wav), "rb") as w:
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        pcm = w.readframes(w.getnframes())
    if sample_width != 2:
        raise RuntimeError("Only 16-bit WAV is supported")
    return pcm, sample_rate, channels


def _new_encoder(sample_rate, channels):
    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_quality(2)
    return enc


def generate_piper_mp3(text, length_scale, pitch_factor, voice):
    """(/voice کے لیے) پورا متن ٹکڑوں میں بنا کر ایک ہی MP3 لوٹاتا ہے۔"""
    enc = None
    parts = []

    for piece in split_text(text):
        wav = generate_piper_wav(piece, length_scale, voice)
        if not wav:
            continue
        pcm, sample_rate, channels = _wav_to_pcm(wav)
        del wav
        if not pcm:
            continue
        if abs(pitch_factor - 1.0) > 0.005:
            pcm = shift_pitch_pcm(pcm, pitch_factor)
        if enc is None:
            enc = _new_encoder(sample_rate, channels)
        data = enc.encode(pcm)
        if data:
            parts.append(bytes(data))
        del pcm

    if enc is None:
        raise RuntimeError("Piper produced no audio")

    parts.append(bytes(enc.flush()))
    mp3 = b"".join(parts)
    del parts
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
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "Luma Voice Server Active. Test: /voice?text=Assalam%20o%20Alaikum"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "server": "online",
        "piper_loaded": VOICES["ur"] is not None,
        "piper_en_loaded": VOICES["en"] is not None,
        "max_loaded_voices": MAX_LOADED_VOICES,
        "stt_key_set": bool(GROQ_API_KEY),
    })


@app.route("/ping", methods=["GET"])
def ping():
    return "ok"


# ─────────────────────────────────────────────
# Jobs: /tts صرف متن محفوظ کر کے فوراً آئی ڈی دیتا ہے (کوئی آڈیو نہیں بنتی)۔
# آڈیو تب بنتی ہے جب ESP32 /audio/<id>.mp3 مانگے، اور بنتے ہی بھیجی جاتی ہے۔
# صرف متن محفوظ ہوتا ہے، اس لیے میموری بالکل کم لگتی ہے۔
# ─────────────────────────────────────────────
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 180
JOB_MAX_ITEMS = 4

ACTIVE_CANCEL = None          # چلتی ہوئی اسٹریم کو روکنے کا اشارہ
ACTIVE_LOCK = threading.Lock()


def job_put(text, lang, length_scale, pitch_factor):
    job_id = uuid.uuid4().hex[:16]
    now = time.time()
    with JOBS_LOCK:
        for key in [k for k, v in JOBS.items() if now - v["t"] > JOB_TTL_SECONDS]:
            del JOBS[key]
        while len(JOBS) >= JOB_MAX_ITEMS:
            oldest = min(JOBS, key=lambda k: JOBS[k]["t"])
            del JOBS[oldest]
        JOBS[job_id] = {
            "text": text,
            "lang": lang,
            "ls": length_scale,
            "pf": pitch_factor,
            "t": now,
        }
    return job_id


def job_get(job_id):
    with JOBS_LOCK:
        item = JOBS.get(job_id)
        return dict(item) if item else None


def _put_item(q, item, cancel):
    """قطار بھری ہو تو انتظار کرو، مگر منسوخ ہو جائے تو فوراً نکل جاؤ۔"""
    while not cancel.is_set():
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _piper_piece(piece, job, enc, key):
    """ایک ٹکڑے کا MP3 بناؤ۔ SYNTH_LOCK صرف Piper کی synthesis کے دوران پکڑا جاتا ہے۔"""
    with SYNTH_LOCK:
        voice = get_voice(key)
        if voice is None:
            raise RuntimeError("Piper voice is not loaded")
        wav = generate_piper_wav(piece, job["ls"], voice)
    if not wav:
        return b"", enc
    pcm, sample_rate, channels = _wav_to_pcm(wav)
    del wav
    if not pcm:
        return b"", enc
    if abs(job["pf"] - 1.0) > 0.005:
        pcm = shift_pitch_pcm(pcm, job["pf"])
    if enc is None:
        enc = _new_encoder(sample_rate, channels)
    data = bytes(enc.encode(pcm))
    del pcm
    return data, enc


def _producer(job, q, cancel):
    """الگ تھریڈ: ٹکڑے بنا کر قطار میں ڈالتا ہے۔ نیٹ ورک سست ہو تب بھی اگلا ٹکڑا بنتا رہتا ہے،
    اس لیے ٹکڑوں کے بیچ وقفہ نہیں آتا۔"""
    key = "en" if job["lang"] == "en" else "ur"
    enc = None
    sent_any = False
    t0 = time.time()
    try:
        pieces = split_for_stream(job["text"])
        logger.info("STREAM start: %d pieces, lang=%s", len(pieces), key)
        for n, piece in enumerate(pieces):
            if cancel.is_set():
                logger.info("STREAM cancelled")
                return
            try:
                data, enc = _piper_piece(piece, job, enc, key)
            except Exception:
                logger.exception("PIPER piece failed")
                if not sent_any:
                    # پہلے ہی ٹکڑے پر ناکامی: پورے متن کے لیے گوگل بیک اپ
                    try:
                        mp3 = generate_google_mp3(job["text"], "en" if key == "en" else "ur")
                        _put_item(q, mp3, cancel)
                        logger.info("STREAM fallback to gTTS")
                    except Exception:
                        logger.exception("GOOGLE TTS FAILED")
                    return
                break   # بیچ میں ناکامی: جو بھیجا جا چکا وہ ٹھیک ہے، اسٹریم ختم
            if data:
                if not _put_item(q, data, cancel):
                    return
                if not sent_any:
                    logger.info("STREAM first audio after %.2fs", time.time() - t0)
                sent_any = True
        if enc is not None:
            tail = bytes(enc.flush())
            if tail:
                _put_item(q, tail, cancel)
        logger.info("STREAM done in %.2fs", time.time() - t0)
    except Exception:
        logger.exception("STREAM producer crashed")
    finally:
        enc = None
        if not cancel.is_set():
            _put_item(q, None, cancel)
        trim_memory()


def synthesize(text, lang, length_scale, pitch_factor):
    """(/voice کے لیے) غیر اسٹریمنگ، مکمل MP3۔"""
    audio = None
    engine = "google"
    is_en = (lang == "en")

    try:
        with SYNTH_LOCK:
            voice = get_voice("en" if is_en else "ur")
            if voice is not None:
                audio = generate_piper_mp3(text, length_scale, pitch_factor, voice)
                engine = "piper-en-female" if is_en else "piper-mp3"
            trim_memory()
    except Exception:
        audio = None
        logger.exception("PIPER SYNTHESIS FAILED")

    if audio is None:
        try:
            audio = generate_google_mp3(text, "en" if is_en else "ur")
            engine = "google-en" if is_en else "google"
        except Exception:
            logger.exception("GOOGLE TTS FAILED")
            raise

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
        return mp3_response(audio, engine)
    except Exception:
        logger.exception("ALL TTS ENGINES FAILED")
        return jsonify({"error": "TTS failed"}), 500


@app.route("/tts", methods=["POST"])
def tts():
    """فوراً لوٹتا ہے: متن محفوظ کر کے آئی ڈی دیتا ہے۔"""
    text = request.get_data(as_text=True).strip()
    lang = request.args.get("lang", "ur").strip().lower()
    rate_pct, pitch_hz, length_scale, pitch_factor = voice_params()

    if not text or len(text) > MAX_TEXT_CHARS:
        return Response("", status=400, mimetype="text/plain")

    job_id = job_put(text, lang, length_scale, pitch_factor)
    return Response(job_id, status=200, mimetype="text/plain")


def _build_full_audio(job, register=True, cancel=None):
    """پورا متن بنا کر ایک ہی MP3 (bytes) لوٹاتا ہے۔ register=True ہو تو پچھلی چلتی ہوئی جنریشن بند کر دیتا ہے۔"""
    global ACTIVE_CANCEL
    if cancel is None:
        cancel = threading.Event()
    if register:
        with ACTIVE_LOCK:
            if ACTIVE_CANCEL is not None:
                ACTIVE_CANCEL.set()
            ACTIVE_CANCEL = cancel

    q = queue.Queue(maxsize=STREAM_QUEUE_MAX)
    th = threading.Thread(target=_producer, args=(job, q, cancel), daemon=True)
    th.start()
    parts = []
    last = time.time()
    try:
        while True:
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                if cancel.is_set() or not th.is_alive():
                    break
                if time.time() - last > STREAM_IDLE_TIMEOUT:
                    break
                continue
            last = time.time()
            if item is None:
                break
            parts.append(item)
    finally:
        cancel.set()
    audio = b"".join(parts)
    del parts
    return audio


def _full_response(job):
    audio = _build_full_audio(job)
    if not audio:
        return Response("TTS failed", status=500, mimetype="text/plain")
    return mp3_response(audio, "piper-full")


# ─────────────────────────────────────────────
# اگلے جملے کی پیشگی تیاری (/speak)
# ESP32 موجودہ جملے کے ساتھ اگلے جملے کا متن بھی بھیجتا ہے۔ موجودہ جملہ تیار ہوتے ہی سرور اگلا
# پس منظر میں بنانا شروع کر دیتا ہے، تاکہ جب ESP32 اگلا مانگے تو وہ پہلے سے تیار ملے۔
# پیشگی تیاری موجودہ جملے کے بعد شروع ہوتی ہے، اس لیے پہلی آواز کو سست نہیں کرتی۔
# ─────────────────────────────────────────────
PRE = {}
PRE_LOCK = threading.Lock()
PRE_MAX = 2
PRE_TTL = 180


def _pre_key(text, lang, ls, pf):
    raw = "%s|%s|%.4f|%.4f" % (lang, text, ls, pf)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _pre_start(text, lang, ls, pf):
    key = _pre_key(text, lang, ls, pf)
    now = time.time()
    with PRE_LOCK:
        for k in [k for k, v in PRE.items() if now - v["t"] > PRE_TTL]:
            PRE[k]["cancel"].set()
            del PRE[k]
        if key in PRE:
            return
        while len(PRE) >= PRE_MAX:
            oldest = min(PRE, key=lambda k: PRE[k]["t"])
            PRE[oldest]["cancel"].set()
            del PRE[oldest]
        entry = {"event": threading.Event(), "audio": None, "t": now, "cancel": threading.Event()}
        PRE[key] = entry

    def run():
        try:
            entry["audio"] = _build_full_audio(
                {"text": text, "lang": lang, "ls": ls, "pf": pf},
                register=False, cancel=entry["cancel"])
            logger.info("PRE ready: %d bytes", len(entry["audio"] or b""))
        except Exception:
            logger.exception("PRE generation failed")
        finally:
            entry["event"].set()

    threading.Thread(target=run, daemon=True).start()


def _get_full_audio(job):
    key = _pre_key(job["text"], job["lang"], job["ls"], job["pf"])
    entry = None
    with PRE_LOCK:
        entry = PRE.pop(key, None)
        # نیا سوال: باقی پرانی پیشگی تیاریاں بند، CPU نئی درخواست کو ملے
        for k in list(PRE.keys()):
            PRE[k]["cancel"].set()
            del PRE[k]

    audio = None
    if entry is not None:
        entry["event"].wait(timeout=90)
        audio = entry["audio"]
        if audio:
            logger.info("PRE hit: served prepared audio (%d bytes)", len(audio))
    if not audio:
        audio = _build_full_audio(job)

    nxt = job.get("next", "")
    if audio and nxt:
        _pre_start(nxt, job["lang"], job["ls"], job["pf"])
    return audio


@app.route("/speak", methods=["POST"])
def speak():
    """ایک ہی درخواست میں متن بھیجو اور اسی جواب میں پوری MP3 (Content-Length کے ساتھ) واپس لو۔
    باڈی: 'موجودہ جملہ' یا 'موجودہ جملہ\x1eاگلا جملہ'۔"""
    body = request.get_data(as_text=True)
    text, _, nxt = body.partition("\x1e")
    text = text.strip()
    nxt = nxt.strip()[:MAX_TEXT_CHARS]
    lang = request.args.get("lang", "ur").strip().lower()
    rate_pct, pitch_hz, length_scale, pitch_factor = voice_params()

    if not text or len(text) > MAX_TEXT_CHARS:
        return Response("", status=400, mimetype="text/plain")

    job = {"text": text, "lang": lang, "ls": length_scale, "pf": pitch_factor, "next": nxt}
    audio = _get_full_audio(job)
    if not audio:
        return Response("TTS failed", status=500, mimetype="text/plain")
    return mp3_response(audio, "piper-speak")


@app.route("/audio/<audio_id>.mp3", methods=["GET"])
def audio_file(audio_id):
    """آڈیو بنتے ہی ٹکڑا بہ ٹکڑا بھیجتا ہے (chunked streaming)۔"""
    global ACTIVE_CANCEL
    job = job_get(audio_id)
    if not job:
        return Response("Audio expired or not found", status=404, mimetype="text/plain")

    # ?full=1 → پوری MP3 Content-Length کے ساتھ (ESP32 کے فلیش ڈاؤن لوڈ موڈ کے لیے، بغیر chunked)
    if request.args.get("full"):
        return _full_response(job)

    cancel = threading.Event()
    with ACTIVE_LOCK:
        if ACTIVE_CANCEL is not None:
            ACTIVE_CANCEL.set()      # پرانی اسٹریم بند: CPU نئی درخواست کو ملے
        ACTIVE_CANCEL = cancel

    q = queue.Queue(maxsize=STREAM_QUEUE_MAX)
    threading.Thread(target=_producer, args=(job, q, cancel), daemon=True).start()

    def generate():
        try:
            while True:
                try:
                    item = q.get(timeout=STREAM_IDLE_TIMEOUT)
                except queue.Empty:
                    break
                if item is None:
                    break
                yield item
        finally:
            cancel.set()             # کلائنٹ چلا گیا یا ختم: پروڈیوسر رک جائے

    return Response(
        generate(),
        mimetype="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "X-Luma-TTS": "piper-stream",
        },
    )


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
# 🔹 STT (Groq Whisper)
#    lang = "en" → صرف انگلش | "ur" → صرف اردو | "" → آٹو
#    آٹو میں انگلش نہ ہو تو اردو کے ساتھ دوبارہ، تاکہ اردو رسم الخط ملے۔
# ─────────────────────────────────────────────
def _groq_request(wav_bytes, lang, verbose):
    """Groq Whisper کو ایک کال۔ (text, detected_language) لوٹاتا ہے۔"""
    boundary = "----luma" + uuid.uuid4().hex

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = b""
    body += field("model", GROQ_STT_MODEL)
    body += field("response_format", "verbose_json" if verbose else "text")
    if lang:
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
        raw = resp.read().decode("utf-8", errors="replace").strip()

    del body

    if verbose:
        try:
            data = json.loads(raw)
            return str(data.get("text", "")).strip(), str(data.get("language", "")).strip().lower()
        except Exception:
            return raw, ""
    return raw, lang


def _send_to_groq_stt(pcm, lang=""):
    """پورے PCM کو WAV بنا کر Groq Whisper پر بھیجیں اور ٹیکسٹ لوٹائیں۔"""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    wav_bytes = pcm_to_wav(pcm, 16000)
    lang = (lang or "").strip().lower()

    if lang in ("en", "ur"):
        text, _ = _groq_request(wav_bytes, lang, False)
    else:
        text, detected = _groq_request(wav_bytes, "", True)
        logger.info("STT auto-detected language: %s", detected)
        if detected in ("english", "en"):
            pass
        elif detected in ("urdu", "ur") and text:
            pass
        else:
            text, _ = _groq_request(wav_bytes, "ur", False)

    del wav_bytes
    return text


@app.route("/stt", methods=["POST"])
def stt():
    if not GROQ_API_KEY:
        return Response("", status=500, mimetype="text/plain")

    pcm = request.get_data()
    if len(pcm) < 3200:
        return Response("", status=200, mimetype="text/plain")

    lang = request.args.get("lang", "").strip().lower()

    try:
        text = _send_to_groq_stt(pcm, lang=lang)
        del pcm
        return Response(text, status=200, mimetype="text/plain; charset=utf-8")
    except Exception:
        logger.exception("STT FAILED")

    return Response("", status=500, mimetype="text/plain")


# ─────────────────────────────────────────────
# 🔹 اسٹریمنگ STT (start / chunk / end)
# ─────────────────────────────────────────────
STT_SESSIONS = {}
STT_SESSIONS_LOCK = threading.Lock()
STT_SESSION_TTL = 60
STT_MAX_CHUNKS = 500  # ~500 * 4KB ≈ 2MB


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

    lang = request.args.get("lang", "").strip().lower()

    with STT_SESSIONS_LOCK:
        session = STT_SESSIONS.pop(session_id, None)

    if session is None:
        return Response("session_expired", status=404, mimetype="text/plain")

    pcm = b"".join(session["chunks"])
    total_chunks = len(session["chunks"])
    del session

    logger.info("STT end: id=%s chunks=%d bytes=%d", session_id, total_chunks, len(pcm))

    if len(pcm) < 3200:
        del pcm
        return Response("", status=200, mimetype="text/plain")

    try:
        text = _send_to_groq_stt(pcm, lang=lang)
        del pcm
        logger.info("STT end → text: %s", text[:80])
        return Response(text, status=200, mimetype="text/plain; charset=utf-8")
    except Exception:
        logger.exception("STT END FAILED")
        return Response("", status=500, mimetype="text/plain")


# Urdu ماڈل ریکوئسٹس سے پہلے لوڈ (انگلش پہلی انگلش درخواست پر)۔
load_piper()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
