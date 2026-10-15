import os
import io
import wave
import logging
import urllib.request

from flask import Flask, request, Response, jsonify
from piper import PiperVoice
from gtts import gTTS

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LumaVoice")

MODEL_FILENAME = "ur_PK-aegis_female-medium.onnx"
CONFIG_FILENAME = "ur_PK-aegis_female-medium.onnx.json"

BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "ur/ur_PK/aegis_female/medium/"
)

MODEL_URL = BASE_URL + MODEL_FILENAME
CONFIG_URL = BASE_URL + CONFIG_FILENAME

piper_voice = None


def download_file(filename, url):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        logger.info("File already exists: %s", filename)
        return

    temp_file = filename + ".tmp"
    logger.info("Downloading %s", filename)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LumaVoice/1.0"}
        )

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

        piper_voice = PiperVoice.load(
            MODEL_FILENAME,
            config_path=CONFIG_FILENAME
        )

        logger.info("PIPER VOICE LOADED SUCCESSFULLY")

    except Exception:
        piper_voice = None
        logger.exception("PIPER MODEL LOAD FAILED")


def generate_piper_wav(text):
    if piper_voice is None:
        raise RuntimeError("Piper voice is not loaded")

    wav_io = io.BytesIO()

    # Piper writes WAV parameters and audio frames.
    with wave.open(wav_io, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)

    audio_data = wav_io.getvalue()

    if len(audio_data) <= 44:
        raise RuntimeError("Piper generated an empty WAV")

    # Validate the resulting WAV.
    with wave.open(io.BytesIO(audio_data), "rb") as check:
        if check.getnchannels() < 1 or check.getnframes() < 1:
            raise RuntimeError("Invalid WAV audio")

    return audio_data


def generate_google_mp3(text):
    output = io.BytesIO()

    tts = gTTS(text=text, lang="ur", slow=False)
    tts.write_to_fp(output)

    audio_data = output.getvalue()

    if not audio_data:
        raise RuntimeError("Google TTS generated empty audio")

    return audio_data


@app.route("/", methods=["GET"])
def home():
    return (
        "Luma Voice Server Active. "
        "Test: /voice?text=Assalam%20o%20Alaikum"
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "server": "online",
        "piper_loaded": piper_voice is not None
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

    if not text:
        return jsonify({
            "error": "No text provided",
            "example": "/voice?text=Assalam%20o%20Alaikum"
        }), 400

    if len(text) > 2000:
        return jsonify({"error": "Text too long"}), 413

    # Try Piper first.
    if piper_voice is not None:
        try:
            audio = generate_piper_wav(text)

            logger.info(
                "PIPER SUCCESS: %d bytes",
                len(audio)
            )

            return Response(
                audio,
                mimetype="audio/wav",
                headers={
                    "Content-Length": str(len(audio)),
                    "Cache-Control": "no-store",
                    "X-Luma-TTS": "piper"
                }
            )

        except Exception:
            logger.exception("PIPER SYNTHESIS FAILED")

    else:
        logger.error("Piper unavailable; trying Google TTS")

    # Use Google only if Piper fails.
    try:
        audio = generate_google_mp3(text)

        logger.info(
            "GOOGLE TTS SUCCESS: %d bytes",
            len(audio)
        )

        return Response(
            audio,
            mimetype="audio/mpeg",
            headers={
                "Content-Length": str(len(audio)),
                "Cache-Control": "no-store",
                "X-Luma-TTS": "google"
            }
        )

    except Exception:
        logger.exception("GOOGLE TTS ALSO FAILED")

        return jsonify({
            "error": "Both TTS engines failed",
            "details": "Check Render logs"
        }), 500


# Load the model before accepting requests.
load_piper()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
