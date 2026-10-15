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

MODEL_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "ur/ur_PK/aegis_female/medium/"
    "ur_PK-aegis_female-medium.onnx"
)

CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "ur/ur_PK/aegis_female/medium/"
    "ur_PK-aegis_female-medium.onnx.json"
)

piper_voice = None


def download_file_if_missing(filename, url):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        logger.info("%s already exists", filename)
        return

    logger.info("Downloading %s", filename)
    temp_filename = filename + ".download"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LumaVoiceServer/1.0"}
        )

        with urllib.request.urlopen(req, timeout=180) as response:
            with open(temp_filename, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if os.path.getsize(temp_filename) == 0:
            raise RuntimeError("Downloaded file is empty")

        os.replace(temp_filename, filename)
        logger.info("Download completed: %s", filename)

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def load_piper():
    global piper_voice

    try:
        download_file_if_missing(MODEL_FILENAME, MODEL_URL)
        download_file_if_missing(CONFIG_FILENAME, CONFIG_URL)

        piper_voice = PiperVoice.load(
            MODEL_FILENAME,
            config_path=CONFIG_FILENAME
        )

        logger.info("Piper Urdu female voice loaded successfully")

    except Exception:
        piper_voice = None
        logger.exception("Piper model could not be loaded")


# Load the voice when the server starts
load_piper()


@app.route("/", methods=["GET"])
def home():
    return (
        "Luma Voice Server Active! "
        "Test voice at /voice?text=Assalam%20o%20Alaikum"
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "server": "online",
        "piper_loaded": piper_voice is not None,
        "google_backup": True
    })


def generate_piper_wav(text):
    wav_io = io.BytesIO()

    # Piper writes a complete WAV file into this stream.
    with wave.open(wav_io, "wb") as wav_file:
        piper_voice.synthesize(text, wav_file)

    audio_data = wav_io.getvalue()

    if len(audio_data) <= 44:
        raise RuntimeError(
            f"Piper returned an empty or invalid WAV: {len(audio_data)} bytes"
        )

    return audio_data


def generate_google_mp3(text):
    mp3_io = io.BytesIO()

    tts = gTTS(
        text=text,
        lang="ur",
        slow=False
    )

    tts.write_to_fp(mp3_io)
    audio_data = mp3_io.getvalue()

    if not audio_data:
        raise RuntimeError("Google TTS returned empty audio")

    return audio_data


@app.route("/voice", methods=["GET", "POST"])
def text_to_speech():
    if request.method == "POST":
        user_text = (
            request.form.get("text", "")
            or request.args.get("text", "")
        )
        if not user_text and request.is_json:
            user_text = (request.json or {}).get("text", "")
    else:
        user_text = request.args.get("text", "")

    user_text = user_text.strip()

    if not user_text:
        return jsonify({
            "error": "No text provided",
            "example": "/voice?text=Assalam%20o%20Alaikum"
        }), 400

    # Prevent excessively large requests
    if len(user_text) > 2000:
        return jsonify({"error": "Text is too long"}), 413

    # 1. Try Piper Urdu female voice
    if piper_voice is not None:
        try:
            audio_data = generate_piper_wav(user_text)

            logger.info(
                "Piper generated %s bytes for %s characters",
                len(audio_data),
                len(user_text)
            )

            return Response(
                audio_data,
                status=200,
                mimetype="audio/wav",
                headers={
                    "Content-Length": str(len(audio_data)),
                    "Cache-Control": "no-store",
                    "X-Luma-TTS": "piper"
                }
            )

        except Exception:
            logger.exception("Piper synthesis failed; trying Google TTS")

    # 2. Google TTS backup
    try:
        audio_data = generate_google_mp3(user_text)

        logger.info(
            "Google TTS generated %s bytes",
            len(audio_data)
        )

        return Response(
            audio_data,
            status=200,
            mimetype="audio/mpeg",
            headers={
                "Content-Length": str(len(audio_data)),
                "Cache-Control": "no-store",
                "X-Luma-TTS": "google"
            }
        )

    except Exception:
        logger.exception("Google TTS also failed")

        return jsonify({
            "error": "Both Piper and Google TTS failed",
            "message": "Please check Render Logs"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
