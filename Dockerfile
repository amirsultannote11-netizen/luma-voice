FROM python:3.10-slim

# ─── ضروری سسٹم پیکجز (espeak-ng سمیت) ───
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    ffmpeg \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── پائتھون لائبریریاں انسٹال کریں ───
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Piper TTS انجن انسٹال کریں اور تصدیق کریں ───
RUN pip install --no-cache-dir piper-tts && \
    python -c "import piper; print('[STATUS] Piper installed successfully')"

# ─── Piper Urdu (Aegis Female) وائس ڈاؤن لوڈ کریں ───
RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium.onnx" -O ur_PK-aegis_female-medium.onnx

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium.onnx.json" -O ur_PK-aegis_female-medium.onnx.json

# ─── Piper English (Lessac Medium) وائس ڈاؤن لوڈ کریں ───
RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -O en_US-lessac-medium.onnx

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -O en_US-lessac-medium.onnx.json

# ─── باقی کوڈ کاپی کریں ───
COPY . .

EXPOSE 10000
CMD ["python", "app.py"]
