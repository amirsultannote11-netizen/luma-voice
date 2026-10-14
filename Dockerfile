FROM python:3.10-slim

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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir piper-tts && \
    python -c "import piper; print('[STATUS] Piper installed successfully')"

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium.onnx" -O ur_PK-aegis_female-medium.onnx && \
    wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium.onnx.json" -O ur_PK-aegis_female-medium.onnx.json

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -O en_US-lessac-medium.onnx && \
    wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -O en_US-lessac-medium.onnx.json

COPY . .

EXPOSE 10000
CMD ["python", "app.py"]
