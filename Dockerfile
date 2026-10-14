FROM python:3.10-slim

# जरूरी सिस्टम पैकेज इंस्टॉल करें
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# लाइब्ररीज इंस्टॉल करें
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Piper Urdu (Aegis) और English Voices डाउनलोड करें ───
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis/medium/ur_PK-aegis-medium.onnx" -O ur_PK-aegis_female-medium.onnx
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis/medium/ur_PK-aegis-medium.onnx.json" -O ur_PK-aegis_female-medium.onnx.json

RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -O en_US-lessac-medium.onnx
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -O en_US-lessac-medium.onnx.json

# बाकी कोड कॉपी करें
COPY . .

EXPOSE 10000
CMD ["python", "app.py"]
