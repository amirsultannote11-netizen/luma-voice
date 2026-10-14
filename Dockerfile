FROM python:3.10-slim

# ضروری سسٹم پیکیجز انسٹال کریں
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# لائبریریز انسٹال کریں
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Piper Urdu (Aegis Female) اور English Voices ڈاؤن لوڈ کریں ───
# نوٹ: فولڈر کا نام 'aegis_female' ہے اور فائل کا نام 'ur_PK-aegis_female-medium.onnx' ہے

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium.onnx" -O ur_PK-aegis_female-medium.onnx

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium.onnx.json" -O ur_PK-aegis_female-medium.onnx.json

# ─── English Voice (Lessac Medium) ───
RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -O en_US-lessac-medium.onnx

RUN wget --tries=5 --timeout=60 -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -O en_US-lessac-medium.onnx.json

# باقی پروجیکٹ کا کوڈ کاپی کریں
COPY . .

EXPOSE 10000
CMD ["python", "app.py"]
