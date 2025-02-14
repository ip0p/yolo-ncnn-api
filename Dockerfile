# Basis-Image (Debian für ARM64)
FROM arm64v8/debian:bookworm-slim

# Setze Umgebungsvariablen
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Installiere Linux-Pakete
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip git zip unzip wget curl gcc g++ \
    libgl1 libglib2.0-0 libpython3-dev libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setze Arbeitsverzeichnis
WORKDIR /ultralytics

# Klone Ultralytics YOLO
RUN git clone --depth=1 https://github.com/ultralytics/ultralytics.git .

# Installiere Python-Abhängigkeiten
RUN pip install uv
RUN uv pip install --system -e ".[export]" --break-system-packages

# Installiere FastAPI, OpenCV und NCNN-Python-Wrapper
RUN pip install fastapi uvicorn pillow numpy opencv-python-headless ncnn

# YOLO-Modell herunterladen und in NCNN exportieren
RUN yolo download model=yolo11n.pt && \
    yolo export model=yolo11n.pt format=ncnn

# Kopiere API-Skript ins Image
COPY api_ncnn.py /ultralytics/api_ncnn.py

# Öffne den API-Port
EXPOSE 8000

# Starte die API mit NCNN
CMD ["uvicorn", "api_ncnn:app", "--host", "0.0.0.0", "--port", "8000"]