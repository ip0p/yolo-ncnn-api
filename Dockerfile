# Verwende direkt das Ultralytics-YOLO-Image für ARM64
FROM ultralytics/ultralytics:latest-arm64

# Setze Arbeitsverzeichnis
WORKDIR /ultralytics

# Installiere benötigte Pakete
RUN pip install fastapi uvicorn pillow numpy opencv-python-headless ncnn python-multipart

# Kopiere API-Skript ins Image
COPY api_ncnn.py /ultralytics/api_ncnn.py

# Öffne den API-Port
EXPOSE 8863

# Starte die API mit NCNN
CMD ["uvicorn", "api_ncnn:app", "--host", "0.0.0.0", "--port", "8863"]
