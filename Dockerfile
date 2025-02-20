# Verwende direkt das Ultralytics-YOLO-Image für ARM64
FROM ultralytics/ultralytics:latest-arm64

# Setze Arbeitsverzeichnis
WORKDIR /ultralytics

# Installiere benötigte Pakete
RUN pip install fastapi uvicorn pillow numpy opencv-python-headless ncnn python-multipart

# Modell-Pfad setzen
ENV YOLO_MODEL=yolov11-world.pt

# Prüfe und lade das YOLO-World Modell nur, wenn es nicht existiert
RUN test -f "$YOLO_MODEL" || yolo download model=ultralytics/yolov11-world

# Konvertiere Modell zu NCNN (nur wenn nötig)
RUN test -f yolov11-world_ncnn_model/model.ncnn.param || yolo export model="$YOLO_MODEL" format=ncnn

# Kopiere API-Skript ins Image
COPY api_ncnn.py /ultralytics/api_ncnn.py

# Öffne den API-Port
EXPOSE 8863

# Starte die API mit NCNN
CMD ["uvicorn", "api_ncnn:app", "--host", "0.0.0.0", "--port", "8863"]
