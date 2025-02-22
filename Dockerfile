# Basisimage für eine ARM-Architektur, z.B. Ubuntu 22.04
FROM ubuntu:22.04

# Umgebungsvariablen für nicht-interaktive Installationen
ENV DEBIAN_FRONTEND=noninteractive

# Aktualisieren der Paketquellen und Installation von Python 3 und pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install -U pip

RUN apt-get install -y git-svn

# Installation des ultralytics-Pakets mit optionalen Abhängigkeiten für den Export
RUN pip3 install ultralytics[export]

RUN pip3 install fastapi uvicorn pillow numpy opencv-python-headless ncnn python-multipart setuptools

# Setze Arbeitsverzeichnis
WORKDIR /ultralytics

# Kopiere API-Skript ins Image
COPY api_ncnn.py /ultralytics/api_ncnn.pyc
COPY egg.pt /ultralytics/egg.pt

# Öffne den API-Port
EXPOSE 8863

# Starte die API mit NCNN
CMD ["uvicorn", "api_ncnn:app", "--host", "0.0.0.0", "--port", "8863"]
