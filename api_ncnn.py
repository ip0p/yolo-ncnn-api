from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import io
import base64
from PIL import Image, ImageDraw
import os
from ultralytics import YOLO

app = FastAPI()

# Pfade für das Modell
MODEL_PT_PATH = "yolo11n.pt"
MODEL_NCNN_PATH = "yolo11n_ncnn_model"

# Prüfen, ob das NCNN-Modell existiert, sonst exportieren
if not os.path.exists(MODEL_NCNN_PATH):
    print("NCNN-Modell nicht gefunden. Exportiere jetzt...")
    model = YOLO(MODEL_PT_PATH)
    model.export(format="ncnn")

# Exportiertes NCNN-Modell laden
ncnn_model = YOLO(MODEL_NCNN_PATH)


@app.get("/")
def read_root():
    return {"message": "YOLOv11 NCNN API läuft mit Ultralytics!"}


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Bild als PIL-Image öffnen
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # YOLO Inferenz ausführen
        results = ncnn_model(image)

        # Bounding-Boxes auf das Bild zeichnen
        draw = ImageDraw.Draw(image)
        detections = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = map(float, box.xywh[0])  # Sicherstellen, dass Werte floats sind
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                class_id = int(box.cls.item())  # Integer-Klassen-ID
                confidence = float(box.conf.item())  # Float-Konfidenz

                # Bounding-Box zeichnen
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), f"ID: {class_id} ({confidence:.2f})", fill="red")

                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        # Bild mit Bounding-Boxes als Base64 encodieren
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        return {"detections": detections, "image": img_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/upload/", response_class=HTMLResponse)
def upload_page():
    """Web-Interface zum Hochladen von Bildern mit Vorschau und Bounding-Boxes."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv11 NCNN Web Interface</title>
        <script>
            async function uploadImage(event) {
                event.preventDefault();
                let formData = new FormData();
                formData.append("file", document.getElementById("file").files[0]);

                let response = await fetch("/detect/", {
                    method: "POST",
                    body: formData
                });

                let result = await response.json();
                document.getElementById("results").innerText = JSON.stringify(result.detections, null, 2);

                // Bild mit Bounding-Boxes anzeigen
                let image = new Image();
                image.src = "data:image/png;base64," + result.image;
                image.style.maxWidth = "500px";
                document.getElementById("image-preview").innerHTML = "";
                document.getElementById("image-preview").appendChild(image);
            }

            function previewImage(event) {
                let reader = new FileReader();
                reader.onload = function(){
                    let output = document.getElementById("original-image");
                    output.src = reader.result;
                    output.style.display = "block";
                }
                reader.readAsDataURL(event.target.files[0]);
            }
        </script>
    </head>
    <body>
        <h2>YOLOv11 NCNN Web Interface</h2>
        <form onsubmit="uploadImage(event)">
            <input type="file" id="file" accept="image/*" onchange="previewImage(event)" required>
            <button type="submit">Bild hochladen</button>
        </form>

        <h3>Originalbild:</h3>
        <img id="original-image" style="max-width: 500px; display: none;"/>

        <h3>Erkannte Objekte:</h3>
        <pre id="results"></pre>

        <h3>Ergebnisbild mit Bounding-Boxes:</h3>
        <div id="image-preview"></div>
    </body>
    </html>
    """
