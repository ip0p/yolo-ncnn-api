from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import io
from PIL import Image
import ncnn

# NCNN-Modell laden
param_path = "yolo11n_ncnn_model/model.ncnn.param"
bin_path = "yolo11n_ncnn_model/model.ncnn.bin"
net = ncnn.Net()
net.load_param(param_path)
net.load_model(bin_path)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "YOLOv11 NCNN API läuft!"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Bild in OpenCV-Format umwandeln
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # NCNN Inference
        mat = ncnn.Mat.from_pixels(image_cv, ncnn.Mat.PixelType.PIXEL_BGR, image_cv.shape[1], image_cv.shape[0])
        ex = net.create_extractor()
        ex.input("in0", mat)

        # Ausgabe-Blob extrahieren
        ret, out0 = ex.extract("out0")  # Verwenden Sie "out0" als Ausgabe-Blob

        # Ausgabe in ein Numpy-Array umwandeln
        out0_np = np.array(out0)

        # Ergebnisse verarbeiten
        detections = []
        num_boxes = out0_np.shape[0]
        for i in range(num_boxes):
            box = out0_np[i]
            x, y, w, h = box[0:4]  # Bounding-Box-Koordinaten
            confidence = box[4]     # Konfidenzniveau
            class_id = np.argmax(box[5:])  # Klassen-ID

            if confidence > 0.5:  # Nur Objekte mit hoher Konfidenz berücksichtigen
                detections.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "bbox": [float(x), float(y), float(w), float(h)]
                })

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
