from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import io
from PIL import Image
import ncnn

# NCNN-Modell laden
param_path = "yolo11n_ncnn_model/yolo11n_ncnn.param"
bin_path = "yolo11n_ncnn_model/yolo11n_ncnn.bin"
net = ncnn.Net()
net.load_param(param_path)
net.load_model(bin_path)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "YOLOv11 NCNN API läuft!"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Bild in OpenCV-Format umwandeln
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # NCNN Inference
    mat = ncnn.Mat.from_pixels(image_cv, ncnn.Mat.PixelType.PIXEL_BGR, image_cv.shape[1], image_cv.shape[0])
    ex = net.create_extractor()
    ex.input("data", mat)

    # Ergebnisse abrufen
    detections = []
    for i in range(10):  # Annahme: maximal 10 Objekte
        ret, out = ex.extract(f"output_{i}")
        if ret == 0:
            bbox = [float(out[i]) for i in range(4)]
            confidence = float(out[4])
            class_id = int(out[5])
            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "bbox": bbox
            })

    return {"detections": detections}