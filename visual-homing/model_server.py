"""
Grounding DINO server (runs on your laptop's GPU).

Start:  uvicorn model_server:app --host 127.0.0.1 --port 8000
"""
import io

import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded once, when the server starts
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
print(f"Grounding DINO loaded on {device}")

app = FastAPI()


@app.get("/health")
def health():
    return {"model": MODEL_ID, "device": device}


@app.post("/detect")
def detect(image: UploadFile = File(...),
           prompts: str = Form(...),            # comma-separated, e.g. "a door, a car, a tree"
           threshold: float = Form(0.35),
           text_threshold: float = Form(0.25)):
    img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    labels = [p.strip() for p in prompts.split(",") if p.strip()]

    inputs = processor(images=img, text=[labels], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
        target_sizes=[img.size[::-1]])[0]

    names = result.get("text_labels", result.get("labels"))
    return [{"label": name, "score": round(float(score), 3),
             "box": [round(float(v)) for v in box.tolist()]}        # [x_min, y_min, x_max, y_max]
            for box, score, name in zip(result["boxes"], result["scores"], names)]
