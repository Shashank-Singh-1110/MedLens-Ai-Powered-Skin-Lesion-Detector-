import sys
import io
import json
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from PIL import Image
import numpy as np

from backend.utils.inference import load_model, predict, transform
from backend.utils.prompt_adapter import build_prompt
from backend.utils.llm import generate_response, generate_response_sync
from backend.utils.gradcam import GradCAM, overlay_heatmap
from backend.utils.report import generate_report
from config import CLASS_LIST

app = FastAPI(title="MedLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, device = load_model()
cam = GradCAM(model, device)


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}

    result = predict(model, device, image)

    pred_idx = CLASS_LIST.index(result["class_code"])
    img_array = np.array(image.resize((224, 224)))
    input_tensor = transform(image=img_array)["image"].unsqueeze(0)
    heatmap = cam.generate(input_tensor, pred_idx)
    overlay = overlay_heatmap(image, heatmap)
    system_prompt, user_prompt = build_prompt(result)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'classification', 'class_code': result['class_code'], 'class_name': result['class_name'], 'confidence': result['confidence'], 'severity': result['severity'], 'urgency': result['urgency'], 'all_predictions': result['all_predictions']})}\n\n"
        yield f"data: {json.dumps({'type': 'heatmap', 'image': image_to_base64(overlay)})}\n\n"
        async for chunk in generate_response(system_prompt, user_prompt):
            yield f"data: {json.dumps({'type': 'llm_chunk', 'content': chunk})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/report")
async def report(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}

    result = predict(model, device, image)

    pred_idx = CLASS_LIST.index(result["class_code"])
    img_array = np.array(image.resize((224, 224)))
    input_tensor = transform(image=img_array)["image"].unsqueeze(0)
    heatmap = cam.generate(input_tensor, pred_idx)
    overlay = overlay_heatmap(image, heatmap)

    system_prompt, user_prompt = build_prompt(result)
    explanation = generate_response_sync(system_prompt, user_prompt)

    pdf_buffer = generate_report(image, overlay, result, explanation)

    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=MedLens_Report.pdf"},
    )
@app.get("/nearby-dermatologists")
async def nearby_dermatologists(lat: float, lng: float):
    import httpx
    return {
        "search_url": f"https://www.google.com/maps/search/dermatologist/@{lat},{lng},13z",
        "lat": lat,
        "lng": lng,
    }