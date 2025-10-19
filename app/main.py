import os, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from helper_lib import get_model, preprocess, predict, CIFAR10_CLASSES

WEIGHTS = os.getenv("WEIGHTS_PATH", "models/cnn_cifar10.pth")
app = FastAPI(title="Assignment2 CNN API")

@app.on_event("startup")
def startup():
    if not os.path.exists(WEIGHTS):
        raise RuntimeError(f"Missing weights at {WEIGHTS}")
    global MODEL, DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = get_model("CNN", num_classes=len(CIFAR10_CLASSES))
    MODEL.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    MODEL.eval().to(DEVICE)

@app.get("/health")
def health(): return {"status": "ok", "weights": WEIGHTS}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        x = preprocess(await file.read())
        label, conf, probs = predict(MODEL, DEVICE, x)
        return {"label": label, "confidence": round(conf, 4), "classes": CIFAR10_CLASSES, "probs": probs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"ok": True, "try": ["/health", "/docs", "/predict (POST file)"]}
