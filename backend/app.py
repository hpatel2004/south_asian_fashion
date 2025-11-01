# backend/app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.model.inference import load_model, transform
from PIL import Image
import torch

app = FastAPI()

# Allow frontend (React) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device=device)

@app.get("/")
def home():
    return {"message": "Fashion classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)

    label = "Couture" if preds.item() == 0 else "Knockoff"
    return {"prediction": label}
