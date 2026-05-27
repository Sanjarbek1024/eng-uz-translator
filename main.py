import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.translator import load_model, translate

app = FastAPI(title="EN to UZ Translator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, sp = load_model(
    checkpoint_path="D:\ML-projects\eng-uz-translator\saved_model\seq2seq_checkpoint.pt",
    spm_model_path="D:\ML-projects\eng-uz-translator\saved_model\spm.model",
    device=device
)


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translation: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(body: TranslateRequest):
    if not body.text.strip():
        return TranslateResponse(translation="")
    result = translate(body.text, model, sp, device)
    return TranslateResponse(translation=result)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}
