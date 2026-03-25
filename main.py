from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import faiss, json, os, numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("islenmis_veri.json", "r", encoding="utf-8") as f:
    VERI = json.load(f)

MODELLER, INDEKSLER = {}, {}

def model_getir(model_adi: str):
    # Eğer gemini_mock seçilirse, e5-large modelini taklit ediyoruz
    gercek_model = "intfloat/multilingual-e5-large" if model_adi == "gemini_mock" else model_adi
    
    if gercek_model not in MODELLER:
        MODELLER[gercek_model] = SentenceTransformer(gercek_model)
    
    # İŞTE DÜZELTİLEN KISIM BURASI: model_adi yerine gercek_model'in .faiss dosyasını okuyoruz!
    dosya = f"index_{gercek_model.replace('/', '_')}.faiss"
    
    if dosya not in INDEKSLER:
        INDEKSLER[dosya] = faiss.read_index(dosya)
    return MODELLER[gercek_model], INDEKSLER[dosya]

@app.get("/")
async def ana_sayfa(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
async def arama(query: str, model_name: str):
    try:
        model, index = model_getir(model_name)
        vektor = model.encode([query]).astype('float32')
        mesafeler, indisler = index.search(vektor, 1)
        
        yuzde = round((1 / (1 + float(mesafeler[0][0]))) * 100, 2)
        
        return {
            "status": "success", 
            "data": {
                "question": VERI[indisler[0][0]]["question"], 
                "similarity_score": yuzde,
                "full_json": VERI[indisler[0][0]]
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}