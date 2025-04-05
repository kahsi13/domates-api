from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import os
import base64
import zipfile

app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ Domates API çalışıyor!"}

# Global değişkenler
tokenizer = None
session = None

# Dosya adları
MODEL_B64_PATH = "bert_model_base64.txt"
MODEL_ZIP_PATH = "bert_model.zip"
MODEL_PATH = "bert_domates_model_quant.onnx"

@app.on_event("startup")
def startup_event():
    global tokenizer, session

    # Varsa eski model dosyasını sil
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    # ✅ Base64 -> zip dosyası oluştur
    print("📥 Base64 model decode ediliyor...")
    try:
        with open(MODEL_B64_PATH, "rb") as f:
            decoded = base64.b64decode(f.read())
        with open(MODEL_ZIP_PATH, "wb") as f:
            f.write(decoded)
        print("✅ ZIP dosyası yazıldı.")
    except Exception as e:
        print(f"❌ Base64 decode hatası: {e}")
        return

    # ✅ Zip -> onnx dosyasını çıkar
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ .onnx model çıkarıldı.")
    except Exception as e:
        print(f"❌ ZIP çıkarma hatası: {e}")
        return

    # ✅ Tokenizer yükle
    try:
        print("🔤 Tokenizer yükleniyor...")
        tokenizer = AutoTokenizer.from_pretrained("Kahsi13/DomatesRailway")
        print("✅ Tokenizer yüklendi.")
    except Exception as e:
        print(f"❌ Tokenizer yüklenemedi: {e}")
        return

    # ✅ ONNX modeli yükle
    try:
        print("📦 ONNX modeli yükleniyor...")
        session = onnxruntime.InferenceSession(MODEL_PATH)
        print("✅ Model başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")

# Giriş modeli
class InputText(BaseModel):
    text: str

# Tahmin endpoint'i
@app.post("/predict")
def predict(input: InputText):
    try:
        print("🧪 Gelen veri:", input)

        if tokenizer is None or session is None:
            return {"error": "⏳ Model yükleniyor, lütfen birazdan tekrar deneyin."}

        encoding = tokenizer.encode_plus(
            input.text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        ort_outs = session.run(None, ort_inputs)
        prediction = int(np.argmax(ort_outs[0]))

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
