from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import os
import base64

app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ Domates API çalışıyor!"}

# Global değişkenler
tokenizer = None
session = None

# Dosya yolları
MODEL_B64_PATH = "bert_model_base64.txt"
MODEL_PATH = "bert_domates_model_quant.onnx"

@app.on_event("startup")
def startup_event():
    global tokenizer, session

    # Önce varsa eski model dosyasını sil
    if os.path.exists(MODEL_PATH):
        print("🧹 Eski model dosyası siliniyor...")
        os.remove(MODEL_PATH)

    # Base64'ten modeli oluştur
    print("📥 Base64 model dosyası çözümleniyor...")
    try:
        with open(MODEL_B64_PATH, "rb") as encoded_file:
            encoded_data = encoded_file.read()
            with open(MODEL_PATH, "wb") as model_file:
                model_file.write(base64.b64decode(encoded_data))
        print("✅ Model başarıyla oluşturuldu.")
    except Exception as e:
        print(f"❌ Decode hatası: {e}")
        return

    # Hugging Face'ten tokenizer yükle
    try:
        print("🔤 Tokenizer yükleniyor...")
        tokenizer = AutoTokenizer.from_pretrained("Kahsi13/DomatesRailway")
        print("✅ Tokenizer yüklendi.")
    except Exception as e:
        print(f"❌ Tokenizer yüklenemedi: {e}")
        return

    # ONNX modelini yükle
    try:
        print("📦 Model onnxruntime ile yükleniyor...")
        session = onnxruntime.InferenceSession(MODEL_PATH)
        print("✅ Model başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")

# Kullanıcıdan gelen metin yapısı
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    try:
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
