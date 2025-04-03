import onnx

try:
    model = onnx.load("bert_domates_model_quant.onnx")
    onnx.checker.check_model(model)
    print("✅ Model geçerli ve sağlam.")
except Exception as e:
    print("❌ Model bozuk veya geçersiz:", e)
