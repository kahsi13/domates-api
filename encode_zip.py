import base64

with open("bert_model.zip", "rb") as zip_file:
    encoded = base64.b64encode(zip_file.read())

with open("bert_model_base64.txt", "wb") as out_file:
    out_file.write(encoded)

print("✅ Encode tamam! bert_model_base64.txt oluşturuldu.")
