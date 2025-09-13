import requests

model_url = "https://github.com/ondyari/FaceForensics/raw/master/models/xception_model.h5"
model_path = "deepfake_model_real.h5"

print("Downloading model, please wait...")
r = requests.get(model_url, allow_redirects=True)
open(model_path, 'wb').write(r.content)
print("Model downloaded as deepfake_model_real.h5")
